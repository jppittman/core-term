use crate::ansi::commands::AnsiCommand;
use crate::config::Config;
use crate::keys;
use crate::surface::terminal::GlyphCache;
use crate::surface::{GridBuffer, TerminalSurface};
use crate::term::{
    ControlEvent, EmulatorAction, EmulatorInput, TerminalEmulator, TerminalSnapshot,
    UserInputAction,
};
use pixelflow_core::pipe::Surface;
use pixelflow_engine::input::MouseButton;
use pixelflow_engine::{AppAction, AppState, Application, EngineEvent};
use pixelflow_render::{font, glyphs_scaled, BakedMask, Memoize};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;

/// The terminal application.
pub struct TerminalApp {
    /// The terminal emulator - owns all terminal state.
    emulator: TerminalEmulator,
    /// Channel to receive parsed ANSI commands from the PTY read thread.
    pty_rx: Receiver<Vec<AnsiCommand>>,
    /// Channel to send data to the PTY write thread.
    pty_tx: SyncSender<Vec<u8>>,
    /// Application configuration.
    config: Config,
    /// Glyph cache for rendering.
    glyph_cache: GlyphCache,
}

impl TerminalApp {
    /// Creates a new TerminalApp.
    pub fn new(
        emulator: TerminalEmulator,
        pty_rx: Receiver<Vec<AnsiCommand>>,
        pty_tx: SyncSender<Vec<u8>>,
        config: Config,
    ) -> Self {
        let f = font();

        let w = config.appearance.cell_width_px as u32;
        let h = config.appearance.cell_height_px as u32;
        let font_size = config.appearance.font.size_pt as f32;

        let glyph_cache: GlyphCache = Memoize::new(Arc::new(move |c| {
            let g = glyphs_scaled(f, font_size)(c);
            Arc::new(BakedMask::new(&g, w, h))
        }));

        Self {
            emulator,
            pty_rx,
            pty_tx,
            config,
            glyph_cache,
        }
    }

    /// Maximum number of command batches to process per drain cycle.
    const MAX_BATCHES_PER_DRAIN: usize = 10;

    fn drain_pty(&mut self) -> Vec<EmulatorAction> {
        let mut actions = Vec::new();
        for _ in 0..Self::MAX_BATCHES_PER_DRAIN {
            match self.pty_rx.try_recv() {
                Ok(commands) => {
                    for cmd in commands {
                        if let Some(action) = self.emulator.interpret_input(EmulatorInput::Ansi(cmd)) {
                            actions.push(action);
                        }
                    }
                }
                Err(_) => break, // Channel empty or disconnected
            }
        }
        actions
    }

    fn process_engine_event(&mut self, event: EngineEvent) -> Vec<EmulatorAction> {
        let mut actions = Vec::new();

        let emulator_input = match event {
            EngineEvent::Resize(width_px, height_px) => {
                log::info!(
                    "TerminalApp: Resize {}x{} logical px",
                    width_px,
                    height_px
                );
                Some(EmulatorInput::Control(ControlEvent::Resize {
                    width_px: width_px as u16,
                    height_px: height_px as u16,
                }))
            }
            EngineEvent::KeyDown { key, mods, text } => {
                log::debug!("TerminalApp: Key: {:?} + {:?}, text: {:?}", mods, key, text);
                let key_input_action = keys::map_key_event_to_action(key, mods, &self.config)
                    .unwrap_or(UserInputAction::KeyInput {
                        symbol: key,
                        modifiers: mods,
                        text,
                    });
                Some(EmulatorInput::User(key_input_action))
            }
            EngineEvent::MouseClick { button, x, y } => match button {
                MouseButton::Left => Some(EmulatorInput::User(UserInputAction::StartSelection {
                    x_px: x as u16,
                    y_px: y as u16,
                })),
                MouseButton::Middle => {
                    Some(EmulatorInput::User(UserInputAction::RequestPrimaryPaste))
                }
                _ => None,
            },
            EngineEvent::MouseRelease { button, .. } => {
                if button == MouseButton::Left {
                    Some(EmulatorInput::User(UserInputAction::ApplySelectionClear))
                } else {
                    None
                }
            }
            EngineEvent::MouseMove { x, y, .. } => {
                Some(EmulatorInput::User(UserInputAction::ExtendSelection {
                    x_px: x as u16,
                    y_px: y as u16,
                }))
            }
            EngineEvent::FocusGained => Some(EmulatorInput::User(UserInputAction::FocusGained)),
            EngineEvent::FocusLost => Some(EmulatorInput::User(UserInputAction::FocusLost)),
            EngineEvent::Paste(text) => {
                Some(EmulatorInput::User(UserInputAction::PasteText(text)))
            }
            EngineEvent::Wake | EngineEvent::CloseRequested => None,
        };

        if let Some(input) = emulator_input {
            if let Some(action) = self.emulator.interpret_input(input) {
                actions.push(action);
            }
        }

        actions
    }

    fn handle_emulator_actions(&mut self, actions: Vec<EmulatorAction>) -> AppAction {
        for action in actions {
            match action {
                EmulatorAction::WritePty(data) => {
                    if self.pty_tx.send(data).is_err() {
                        log::warn!("TerminalApp: Failed to send to PTY");
                    }
                }
                EmulatorAction::SetTitle(title) => {
                    return AppAction::SetTitle(title);
                }
                EmulatorAction::CopyToClipboard(text) => {
                    return AppAction::CopyToClipboard(text);
                }
                EmulatorAction::RequestClipboardContent => {
                    return AppAction::RequestPaste;
                }
                EmulatorAction::ResizePty { cols, rows } => {
                    log::info!("TerminalApp: PTY resize request {}x{}", cols, rows);
                    // TODO: Send resize to PTY
                }
                EmulatorAction::RingBell => {
                    log::debug!("TerminalApp: Bell");
                    // TODO: Implement bell notification
                }
                EmulatorAction::RequestRedraw => {
                    // In pull model, emulator's dirty line tracking handles this
                    // No action needed
                }
                EmulatorAction::SetCursorVisibility(_) => {
                    // TODO: Implement cursor visibility
                }
            }
        }
        AppAction::Continue
    }
}

impl Application for TerminalApp {
    fn on_event(&mut self, event: EngineEvent) -> AppAction {
        let pty_actions = self.drain_pty();
        let _ = self.handle_emulator_actions(pty_actions);

        match event {
            EngineEvent::Wake => AppAction::Continue,
            EngineEvent::CloseRequested => AppAction::Quit,
            other => {
                let actions = self.process_engine_event(other);
                self.handle_emulator_actions(actions)
            }
        }
    }

    fn render(&mut self, _state: &AppState) -> Option<Box<dyn Surface<u32> + Send + Sync>> {
        let snapshot = self.emulator.get_render_snapshot()?;

        if !snapshot.lines.iter().any(|line| line.is_dirty) {
            return None;
        }

        let default_fg: u32 = self.config.colors.foreground.into();
        let default_bg: u32 = self.config.colors.background.into();

        let grid = GridBuffer::from_snapshot(&snapshot, default_fg, default_bg);

        let terminal = TerminalSurface {
            grid,
            cell_width: self.config.appearance.cell_width_px,
            cell_height: self.config.appearance.cell_height_px,
            glyph_cache: self.glyph_cache.clone(),
        };

        Some(Box::new(terminal))
    }
}
