//! Terminal application Actor implementation.
//!
//! Implements `Actor<EngineEventData, EngineEventControl, EngineEventManagement>` to receive
//! engine events and send responses back via the engine handle.

use crate::ansi::commands::AnsiCommand;
use crate::color::Color;
use crate::config::Config;
use crate::keys;
use crate::surface::{GridBuffer, TerminalSurface};
use crate::term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalEmulator, UserInputAction};
use actor_scheduler::{Actor, ActorScheduler, Message, ParkHint};
use core::marker::PhantomData;
use pixelflow_core::surfaces::Baked;
use pixelflow_core::traits::Surface;
use pixelflow_graphics::fonts::{glyphs, Lazy};
use pixelflow_graphics::render::{font, Pixel};
use pixelflow_runtime::input::MouseButton;
use pixelflow_runtime::{
    AppData, AppManagement, EngineActorHandle, EngineEventControl, EngineEventData,
    EngineEventManagement,
};

use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;

/// Glyph factory type - closure that returns lazily-baked glyphs.
type GlyphFactory = Arc<dyn Fn(char) -> Lazy<'static, Baked<u32>> + Send + Sync>;

/// Terminal application implementing Actor trait.
///
/// Receives engine events via Actor lanes and sends responses back to the engine.
/// Also processes PTY output from a background thread.
pub struct TerminalApp<P: Pixel + Surface<P>> {
    /// The terminal emulator - owns all terminal state
    emulator: TerminalEmulator,
    /// Channel to send data to the PTY write thread
    pty_tx: SyncSender<Vec<u8>>,
    /// Channel to receive PTY command batches
    pty_rx: Receiver<Vec<AnsiCommand>>,
    /// Application configuration
    config: Config,
    /// Glyph factory - shared across frames, caching is automatic via Lazy
    glyph: GlyphFactory,
    /// Engine handle to send responses back
    engine_tx: EngineActorHandle<P>,
    /// Phantom for pixel type
    _pixel: PhantomData<P>,
}

impl<P: Pixel + Surface<P>> TerminalApp<P> {
    /// Creates a new terminal app.
    pub fn new(
        emulator: TerminalEmulator,
        pty_tx: SyncSender<Vec<u8>>,
        pty_rx: Receiver<Vec<AnsiCommand>>,
        config: Config,
        engine_tx: EngineActorHandle<P>,
    ) -> Self {
        let f = font();
        let glyph_fn = glyphs(
            f.clone(),
            config.appearance.cell_width_px as u32,
            config.appearance.cell_height_px as u32,
        );
        Self {
            emulator,
            pty_tx,
            pty_rx,
            config,
            glyph: Arc::new(glyph_fn),
            engine_tx,
            _pixel: PhantomData,
        }
    }

    /// Handle an emulator action, sending AppManagement to engine if needed.
    fn handle_emulator_action(&mut self, action: EmulatorAction) {
        match action {
            EmulatorAction::WritePty(data) => {
                if self.pty_tx.send(data).is_err() {
                    log::warn!("TerminalApp: Failed to send to PTY");
                }
            }
            EmulatorAction::SetTitle(title) => {
                let _ = self
                    .engine_tx
                    .send(Message::Management(AppManagement::SetTitle(title)));
            }
            EmulatorAction::CopyToClipboard(text) => {
                let _ = self
                    .engine_tx
                    .send(Message::Management(AppManagement::CopyToClipboard(text)));
            }
            EmulatorAction::RequestClipboardContent => {
                let _ = self
                    .engine_tx
                    .send(Message::Management(AppManagement::RequestPaste));
            }
            EmulatorAction::ResizePty { cols, rows } => {
                log::info!("TerminalApp: PTY resize request {}x{}", cols, rows);
            }
            EmulatorAction::RingBell => {
                log::debug!("TerminalApp: Bell");
            }
            EmulatorAction::RequestRedraw => {}
            EmulatorAction::SetCursorVisibility(_) => {}
            EmulatorAction::Quit => {
                let _ = self
                    .engine_tx
                    .send(Message::Management(AppManagement::Quit));
            }
        }
    }

    /// Build and send a rendered surface to the engine.
    fn render_and_send(&mut self) {
        // Get logical snapshot from emulator
        let Some(snapshot) = self.emulator.get_render_snapshot() else {
            return;
        };

        // Check if any lines are dirty
        if !snapshot.lines.iter().any(|line| line.is_dirty) {
            // Signal skipped frame to Engine to keep VSync loop alive
            log::trace!("TerminalApp: No dirty lines, skipping frame");
            let _ = self.engine_tx.send(Message::Data(AppData::Skipped.into()));
            return;
        }

        // Use semantic Color for defaults
        let default_fg: Color = self.config.colors.foreground;
        let default_bg: Color = self.config.colors.background;

        // Build Surface from logical snapshot
        let grid = GridBuffer::from_snapshot(&snapshot, default_fg, default_bg);

        let terminal: TerminalSurface<P> = TerminalSurface::with_grid(
            &grid,
            self.glyph.clone(),
            self.config.appearance.cell_width_px as u32,
            self.config.appearance.cell_height_px as u32,
        );

        // Send to engine
        // TerminalSurface implements Manifold<P, u32>, which blanket implements Surface<P, u32>
        // We cast to Arc<dyn Surface...> for AppData
        let surface: Arc<dyn Surface<P, u32> + Send + Sync> = Arc::new(terminal);

        // Log frame generation (throttle if needed)
        log::trace!("TerminalApp: Sending frame");

        let _ = self
            .engine_tx
            .send(Message::Data(AppData::RenderSurfaceU32(surface).into()));
    }

    /// Process any pending PTY commands (non-blocking).
    fn process_pty_commands(&mut self) {
        while let Ok(commands) = self.pty_rx.try_recv() {
            for cmd in commands {
                if let Some(action) = self.emulator.interpret_input(EmulatorInput::Ansi(cmd)) {
                    self.handle_emulator_action(action);
                }
            }
        }
    }
}

/// Implement Actor trait - automatically gets Application trait via blanket impl.
impl<P: Pixel + Surface<P>> Actor<EngineEventData, EngineEventControl, EngineEventManagement>
    for TerminalApp<P>
{
    fn handle_data(&mut self, data: EngineEventData) {
        match data {
            EngineEventData::RequestFrame { .. } => {
                log::trace!("TerminalApp: Received RequestFrame");
                // Process any pending PTY output first
                self.process_pty_commands();
                // Render and send frame
                self.render_and_send();
            }
        }
    }

    fn handle_control(&mut self, ctrl: EngineEventControl) {
        let emulator_input = match ctrl {
            EngineEventControl::Resize(width_px, height_px) => {
                log::info!("TerminalApp: Resize {}x{} logical px", width_px, height_px);
                Some(EmulatorInput::Control(ControlEvent::Resize {
                    width_px: width_px as u16,
                    height_px: height_px as u16,
                }))
            }
            EngineEventControl::ScaleChanged(scale) => {
                log::info!("TerminalApp: Scale changed to {:.2}", scale);
                None
            }
            EngineEventControl::CloseRequested => {
                log::info!("TerminalApp: Close requested");
                // TODO: How to quit with new API?
                None
            }
        };

        if let Some(input) = emulator_input {
            if let Some(action) = self.emulator.interpret_input(input) {
                self.handle_emulator_action(action);
            }
        }

        // Process any pending PTY output
        self.process_pty_commands();
    }

    fn handle_management(&mut self, mgmt: EngineEventManagement) {
        let emulator_input = match mgmt {
            EngineEventManagement::KeyDown { key, mods, text } => {
                log::debug!("TerminalApp: Key: {:?} + {:?}, text: {:?}", mods, key, text);
                let key_input_action = keys::map_key_event_to_action(key, mods, &self.config)
                    .unwrap_or(UserInputAction::KeyInput {
                        symbol: key,
                        modifiers: mods,
                        text,
                    });
                Some(EmulatorInput::User(key_input_action))
            }
            EngineEventManagement::MouseClick { button, x, y } => match button {
                MouseButton::Left => Some(EmulatorInput::User(UserInputAction::StartSelection {
                    x_px: x as u16,
                    y_px: y as u16,
                })),
                MouseButton::Middle => {
                    Some(EmulatorInput::User(UserInputAction::RequestPrimaryPaste))
                }
                _ => None,
            },
            EngineEventManagement::MouseRelease { button, .. } => {
                if button == MouseButton::Left {
                    Some(EmulatorInput::User(UserInputAction::ApplySelectionClear))
                } else {
                    None
                }
            }
            EngineEventManagement::MouseMove { x, y, .. } => {
                Some(EmulatorInput::User(UserInputAction::ExtendSelection {
                    x_px: x as u16,
                    y_px: y as u16,
                }))
            }
            EngineEventManagement::MouseScroll { x, y, dx, dy, .. } => {
                log::trace!("MouseScroll at ({}, {}) delta ({}, {})", x, y, dx, dy);
                None
            }
            EngineEventManagement::FocusGained => {
                Some(EmulatorInput::User(UserInputAction::FocusGained))
            }
            EngineEventManagement::FocusLost => {
                Some(EmulatorInput::User(UserInputAction::FocusLost))
            }
            EngineEventManagement::Paste(text) => {
                Some(EmulatorInput::User(UserInputAction::PasteText(text)))
            }
        };

        if let Some(input) = emulator_input {
            if let Some(action) = self.emulator.interpret_input(input) {
                self.handle_emulator_action(action);
            }
        }

        // Process any pending PTY output
        self.process_pty_commands();
    }

    fn park(&mut self, _hint: ParkHint) {
        // Terminal app has no periodic tasks
    }
}

/// Creates terminal app and spawns it in a thread.
///
/// # Returns
/// - App handle (implements Application trait via blanket impl)
/// - PTY command sender (for PTY read thread)
/// - Thread join handle
pub fn spawn_terminal_app<P: Pixel + Surface<P> + 'static>(
    emulator: TerminalEmulator,
    pty_tx: SyncSender<Vec<u8>>,
    pty_rx: Receiver<Vec<AnsiCommand>>,
    config: Config,
    engine_tx: EngineActorHandle<P>,
) -> std::io::Result<(
    actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>,
    std::thread::JoinHandle<()>,
)> {
    // Create app actor channels
    let (app_tx, mut app_rx) =
        ActorScheduler::<EngineEventData, EngineEventControl, EngineEventManagement>::new(
            10,  // data_burst_limit: max 10 RequestFrame events per wake
            128, // data_buffer_size: event buffer
        );

    // Create app
    let mut app = TerminalApp::new(emulator, pty_tx, pty_rx, config, engine_tx);

    // Spawn app thread
    let handle = std::thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || {
            app_rx.run(&mut app);
        })?;

    Ok((app_tx, handle))
}
