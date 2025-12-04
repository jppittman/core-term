//! Terminal application with Proxy + Worker architecture.
//!
//! The app is split into two parts:
//! - `TerminalAppProxy`: Thin wrapper implementing `Application` trait, lives in engine thread.
//! - `TerminalAppWorker`: Actual terminal logic, runs in its own thread.
//!
//! Communication is via typed channels - engine is oblivious to the threading model.

use crate::ansi::commands::AnsiCommand;
use crate::color::Color;
use crate::config::Config;
use crate::keys;
use crate::messages::{AppEvent, RenderRequest};
use crate::surface::{GridBuffer, TerminalSurface};
use crate::term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalEmulator, UserInputAction};
use core::marker::PhantomData;
use pixelflow_core::surfaces::Baked; // Re-add Baked import
use pixelflow_core::traits::Surface;
use pixelflow_engine::input::MouseButton;
use pixelflow_engine::{AppAction, AppState, Application, EngineEvent};
use pixelflow_fonts::{glyphs, Lazy}; // No longer need CellGlyph here
use pixelflow_render::font;
use pixelflow_render::Pixel;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

/// Glyph factory type - closure that returns lazily-baked glyphs.
type GlyphFactory = Arc<dyn Fn(char) -> Lazy<'static, Baked<u8>> + Send + Sync>;

// =============================================================================
// Proxy (lives in engine thread, implements Application trait)
// =============================================================================

/// Thin proxy that forwards Application trait calls to the worker thread.
///
/// This struct lives in the engine thread and implements `Application`.
/// All calls are forwarded via channels to the actual `TerminalAppWorker`
/// running in its own thread.
pub struct TerminalAppProxy<P: Pixel + Surface<P>> {
    /// Channel to send events to worker.
    event_tx: SyncSender<AppEvent>,
    /// Channel to send render requests to worker.
    render_req_tx: SyncSender<RenderRequest>,
    /// Channel to receive rendered surfaces from worker.
    render_resp_rx: Receiver<Option<Box<dyn Surface<P> + Send + Sync>>>,
    /// Channel to receive actions from worker.
    action_rx: Receiver<AppAction>,
    /// Phantom for pixel type.
    _pixel: PhantomData<P>,
}

impl<P: Pixel + Surface<P>> TerminalAppProxy<P> {
    /// Creates a new proxy with the given channels.
    pub fn new(
        event_tx: SyncSender<AppEvent>,
        render_req_tx: SyncSender<RenderRequest>,
        render_resp_rx: Receiver<Option<Box<dyn Surface<P> + Send + Sync>>>,
        action_rx: Receiver<AppAction>,
    ) -> Self {
        Self {
            event_tx,
            render_req_tx,
            render_resp_rx,
            action_rx,
            _pixel: PhantomData,
        }
    }
}

impl<P: Pixel + Surface<P>> Application<P> for TerminalAppProxy<P> {
    fn on_event(&mut self, event: EngineEvent) -> AppAction {
        // Forward event to worker (non-blocking send)
        let _ = self.event_tx.try_send(AppEvent::Engine(event));

        // Check for any pending actions from worker (non-blocking)
        match self.action_rx.try_recv() {
            Ok(action) => action,
            Err(TryRecvError::Empty) => AppAction::Continue,
            Err(TryRecvError::Disconnected) => {
                log::warn!("TerminalAppProxy: Worker disconnected");
                AppAction::Quit
            }
        }
    }

    fn render(&mut self, state: &AppState) -> Option<Box<dyn Surface<P> + Send + Sync>> {
        // Send render request to worker
        let req = RenderRequest {
            width_px: state.width_px,
            height_px: state.height_px,
        };
        if self.render_req_tx.try_send(req).is_err() {
            log::warn!("TerminalAppProxy: Failed to send render request");
            return None;
        }

        // Wait for response (blocking - engine expects surface at vsync rate)
        // Timeout prevents deadlock if worker is slow/blocked
        match self.render_resp_rx.recv_timeout(Duration::from_millis(16)) {
            Ok(surface) => surface,
            Err(RecvTimeoutError::Timeout) => {
                log::trace!("TerminalAppProxy: Render timeout, skipping frame");
                None
            }
            Err(RecvTimeoutError::Disconnected) => {
                log::warn!("TerminalAppProxy: Worker disconnected during render");
                None
            }
        }
    }
}

// =============================================================================
// Worker (runs in its own thread)
// =============================================================================

/// The actual terminal application logic running in its own thread.
///
/// Processes PTY commands at its own rate, handles events from the proxy,
/// and responds to render requests.
pub struct TerminalAppWorker<P: Pixel + Surface<P>> {
    /// The terminal emulator - owns all terminal state.
    emulator: TerminalEmulator,
    /// Channel to receive parsed ANSI commands from the PTY read thread.
    pty_rx: Receiver<Vec<AnsiCommand>>,
    /// Channel to send data to the PTY write thread.
    pty_tx: SyncSender<Vec<u8>>,
    /// Application configuration.
    config: Config,
    /// Glyph factory - shared across frames, caching is automatic via Lazy.
    glyph: GlyphFactory,

    // Channels from proxy
    /// Receive events from proxy.
    event_rx: Receiver<AppEvent>,
    /// Receive render requests from proxy.
    render_req_rx: Receiver<RenderRequest>,
    /// Send rendered surfaces to proxy.
    render_resp_tx: SyncSender<Option<Box<dyn Surface<P> + Send + Sync>>>,
    /// Send actions to proxy.
    action_tx: SyncSender<AppAction>,

    /// Phantom for pixel type.
    _pixel: PhantomData<P>,
}

impl<P: Pixel + Surface<P>> TerminalAppWorker<P> {
    /// Maximum number of command batches to process per drain cycle.
    const MAX_BATCHES_PER_DRAIN: usize = 10;

    /// Creates a new worker.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        emulator: TerminalEmulator,
        pty_rx: Receiver<Vec<AnsiCommand>>,
        pty_tx: SyncSender<Vec<u8>>,
        config: Config,
        event_rx: Receiver<AppEvent>,
        render_req_rx: Receiver<RenderRequest>,
        render_resp_tx: SyncSender<Option<Box<dyn Surface<P> + Send + Sync>>>,
        action_tx: SyncSender<AppAction>,
    ) -> Self {
        let f = font();
        let glyph_fn = glyphs(
            f.clone(),
            config.appearance.cell_width_px as u32,
            config.appearance.cell_height_px as u32,
        );
        Self {
            emulator,
            pty_rx,
            pty_tx,
            config,
            glyph: Arc::new(glyph_fn),
            event_rx,
            render_req_rx,
            render_resp_tx,
            action_tx,
            _pixel: PhantomData,
        }
    }

    /// Worker main loop - runs until shutdown.
    pub fn run(mut self) {
        log::info!("TerminalAppWorker: Starting event loop");

        loop {
            // 1. Drain PTY commands (non-blocking, high priority)
            if !self.drain_pty_nonblocking() {
                // PTY closed (shell exited) - request quit
                log::info!("TerminalAppWorker: Requesting quit due to PTY closure");
                let _ = self.action_tx.try_send(AppAction::Quit);
                break;
            }

            // 2. Check for events from proxy (non-blocking)
            match self.event_rx.try_recv() {
                Ok(AppEvent::Engine(event)) => {
                    if let Some(action) = self.handle_engine_event(event) {
                        let _ = self.action_tx.try_send(action);
                    }
                }
                Ok(AppEvent::Shutdown) => {
                    log::info!("TerminalAppWorker: Shutdown received");
                    break;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    log::info!("TerminalAppWorker: Proxy disconnected");
                    break;
                }
            }

            // 3. Check for render requests (non-blocking)
            match self.render_req_rx.try_recv() {
                Ok(req) => {
                    let surface = self.build_surface(req.width_px, req.height_px);
                    let _ = self.render_resp_tx.try_send(surface);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    log::info!("TerminalAppWorker: Render channel disconnected");
                    break;
                }
            }

            // 4. Small sleep to avoid busy loop when idle
            std::thread::sleep(Duration::from_micros(100));
        }

        log::info!("TerminalAppWorker: Event loop exited");
    }

    /// Drain PTY commands (non-blocking).
    ///
    /// Returns `true` if the PTY channel is still connected, `false` if disconnected (PTY closed).
    fn drain_pty_nonblocking(&mut self) -> bool {
        for _ in 0..Self::MAX_BATCHES_PER_DRAIN {
            match self.pty_rx.try_recv() {
                Ok(commands) => {
                    for cmd in commands {
                        if let Some(action) =
                            self.emulator.interpret_input(EmulatorInput::Ansi(cmd))
                        {
                            self.handle_emulator_action(action);
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    log::info!("TerminalAppWorker: PTY channel disconnected (shell exited)");
                    return false;
                }
            }
        }
        true
    }

    /// Handle an engine event, returning an AppAction if needed.
    fn handle_engine_event(&mut self, event: EngineEvent) -> Option<AppAction> {
        let emulator_input = match event {
            EngineEvent::Resize(width_px, height_px) => {
                log::info!(
                    "TerminalAppWorker: Resize {}x{} logical px",
                    width_px,
                    height_px
                );
                Some(EmulatorInput::Control(ControlEvent::Resize {
                    width_px: width_px as u16,
                    height_px: height_px as u16,
                }))
            }
            EngineEvent::KeyDown { key, mods, text } => {
                log::debug!(
                    "TerminalAppWorker: Key: {:?} + {:?}, text: {:?}",
                    mods,
                    key,
                    text
                );
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
            EngineEvent::MouseScroll { x, y, dx, dy, .. } => {
                // TODO: Implement scroll handling (e.g., alternate screen scrollback)
                log::trace!("MouseScroll at ({}, {}) delta ({}, {})", x, y, dx, dy);
                None
            }
            EngineEvent::ScaleChanged(scale) => {
                log::info!("TerminalAppWorker: Scale changed to {:.2}", scale);
                // Scale changes are handled by the engine, we just need to know about them
                None
            }
            EngineEvent::FocusGained => Some(EmulatorInput::User(UserInputAction::FocusGained)),
            EngineEvent::FocusLost => Some(EmulatorInput::User(UserInputAction::FocusLost)),
            EngineEvent::Paste(text) => Some(EmulatorInput::User(UserInputAction::PasteText(text))),
            EngineEvent::Wake => return None,
            EngineEvent::CloseRequested => return Some(AppAction::Quit),
        };

        if let Some(input) = emulator_input {
            if let Some(action) = self.emulator.interpret_input(input) {
                return self.handle_emulator_action(action);
            }
        }

        None
    }

    /// Handle an emulator action, returning an AppAction if needed.
    fn handle_emulator_action(&mut self, action: EmulatorAction) -> Option<AppAction> {
        match action {
            EmulatorAction::WritePty(data) => {
                if self.pty_tx.send(data).is_err() {
                    log::warn!("TerminalAppWorker: Failed to send to PTY");
                }
                None
            }
            EmulatorAction::SetTitle(title) => Some(AppAction::SetTitle(title)),
            EmulatorAction::CopyToClipboard(text) => Some(AppAction::CopyToClipboard(text)),
            EmulatorAction::RequestClipboardContent => Some(AppAction::RequestPaste),
            EmulatorAction::ResizePty { cols, rows } => {
                log::info!("TerminalAppWorker: PTY resize request {}x{}", cols, rows);
                None
            }
            EmulatorAction::RingBell => {
                log::debug!("TerminalAppWorker: Bell");
                None
            }
            EmulatorAction::RequestRedraw => None,
            EmulatorAction::SetCursorVisibility(_) => None,
        }
    }

    /// Build a surface for rendering.
    fn build_surface(
        &mut self,
        _width_px: u32,
        _height_px: u32,
    ) -> Option<Box<dyn Surface<P> + Send + Sync>> {
        // Get logical snapshot from emulator (has per-line dirty flags)
        let snapshot = self.emulator.get_render_snapshot()?;

        // Check if any lines are dirty - if not, skip this frame
        if !snapshot.lines.iter().any(|line| line.is_dirty) {
            return None;
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

        Some(Box::new(terminal))
    }
}

// =============================================================================
// Factory function to create both proxy and worker
// =============================================================================

/// Creates a proxy-worker pair and spawns the worker thread.
///
/// Returns the proxy (to be passed to the engine) and a handle to the worker thread.
pub fn spawn_terminal_app<P: Pixel + Surface<P> + 'static>(
    emulator: TerminalEmulator,
    pty_rx: Receiver<Vec<AnsiCommand>>,
    pty_tx: SyncSender<Vec<u8>>,
    config: Config,
) -> std::io::Result<(TerminalAppProxy<P>, std::thread::JoinHandle<()>)> {
    use std::sync::mpsc::sync_channel;

    // Create channels for proxy <-> worker communication
    let (event_tx, event_rx) = sync_channel(64);
    let (render_req_tx, render_req_rx) = sync_channel(1);
    let (render_resp_tx, render_resp_rx) = sync_channel(1);
    let (action_tx, action_rx) = sync_channel(16);

    // Create worker
    let worker = TerminalAppWorker::new(
        emulator,
        pty_rx,
        pty_tx,
        config,
        event_rx,
        render_req_rx,
        render_resp_tx,
        action_tx,
    );

    // Spawn worker thread
    let handle = std::thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || worker.run())?;

    // Create proxy
    let proxy = TerminalAppProxy::new(event_tx, render_req_tx, render_resp_rx, action_rx);

    Ok((proxy, handle))
}
