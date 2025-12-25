//! Terminal application Actor implementation.
//!
//! Handles engine events (keyboard, mouse, resize) and produces render surfaces.

use crate::ansi::commands::AnsiCommand;
use crate::config::Config;
use crate::term::TerminalEmulator;
use actor_scheduler::{Actor, ActorScheduler, Message, ParkHint};
use core::marker::PhantomData;
use pixelflow_graphics::render::Pixel;
use pixelflow_runtime::api::public::AppData;
use pixelflow_runtime::{
    EngineActorHandle, EngineEventControl, EngineEventData, EngineEventManagement,
};

use std::sync::mpsc::{Receiver, SyncSender};

/// Terminal application implementing Actor trait.
pub struct TerminalApp<P: Pixel> {
    _emulator: TerminalEmulator,
    pty_tx: SyncSender<Vec<u8>>,
    _pty_rx: Receiver<Vec<AnsiCommand>>,
    _config: Config,
    engine_tx: EngineActorHandle<P>,
    _pixel: PhantomData<P>,
}

impl<P: Pixel> TerminalApp<P> {
    /// Creates a new terminal app.
    pub fn new(
        emulator: TerminalEmulator,
        pty_tx: SyncSender<Vec<u8>>,
        pty_rx: Receiver<Vec<AnsiCommand>>,
        config: Config,
        engine_tx: EngineActorHandle<P>,
    ) -> Self {
        Self {
            _emulator: emulator,
            pty_tx,
            _pty_rx: pty_rx,
            _config: config,
            engine_tx,
            _pixel: PhantomData,
        }
    }

    /// Write bytes to the PTY.
    fn write_to_pty(&self, data: &[u8]) {
        if !data.is_empty() {
            let _ = self.pty_tx.send(data.to_vec());
        }
    }
}

impl<P: Pixel> Actor<EngineEventData, EngineEventControl, EngineEventManagement>
    for TerminalApp<P>
{
    fn handle_data(&mut self, data: EngineEventData) {
        match data {
            EngineEventData::RequestFrame { .. } => {
                // TODO: Generate actual render surface from terminal state
                // For now, send Skipped to indicate we're alive
                let _ = self.engine_tx.send(Message::Data(AppData::Skipped.into()));
            }
        }
    }

    fn handle_control(&mut self, ctrl: EngineEventControl) {
        match ctrl {
            EngineEventControl::Resize(_width_px, _height_px) => {
                // TODO: Convert pixels to cols/rows and resize emulator
                // For now, the terminal_app_tdd_tests just check that we don't panic
            }
            EngineEventControl::CloseRequested => {
                // Shutdown will be handled by the actor system
            }
            EngineEventControl::ScaleChanged(_scale) => {
                // TODO: Update rendering scale factor
            }
        }
    }

    fn handle_management(&mut self, mgmt: EngineEventManagement) {
        match mgmt {
            EngineEventManagement::KeyDown { text, .. } => {
                if let Some(t) = text {
                    self.write_to_pty(t.as_bytes());
                }
            }
            EngineEventManagement::Paste(content) => {
                self.write_to_pty(content.as_bytes());
            }
            EngineEventManagement::FocusGained => {
                // TODO: Track focus state for cursor blinking
            }
            EngineEventManagement::FocusLost => {
                // TODO: Track focus state for cursor blinking
            }
            EngineEventManagement::MouseClick { .. }
            | EngineEventManagement::MouseRelease { .. }
            | EngineEventManagement::MouseMove { .. }
            | EngineEventManagement::MouseScroll { .. } => {
                // TODO: Handle mouse events for selection/scrolling
            }
        }
    }

    fn park(&mut self, _hint: ParkHint) -> ParkHint {
        ParkHint::Wait
    }
}

/// Creates terminal app and spawns it in a thread.
pub fn spawn_terminal_app<P: Pixel + 'static>(
    emulator: TerminalEmulator,
    pty_tx: SyncSender<Vec<u8>>,
    pty_rx: Receiver<Vec<AnsiCommand>>,
    config: Config,
    engine_tx: EngineActorHandle<P>,
) -> std::io::Result<(
    actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>,
    std::thread::JoinHandle<()>,
)> {
    let (app_tx, mut app_rx) =
        ActorScheduler::<EngineEventData, EngineEventControl, EngineEventManagement>::new(10, 128);

    let mut app = TerminalApp::new(emulator, pty_tx, pty_rx, config, engine_tx);

    let handle = std::thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || {
            app_rx.run(&mut app);
        })?;

    Ok((app_tx, handle))
}
