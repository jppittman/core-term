//! Terminal application Actor implementation.
//!
//! NOTE: This module is temporarily stubbed during the pixelflow-core color system refactoring.
//! The previous implementation depended on APIs (Batch, surfaces::Baked, traits::Surface)
//! that have been replaced with the new Field-based Manifold system.
//!
//! TODO: Reimplement using the new pixelflow-core API.

use crate::ansi::commands::AnsiCommand;
use crate::config::Config;
use crate::term::TerminalEmulator;
use actor_scheduler::{Actor, ActorScheduler, ParkHint};
use core::marker::PhantomData;
use pixelflow_graphics::render::Pixel;
use pixelflow_runtime::{
    EngineActorHandle, EngineEventControl, EngineEventData, EngineEventManagement,
};

use std::sync::mpsc::{Receiver, SyncSender};

/// Terminal application implementing Actor trait.
///
/// Placeholder implementation - needs to be updated for new pixelflow-core API.
pub struct TerminalApp<P: Pixel> {
    _emulator: TerminalEmulator,
    _pty_tx: SyncSender<Vec<u8>>,
    _pty_rx: Receiver<Vec<AnsiCommand>>,
    _config: Config,
    _pixel: PhantomData<P>,
}

impl<P: Pixel> TerminalApp<P> {
    /// Creates a new terminal app.
    pub fn new(
        emulator: TerminalEmulator,
        pty_tx: SyncSender<Vec<u8>>,
        pty_rx: Receiver<Vec<AnsiCommand>>,
        config: Config,
        _engine_tx: EngineActorHandle<P>,
    ) -> Self {
        Self {
            _emulator: emulator,
            _pty_tx: pty_tx,
            _pty_rx: pty_rx,
            _config: config,
            _pixel: PhantomData,
        }
    }
}

impl<P: Pixel> Actor<EngineEventData, EngineEventControl, EngineEventManagement>
    for TerminalApp<P>
{
    fn handle_data(&mut self, _data: EngineEventData) {}
    fn handle_control(&mut self, _ctrl: EngineEventControl) {}
    fn handle_management(&mut self, _mgmt: EngineEventManagement) {}
    fn park(&mut self, _hint: ParkHint) {}
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
