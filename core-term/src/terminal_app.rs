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

/// Parameters for constructing a TerminalApp.
pub struct TerminalAppParams<P: Pixel> {
    /// Terminal emulator instance.
    pub emulator: TerminalEmulator,
    /// Channel to send data to PTY.
    pub pty_tx: SyncSender<Vec<u8>>,
    /// Channel to receive commands from PTY.
    pub pty_rx: Receiver<Vec<AnsiCommand>>,
    /// Application configuration.
    pub config: Config,
    /// Handle to the engine actor.
    pub engine_tx: EngineActorHandle<P>,
}

impl<P: Pixel> TerminalApp<P> {
    /// Creates a new terminal app.
    pub fn new(params: TerminalAppParams<P>) -> Self {
        Self {
            _emulator: params.emulator,
            _pty_tx: params.pty_tx,
            _pty_rx: params.pty_rx,
            _config: params.config,
            _pixel: PhantomData,
        }
    }
}

impl<P: Pixel> Actor<EngineEventData, EngineEventControl, EngineEventManagement>
    for TerminalApp<P>
{
    fn handle_data(&mut self, _data: EngineEventData) {}

    fn handle_control(&mut self, ctrl: EngineEventControl) {
        match ctrl {
            EngineEventControl::Resize(width_px, height_px) => {
                use crate::term::{ControlEvent, EmulatorInput, TerminalInterface};
                // Convert u32 pixels to u16 for ControlEvent
                // Saturate at u16::MAX to prevent overflow panics
                let width_u16 = width_px.min(u16::MAX as u32) as u16;
                let height_u16 = height_px.min(u16::MAX as u32) as u16;

                let input = EmulatorInput::Control(ControlEvent::Resize {
                    width_px: width_u16,
                    height_px: height_u16,
                });
                self._emulator.interpret_input(input);
            }
            EngineEventControl::CloseRequested => {
                // Handle close request?
                // Probably should signal application exit or similar
            }
            EngineEventControl::ScaleChanged(_scale) => {
                // Handle scale change
            }
        }
    }

    fn handle_management(&mut self, mgmt: EngineEventManagement) {
        match mgmt {
            EngineEventManagement::KeyDown { key, mods, text } => {
                use crate::term::{
                    EmulatorAction, EmulatorInput, TerminalInterface, UserInputAction,
                };

                let input = EmulatorInput::User(UserInputAction::KeyInput {
                    symbol: key,
                    modifiers: mods,
                    text,
                });

                if let Some(action) = self._emulator.interpret_input(input) {
                    match action {
                        EmulatorAction::WritePty(bytes) => {
                            if let Err(e) = self._pty_tx.send(bytes) {
                                log::warn!("Failed to send input to PTY: {}", e);
                            }
                        }
                        EmulatorAction::Quit => {
                            // Handle quit
                            // self._engine_tx.send(...)
                        }
                        _ => {
                            // Handle other actions (e.g., paste, copy, etc.) if necessary
                        }
                    }
                }
            }
            // Add other event handling as needed
            _ => {}
        }
    }

    fn park(&mut self, _hint: ParkHint) -> ParkHint {
        ParkHint::Wait
    }
}

/// Creates terminal app and spawns it in a thread.
pub fn spawn_terminal_app<P: Pixel + 'static>(
    params: TerminalAppParams<P>,
) -> std::io::Result<(
    actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>,
    std::thread::JoinHandle<()>,
)> {
    let (app_tx, mut app_rx) =
        ActorScheduler::<EngineEventData, EngineEventControl, EngineEventManagement>::new(10, 128);

    let mut app = TerminalApp::new(params);

    let handle = std::thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || {
            app_rx.run(&mut app);
        })?;

    Ok((app_tx, handle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansi::commands::AnsiCommand;
    use crate::term::{EmulatorInput, TerminalEmulator, UserInputAction};
    use actor_scheduler::{Actor, ParkHint};
    use pixelflow_runtime::input::{KeySymbol, Modifiers};
    use pixelflow_runtime::{EngineEventControl, EngineEventManagement};
    use std::sync::mpsc::{Receiver, SyncSender};

    // Define a DummyPixel struct for testing
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    struct DummyPixel;
    impl pixelflow_graphics::render::Pixel for DummyPixel {
        fn from_u32(_: u32) -> Self {
            Self
        }
        fn to_u32(self) -> u32 {
            0
        }
        fn from_rgba(_r: f32, _g: f32, _b: f32, _a: f32) -> Self {
            Self
        }
    }

    // Helper to create a test instance
    fn create_test_app() -> (
        TerminalApp<DummyPixel>,
        Receiver<Vec<u8>>,
        SyncSender<Vec<AnsiCommand>>,
        pixelflow_runtime::EngineActorHandle<DummyPixel>,
    ) {
        let emulator = TerminalEmulator::new(80, 24);
        let (pty_tx, pty_rx) = std::sync::mpsc::sync_channel(128);
        let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel(128);

        // Create a dummy engine handle
        let (engine_tx, _) = actor_scheduler::ActorScheduler::new(10, 10);

        let config = Config::default();
        let params = TerminalAppParams {
            emulator,
            pty_tx,
            pty_rx: cmd_rx,
            config,
            engine_tx: engine_tx.clone(),
        };
        let app = TerminalApp::new(params);

        (app, pty_rx, cmd_tx, engine_tx)
    }

    #[test]
    fn test_handle_control_resize() {
        let (mut app, _pty_rx, _cmd_tx, _) = create_test_app();

        // Initial size is 80x24
        use crate::term::TerminalInterface;
        let snapshot_initial = app._emulator.get_render_snapshot().expect("Snapshot");
        assert_eq!(snapshot_initial.dimensions, (80, 24));

        // Send resize event
        // Default config: cell width 10, height 16.
        // Resize to 1000x800 -> 100x50 cells.
        let resize_event = EngineEventControl::Resize(1000, 800);
        app.handle_control(resize_event);

        // Verify resize via snapshot
        let snapshot_new = app._emulator.get_render_snapshot().expect("Snapshot");
        assert_eq!(
            snapshot_new.dimensions,
            (100, 50),
            "Emulator should have resized to 100x50"
        );
    }

    #[test]
    fn test_handle_management_keydown() {
        let (mut app, pty_rx, _cmd_tx, _) = create_test_app();

        // Simulate KeyDown
        let key_event = EngineEventManagement::KeyDown {
            key: KeySymbol::Char('a'),
            mods: Modifiers::empty(),
            text: Some("a".to_string()),
        };

        app.handle_management(key_event);

        // We expect 'a' to be sent to PTY
        let received = pty_rx.try_recv();
        assert!(received.is_ok(), "Should receive data on PTY channel");
        let bytes = received.unwrap();
        assert_eq!(bytes, vec![b'a']);
    }
}
