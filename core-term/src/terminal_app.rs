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

    fn handle_control(&mut self, ctrl: EngineEventControl) {
        match ctrl {
            EngineEventControl::Resize(width_px, height_px) => {
                let cell_width = self._config.appearance.cell_width_px;
                let cell_height = self._config.appearance.cell_height_px;
                if cell_width > 0 && cell_height > 0 {
                    let cols = width_px as usize / cell_width;
                    let rows = height_px as usize / cell_height;
                    self._emulator.resize(cols, rows);
                }
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
            EngineEventManagement::KeyDown {
                key: _,
                mods: _,
                text,
            } => {
                if let Some(txt) = text {
                    // Forward text to PTY
                    if let Err(e) = self._pty_tx.send(txt.into_bytes()) {
                        log::warn!("Failed to send input to PTY: {}", e);
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
        let app = TerminalApp::new(emulator, pty_tx, cmd_rx, config, engine_tx.clone());

        (app, pty_rx, cmd_tx, engine_tx)
    }

    #[test]
    fn test_handle_control_resize() {
        let (mut app, _pty_rx, _cmd_tx, _) = create_test_app();

        // Initial size is 80x24
        assert_eq!(app._emulator.dimensions(), (80, 24));

        // Send resize event (assuming cells for now, or pixels that map to cells)
        // Note: EngineEventControl::Resize usually sends window pixels.
        // TerminalApp needs to convert.
        // For this test, let's assume we implement logic that eventually resizes the emulator.
        // To make the test meaningful, we should check if _emulator.dimensions() changes.

        // Simulating a resize that results in 100x30 cells.
        // We'll pass raw pixels, and inside handle_control we'll need to use config.cell_width/height.
        // Config defaults are likely 0 or something unless we set them.
        // Config::default() -> defaults.

        // Let's assume standard cell size 10x20
        // Resize(1000, 600) -> 100x30
        let resize_event = EngineEventControl::Resize(1000, 600);
        app.handle_control(resize_event);

        // Since handle_control is empty, this will remain 80x24.
        // We ASSERT that it CHANGED to prove it fails.
        // If it passes, it means it didn't change (if we asserted eq).
        // We want to prove the implementation is missing.
        assert_ne!(
            app._emulator.dimensions(),
            (80, 24),
            "Emulator should have resized"
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
