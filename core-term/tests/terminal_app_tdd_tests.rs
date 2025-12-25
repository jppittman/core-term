//! TDD Tests for TerminalApp Behavior
//!
//! These tests verify that TerminalApp correctly handles engine events.
//! They test the REAL TerminalApp, not mocks, to expose missing implementations.
//!
//! EXPECTED: These tests WILL FAIL until TerminalApp handlers are implemented.

use actor_scheduler::{Actor, ParkHint};
use core_term::config::Config;
use core_term::term::TerminalEmulator;
use core_term::terminal_app::TerminalApp;
use pixelflow_graphics::Rgba8;
use pixelflow_runtime::api::public::{
    AppManagement, EngineEventControl, EngineEventData, EngineEventManagement,
};
use pixelflow_runtime::channel::{create_engine_actor, EngineControl, EngineData};
use pixelflow_runtime::input::{KeySymbol, Modifiers};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// Test Fixtures
// =============================================================================

/// Creates a TerminalApp with mock channels for testing.
/// Returns: (app, pty_write_rx) - the app and the receiver to check PTY writes
fn create_test_terminal_app() -> (TerminalApp<Rgba8>, Receiver<Vec<u8>>) {
    let emulator = TerminalEmulator::new(80, 24);
    let config = Config::default();

    // PTY write channel: App -> PTY (for keystrokes, paste)
    let (pty_write_tx, pty_write_rx): (SyncSender<Vec<u8>>, Receiver<Vec<u8>>) = sync_channel(64);

    // ANSI command channel: Parser -> App (for display commands)
    let (_ansi_cmd_tx, ansi_cmd_rx): (
        SyncSender<Vec<core_term::ansi::AnsiCommand>>,
        Receiver<Vec<core_term::ansi::AnsiCommand>>,
    ) = sync_channel(64);

    // Create proper engine actor handle
    let (engine_tx, mut engine_rx) = create_engine_actor::<Rgba8>(None);

    // Spawn a drain thread for engine messages
    thread::spawn(move || {
        struct DrainActor;
        impl Actor<EngineData<Rgba8>, EngineControl<Rgba8>, AppManagement> for DrainActor {
            fn handle_data(&mut self, _: EngineData<Rgba8>) {}
            fn handle_control(&mut self, _: EngineControl<Rgba8>) {}
            fn handle_management(&mut self, _: AppManagement) {}
            fn park(&mut self, _: ParkHint) -> ParkHint {
                ParkHint::Wait
            }
        }
        engine_rx.run(&mut DrainActor);
    });

    let app = TerminalApp::new(emulator, pty_write_tx, ansi_cmd_rx, config, engine_tx);

    (app, pty_write_rx)
}

// =============================================================================
// ENG-04: KeyDown Should Write to PTY
// =============================================================================

#[test]
fn tdd_eng04_keydown_text_writes_to_pty() {
    // Given: A real TerminalApp
    let (mut app, pty_rx) = create_test_terminal_app();

    // When: KeyDown event with text is sent
    app.handle_management(EngineEventManagement::KeyDown {
        key: KeySymbol::Char('a'),
        mods: Modifiers::empty(),
        text: Some("a".to_string()),
    });

    // Then: The text should be written to PTY
    let result = pty_rx.recv_timeout(Duration::from_millis(100));
    assert!(
        result.is_ok(),
        "KeyDown with text should write to PTY, but nothing was received"
    );
    assert_eq!(
        result.unwrap(),
        b"a".to_vec(),
        "KeyDown should write the text bytes to PTY"
    );
}

#[test]
fn tdd_eng04_keydown_enter_writes_crlf() {
    // Given: A real TerminalApp
    let (mut app, pty_rx) = create_test_terminal_app();

    // When: Enter key is pressed
    app.handle_management(EngineEventManagement::KeyDown {
        key: KeySymbol::Enter,
        mods: Modifiers::empty(),
        text: Some("\r".to_string()),
    });

    // Then: CR should be written to PTY
    let result = pty_rx.recv_timeout(Duration::from_millis(100));
    assert!(result.is_ok(), "Enter key should write to PTY");
    assert_eq!(result.unwrap(), b"\r".to_vec());
}

#[test]
fn tdd_eng04_keydown_no_text_no_write() {
    // Given: A real TerminalApp
    let (mut app, pty_rx) = create_test_terminal_app();

    // When: KeyDown without text (e.g., just Shift pressed)
    app.handle_management(EngineEventManagement::KeyDown {
        key: KeySymbol::Shift,
        mods: Modifiers::SHIFT,
        text: None,
    });

    // Then: Nothing should be written to PTY
    let result = pty_rx.recv_timeout(Duration::from_millis(50));
    assert!(
        result.is_err(),
        "KeyDown without text should NOT write to PTY"
    );
}

// =============================================================================
// ENG-08: Paste Should Write to PTY
// =============================================================================

#[test]
fn tdd_eng08_paste_writes_to_pty() {
    // Given: A real TerminalApp
    let (mut app, pty_rx) = create_test_terminal_app();

    // When: Paste event is sent
    app.handle_management(EngineEventManagement::Paste("Hello, World!".to_string()));

    // Then: The pasted text should be written to PTY
    let result = pty_rx.recv_timeout(Duration::from_millis(100));
    assert!(result.is_ok(), "Paste should write to PTY");
    assert_eq!(result.unwrap(), b"Hello, World!".to_vec());
}

#[test]
fn tdd_eng08_paste_empty_no_write() {
    // Given: A real TerminalApp
    let (mut app, pty_rx) = create_test_terminal_app();

    // When: Empty paste event is sent
    app.handle_management(EngineEventManagement::Paste(String::new()));

    // Then: Nothing should be written to PTY (or empty write is acceptable)
    let result = pty_rx.recv_timeout(Duration::from_millis(50));
    // Either no write, or empty write is fine
    if let Ok(bytes) = result {
        assert!(bytes.is_empty(), "Empty paste should not write non-empty data");
    }
}

// =============================================================================
// ENG-01: Resize Should Update Terminal Dimensions
// =============================================================================

#[test]
fn tdd_eng01_resize_updates_emulator() {
    // Given: A real TerminalApp with 80x24 terminal
    let (mut app, _pty_rx) = create_test_terminal_app();

    // When: Resize event for larger window is sent
    // Assuming 10x16 pixel cells, 1920x1080 = 192 cols x 67 rows
    app.handle_control(EngineEventControl::Resize(1920, 1080));

    // Then: Terminal emulator should be resized
    // NOTE: We can't directly verify this without exposing emulator state,
    // but the test documents the expected behavior.
    // For now, this test verifies the handler doesn't panic.
}

// =============================================================================
// ENG-09: RequestFrame Should Trigger Render
// =============================================================================

#[test]
fn tdd_eng09_request_frame_produces_response() {
    // Given: A real TerminalApp with an engine channel
    let emulator = TerminalEmulator::new(80, 24);
    let config = Config::default();
    let (pty_write_tx, _pty_write_rx): (SyncSender<Vec<u8>>, Receiver<Vec<u8>>) = sync_channel(64);
    let (_ansi_cmd_tx, ansi_cmd_rx): (
        SyncSender<Vec<core_term::ansi::AnsiCommand>>,
        Receiver<Vec<core_term::ansi::AnsiCommand>>,
    ) = sync_channel(64);

    // Create engine channel that we can monitor for AppData responses
    let engine_responses = Arc::new(std::sync::Mutex::new(Vec::new()));
    let responses_clone = engine_responses.clone();

    struct ResponseCollector {
        responses: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl Actor<EngineData<Rgba8>, EngineControl<Rgba8>, AppManagement> for ResponseCollector {
        fn handle_data(&mut self, data: EngineData<Rgba8>) {
            // App sends AppData wrapped in EngineData::FromApp
            if let EngineData::FromApp(app_data) = data {
                let name = match app_data {
                    pixelflow_runtime::api::public::AppData::RenderSurface(_) => "RenderSurface",
                    pixelflow_runtime::api::public::AppData::RenderSurfaceU32(_) => {
                        "RenderSurfaceU32"
                    }
                    pixelflow_runtime::api::public::AppData::Skipped => "Skipped",
                    pixelflow_runtime::api::public::AppData::_Phantom(_) => "Phantom",
                };
                self.responses.lock().unwrap().push(name.to_string());
            }
        }
        fn handle_control(&mut self, _: EngineControl<Rgba8>) {}
        fn handle_management(&mut self, _: AppManagement) {}
        fn park(&mut self, _: ParkHint) -> ParkHint {
            ParkHint::Wait
        }
    }

    let (engine_tx, mut engine_rx) = create_engine_actor::<Rgba8>(None);

    // Spawn collector thread (will be detached - no explicit shutdown mechanism exists)
    thread::spawn(move || {
        let mut collector = ResponseCollector {
            responses: responses_clone,
        };
        engine_rx.run(&mut collector);
    });

    let mut app = TerminalApp::new(emulator, pty_write_tx, ansi_cmd_rx, config, engine_tx);

    // When: RequestFrame is sent
    let now = Instant::now();
    app.handle_data(EngineEventData::RequestFrame {
        timestamp: now,
        target_timestamp: now + Duration::from_millis(16),
        refresh_interval: Duration::from_millis(16),
    });

    // Poll for response with timeout (don't rely on handle drops for shutdown)
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        let responses = engine_responses.lock().unwrap();
        if !responses.is_empty() {
            // Then: App should send a render response (RenderSurface or Skipped)
            assert!(
                responses.iter().any(|r| r == "Skipped" || r == "RenderSurface"),
                "Expected Skipped or RenderSurface, got: {:?}",
                *responses
            );
            return;
        }
        drop(responses);

        if Instant::now() > deadline {
            panic!("RequestFrame should produce a render response within 500ms");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

// =============================================================================
// Focus Events Should Update State
// =============================================================================

#[test]
fn tdd_focus_lost_should_update_state() {
    // Given: A real TerminalApp
    let (mut app, _pty_rx) = create_test_terminal_app();

    // When: Focus lost event is sent
    app.handle_management(EngineEventManagement::FocusLost);

    // Then: App should track focus state (test doesn't panic)
    // In a real implementation, this might affect cursor blinking, etc.
}

#[test]
fn tdd_focus_gained_should_update_state() {
    // Given: A real TerminalApp
    let (mut app, _pty_rx) = create_test_terminal_app();

    // When: Focus gained event is sent
    app.handle_management(EngineEventManagement::FocusGained);

    // Then: App should track focus state (test doesn't panic)
}
