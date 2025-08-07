// src/orchestrator/tests.rs

use super::*;
use crate::platform::mock::MockPlatform;

use crate::platform::backends::BackendEvent;

#[test]
fn it_should_shutdown_on_backend_close_event() {
    let mut platform = MockPlatform::new();
    let mut term_emulator = crate::term::TerminalEmulator::new(80, 24);
    let mut ansi_parser = crate::ansi::AnsiProcessor::new();
    let renderer = crate::renderer::Renderer::new();

    platform.push_event(PlatformEvent::BackendEvent(BackendEvent::CloseRequested));

    let mut orchestrator = AppOrchestrator::new(
        &mut platform,
        &mut term_emulator,
        &mut ansi_parser,
        renderer,
    );

    let status = orchestrator.process_event_cycle().unwrap();
    assert_eq!(status, OrchestratorStatus::Shutdown);
}
