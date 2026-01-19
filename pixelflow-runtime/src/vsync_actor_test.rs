#[cfg(test)]
mod tests {

    use crate::testing::mock_engine::MockEngine;
    use crate::vsync_actor::{VsyncActor, VsyncCommand};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn test_vsync_actor_start_stop() {
        // Setup mock engine
        let mock_engine = MockEngine::new();
        let engine_handle = mock_engine.handle();

        // Create vsync actor
        let refresh_rate = 60.0;
        let actor_handle = VsyncActor::spawn(refresh_rate, engine_handle);

        // Start the actor
        actor_handle.send(VsyncCommand::Start).expect("Failed to send Start command");

        // Request FPS to verify it's responsive
        let (tx, rx) = mpsc::channel();
        actor_handle.send(VsyncCommand::RequestCurrentFPS(tx)).expect("Failed to send RequestCurrentFPS");

        let fps = rx.recv_timeout(Duration::from_secs(1)).expect("Failed to receive FPS response");
        assert_eq!(fps, 0.0, "Initial FPS should be 0.0");

        // Stop the actor
        actor_handle.send(VsyncCommand::Stop).expect("Failed to send Stop command");

        // Shutdown
        actor_handle.send(VsyncCommand::Shutdown).expect("Failed to send Shutdown command");

        // Give it a moment to shutdown (thread join is implicit in test teardown usually,
        // but we want to ensure no panics during shutdown)
        std::thread::sleep(Duration::from_millis(50));
    }

    #[test]
    fn test_vsync_actor_update_refresh_rate() {
        let mock_engine = MockEngine::new();
        let engine_handle = mock_engine.handle();
        let actor_handle = VsyncActor::spawn(60.0, engine_handle);

        actor_handle.send(VsyncCommand::Start).expect("Failed to start");

        // Update refresh rate
        actor_handle.send(VsyncCommand::UpdateRefreshRate(120.0)).expect("Failed to update rate");

        // Verify it didn't crash and still responds
        let (tx, rx) = mpsc::channel();
        actor_handle.send(VsyncCommand::RequestCurrentFPS(tx)).expect("Failed to request FPS");
        let _ = rx.recv_timeout(Duration::from_secs(1)).expect("Failed to receive FPS");

        actor_handle.send(VsyncCommand::Shutdown).expect("Failed to shutdown");
    }

    // Since VsyncActor logic for ticking and sending vsyncs depends on time and thread scheduling,
    // we don't deeply test the timing accuracy here, but we can verify the structure and command handling.
    // The "cannot run" comments are removed as we now have a working test harness.
}
