# Issue: Actor-Scheduler Needs Explicit Shutdown Mechanism

**Component:** actor-scheduler
**Priority:** Medium
**Type:** Enhancement

## Problem

The actor-scheduler lacks an explicit shutdown mechanism. Currently, tests and applications must rely on implicit behaviors like:
- Dropping the `ActorHandle` to signal the actor should stop
- Waiting for threads to exit after handle drops
- Using arbitrary timeouts with polling loops

This coupling of lifetimes to handle drops creates fragile tests and makes graceful shutdown difficult to implement correctly.

## Current Workaround

Tests use polling with timeouts instead of relying on handle drops:

```rust
// Poll for response with timeout (don't rely on handle drops for shutdown)
let deadline = Instant::now() + Duration::from_millis(500);
loop {
    let responses = engine_responses.lock().unwrap();
    if !responses.is_empty() {
        // Process responses
        return;
    }
    drop(responses);
    if Instant::now() > deadline {
        panic!("Timeout waiting for response");
    }
    thread::sleep(Duration::from_millis(10));
}
```

## Proposed Solution

Add an explicit shutdown mechanism to the actor-scheduler:

1. **Shutdown message type**: Add a `Message::Shutdown` variant or similar that signals the actor loop to exit cleanly
2. **Shutdown method on ActorHandle**: `handle.shutdown()` that sends the shutdown signal
3. **Graceful drain**: Allow pending messages to be processed before exit
4. **Join support**: `handle.shutdown_and_join()` that waits for the actor thread to exit

### API Sketch

```rust
impl<D, C, M> ActorHandle<D, C, M> {
    /// Signal the actor to shutdown gracefully.
    pub fn shutdown(&self) -> Result<(), SendError> {
        self.send(Message::Control(/* internal shutdown signal */))
    }

    /// Signal shutdown and wait for the actor thread to exit.
    pub fn shutdown_and_join(self, timeout: Duration) -> Result<(), JoinError> {
        self.shutdown()?;
        // Wait for thread with timeout
    }
}
```

## Context

Discovered during Phase 2 of MESSAGE_CUJ_COVERAGE testing. The `tdd_eng09_request_frame_produces_response` test initially hung because it relied on `drop(app)` to release the engine handle and signal thread exit. This pattern is fragile and should be replaced with explicit shutdown.

## Files Affected

- `actor-scheduler/src/lib.rs` - Core scheduler implementation
- `actor-scheduler/src/handle.rs` - ActorHandle (if separate)
- Tests across all crates using actor-scheduler

## Related

- `core-term/tests/terminal_app_tdd_tests.rs` - Contains workaround pattern
- `core-term/tests/engine_app_message_tests.rs` - Uses drain actors without shutdown
