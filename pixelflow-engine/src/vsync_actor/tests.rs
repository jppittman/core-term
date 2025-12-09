//! VSync actor unit tests - comprehensive testing of timing and backpressure

use super::*;
use crate::api::private::EngineControl;
use pixelflow_render::Rgba;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// Mock Infrastructure
// ============================================================================

/// Mock engine handle that records all VSync messages sent to it
struct MockEngineHandle {
    vsyncs_tx: Sender<VSync>,
}

/// VSync message data extracted from EngineControl
#[derive(Debug, Clone)]
struct VSync {
    timestamp: Instant,
    target_timestamp: Instant,
    refresh_interval: Duration,
}

impl MockEngineHandle {
    fn new() -> (Self, Receiver<VSync>) {
        let (tx, rx) = channel();
        (Self { vsyncs_tx: tx }, rx)
    }

    fn send(&self, ctrl: EngineControl<Rgba>) -> Result<(), std::sync::mpsc::SendError<VSync>> {
        match ctrl {
            EngineControl::VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            } => self.vsyncs_tx.send(VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            }),
            _ => Ok(()), // Ignore other control messages
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Collect all available VSync messages without blocking
fn collect_all_vsyncs(rx: &Receiver<VSync>) -> Vec<VSync> {
    let mut vsyncs = Vec::new();
    while let Ok(v) = rx.try_recv() {
        vsyncs.push(v);
    }
    vsyncs
}

/// Wait for a single VSync message with timeout
fn wait_for_vsync(rx: &Receiver<VSync>, timeout: Duration) -> Option<VSync> {
    rx.recv_timeout(timeout).ok()
}

/// Collect VSync messages for a specific duration
fn collect_vsyncs_for_duration(rx: &Receiver<VSync>, duration: Duration) -> Vec<VSync> {
    let start = Instant::now();
    let mut vsyncs = Vec::new();

    while start.elapsed() < duration {
        let remaining = duration - start.elapsed();
        if let Ok(v) = rx.recv_timeout(remaining) {
            vsyncs.push(v);
        }
    }

    // Drain any remaining messages
    vsyncs.extend(collect_all_vsyncs(rx));
    vsyncs
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_vsync_sends_at_expected_rate() {
    // Spawn VSync at 10Hz (100ms interval)
    let (mock_handle, vsync_rx) = MockEngineHandle::new();
    let mock_handle = Arc::new(mock_handle);
    let mock_clone = mock_handle.clone();

    let vsync = VsyncActor::spawn_with_handler(10.0, move |ctrl| {
        mock_clone.send(ctrl).map_err(|_| anyhow::anyhow!("send failed"))
    });

    // Collect ~5 VSync messages, sending feedback for each
    let mut vsyncs = Vec::new();
    let start = Instant::now();
    let duration = Duration::from_millis(500);

    while start.elapsed() < duration {
        let remaining = duration - start.elapsed();
        if let Some(v) = wait_for_vsync(&vsync_rx, remaining) {
            vsyncs.push(v);
            // Send feedback to refill token
            let _ = vsync.send(RenderedResponse);
        }
    }

    // Should receive ~5 VSync messages (500ms / 100ms = 5)
    assert!(
        vsyncs.len() >= 4 && vsyncs.len() <= 6,
        "Expected ~5 VSync messages, got {}",
        vsyncs.len()
    );

    // Check intervals between consecutive VSync messages
    for i in 1..vsyncs.len() {
        let interval = vsyncs[i].timestamp - vsyncs[i - 1].timestamp;
        let interval_ms = interval.as_millis();
        assert!(
            interval_ms >= 90 && interval_ms <= 110,
            "Interval {}ms not in range [90, 110] (index {})",
            interval_ms,
            i
        );
    }
}

#[test]
fn test_vsync_token_bucket_backpressure() {
    // Spawn VSync at 100Hz (fast rate, 10ms interval)
    let (mock_handle, vsync_rx) = MockEngineHandle::new();
    let mock_handle = Arc::new(mock_handle);
    let mock_clone = mock_handle.clone();

    let _vsync = VsyncActor::spawn_with_handler(100.0, move |ctrl| {
        mock_clone.send(ctrl).map_err(|_| anyhow::anyhow!("send failed"))
    });

    // DON'T send any RenderedResponse feedback
    // Wait for tokens to be consumed
    std::thread::sleep(Duration::from_millis(100));

    let vsyncs = collect_all_vsyncs(&vsync_rx);

    // Should get EXACTLY 3 VSync messages (MAX_TOKENS)
    assert_eq!(
        vsyncs.len(),
        3,
        "Expected exactly 3 VSync (token starvation), got {}",
        vsyncs.len()
    );

    // All 3 should arrive quickly (token bucket drains fast when tokens available)
    if vsyncs.len() >= 3 {
        let first_ts = vsyncs[0].timestamp;
        let last_ts = vsyncs[2].timestamp;
        let elapsed = last_ts - first_ts;
        assert!(
            elapsed < Duration::from_millis(50),
            "Token bucket should drain fast, took {:?}",
            elapsed
        );
    }
}

#[test]
fn test_vsync_token_replenishment() {
    // Spawn VSync at 100Hz
    let (mock_handle, vsync_rx) = MockEngineHandle::new();
    let mock_handle = Arc::new(mock_handle);
    let mock_clone = mock_handle.clone();

    let vsync = VsyncActor::spawn_with_handler(100.0, move |ctrl| {
        mock_clone.send(ctrl).map_err(|_| anyhow::anyhow!("send failed"))
    });

    // Let tokens be exhausted (first 3 VSync)
    std::thread::sleep(Duration::from_millis(50));
    let initial = collect_all_vsyncs(&vsync_rx);
    assert_eq!(initial.len(), 3, "Should exhaust 3 tokens");

    // Send feedback to restore 1 token
    vsync.send(RenderedResponse).unwrap();

    // Wait and check for 4th VSync
    std::thread::sleep(Duration::from_millis(50));
    let after_feedback = collect_all_vsyncs(&vsync_rx);

    assert_eq!(
        after_feedback.len(),
        1,
        "Expected 1 more VSync after token replenishment, got {}",
        after_feedback.len()
    );
}

#[test]
fn test_vsync_missed_tick_recovery() {
    // VSync at 10Hz (100ms interval)
    let (mock_handle, vsync_rx) = MockEngineHandle::new();
    let mock_handle = Arc::new(mock_handle);
    let mock_clone = mock_handle.clone();

    let vsync = VsyncActor::spawn_with_handler(10.0, move |ctrl| {
        mock_clone.send(ctrl).map_err(|_| anyhow::anyhow!("send failed"))
    });

    // Exhaust tokens (wait for 3 intervals @ 100ms each = 300ms+buffer)
    std::thread::sleep(Duration::from_millis(350));
    let initial = collect_all_vsyncs(&vsync_rx);
    assert_eq!(initial.len(), 3, "Should exhaust tokens");

    // Miss ticks (250ms = 2.5 intervals)
    std::thread::sleep(Duration::from_millis(250));

    // Restore token and check for immediate VSync
    let before_feedback = Instant::now();
    vsync.send(RenderedResponse).unwrap();

    // Should get VSync immediately (missed tick recovery)
    let next_vsync = wait_for_vsync(&vsync_rx, Duration::from_millis(50))
        .expect("Should receive VSync after token restore");

    let recovery_time = next_vsync.timestamp - before_feedback;

    assert!(
        recovery_time < Duration::from_millis(20),
        "Missed tick recovery should be immediate, took {:?}",
        recovery_time
    );
}

#[test]
fn test_vsync_timing_accuracy() {
    // VSync at 60Hz (16.67ms interval)
    let (mock_handle, vsync_rx) = MockEngineHandle::new();
    let mock_handle = Arc::new(mock_handle);
    let mock_clone = mock_handle.clone();

    let vsync = VsyncActor::spawn_with_handler(60.0, move |ctrl| {
        mock_clone.send(ctrl).map_err(|_| anyhow::anyhow!("send failed"))
    });

    // Collect VSync and send immediate feedback (no backpressure)
    let mut vsyncs = Vec::new();
    let target_count = 120; // 2 seconds worth

    for _ in 0..target_count {
        if let Some(v) = wait_for_vsync(&vsync_rx, Duration::from_millis(50)) {
            vsyncs.push(v);
            // Send immediate feedback to prevent token starvation
            let _ = vsync.send(RenderedResponse);
        }
    }

    assert!(
        vsyncs.len() >= 100,
        "Should collect most VSync messages, got {}",
        vsyncs.len()
    );

    // Calculate intervals
    let intervals: Vec<Duration> = vsyncs
        .windows(2)
        .map(|w| w[1].timestamp - w[0].timestamp)
        .collect();

    // Mean interval should be ~16.67ms
    let mean_ms = intervals.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>()
        / intervals.len() as f64;

    assert!(
        (mean_ms - 16.67).abs() < 1.0,
        "Mean interval {:.2}ms, expected ~16.67ms",
        mean_ms
    );

    // Max jitter should be reasonable
    let max_jitter = intervals
        .iter()
        .map(|d| (d.as_secs_f64() * 1000.0 - 16.67).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    assert!(
        max_jitter < 3.0,
        "Max jitter {:.2}ms too high",
        max_jitter
    );
}
