//! Integration tests against the **real** `VsyncActor`, not a mock.
//!
//! `vsync_actor_tests.rs` says why it doesn't do this itself: "We can't easily test the real
//! VsyncActor because it requires EngineActorHandle, but we can test the message types and
//! patterns." That's a real, admitted gap — every existing vsync test exercises a hand-rolled
//! mock reimplementing the token-bucket logic, never the actual code in `vsync_actor.rs`.
//!
//! This closes it by constructing a real `EngineActorHandle` via
//! `api::private::create_engine_actor` and draining it with a small collector actor, so the
//! real `VsyncActor` has somewhere real to send its ticks. It was written *before*
//! `VsyncActor`'s internals moved onto `Transducer`/`Credit`
//! (`docs/designs/pixelflow-runtime-engine-mesh-migration.md` §5 step 2) specifically as a
//! before/after regression harness: run against the unconverted actor first to prove it
//! passes, then again after the refactor to prove nothing observable changed. It still passes,
//! unmodified — steps 1–3 below assert exactly what they asserted before the refactor. Step 4
//! is new: `ReturnToken`, a real message, replacing what used to be an untestable same-process
//! global mutation.
//!
//! # Why this is one `#[test]`, not several
//!
//! `VSYNC_TOKEN_BUCKET` is one process-wide global static — not per-`VsyncActor` state. Every
//! `VsyncActor` instance in this test **binary** shares the exact same atomic, it only ever
//! decreases (nothing reachable from an integration test replenishes it — see below), and
//! cargo runs `#[test]` functions in parallel and in unspecified order by default. Splitting
//! this into separate tests asserting exact counts (e.g. "exhausts at exactly 100") would pass
//! or fail depending on what *other* tests in this binary happened to run first and how much of
//! the shared bucket they'd already spent — a real, deterministic cross-test coupling bug, not
//! flakiness to paper over with longer timeouts. One test, one `VsyncActor`, sequential
//! sections, is the only way to get a clean measurement against a bucket that starts full.
//!
//! (This coupling is itself evidence for the migration: converting the bucket to per-instance
//! `Credit` removes it entirely — each actor gets its own, unshared state, and this whole
//! caveat stops applying.)

use actor_scheduler::{Actor, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus};
use pixelflow_runtime::api::private::{EngineControl, EngineData, create_engine_actor};
use pixelflow_runtime::api::public::AppManagement;
use pixelflow_runtime::vsync_actor::{RenderedResponse, VsyncActor, VsyncCommand};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// `VsyncActor`'s own token bucket cap. Not exported by the crate — duplicated here
/// deliberately, the same way the existing mock tests duplicate it as a local `MAX_TOKENS`,
/// so this file doesn't need `vsync_actor` to expose an implementation detail just for tests.
const MAX_TOKENS: usize = 100;

/// Collects every `EngineData::VSync` tick a real `VsyncActor` sends; ignores everything else.
struct TickCollector {
    ticks: Arc<Mutex<Vec<Instant>>>,
}

impl Actor<EngineData, EngineControl, AppManagement> for TickCollector {
    fn handle_data(&mut self, data: EngineData) -> HandlerResult {
        if let EngineData::VSync { timestamp, .. } = data {
            self.ticks.lock().unwrap().push(timestamp);
        }
        Ok(())
    }
    fn handle_control(&mut self, _: EngineControl) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _: AppManagement) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

/// Poll `ticks` until it has at least `n` entries or `deadline` passes.
fn wait_for_at_least(ticks: &Arc<Mutex<Vec<Instant>>>, n: usize, deadline: Instant) -> usize {
    loop {
        let count = ticks.lock().unwrap().len();
        if count >= n || Instant::now() >= deadline {
            return count;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

/// Exercises a single real `VsyncActor` end to end: starts, ticks, saturates its token bucket
/// under sustained demand, and confirms `RenderedResponse` does not (today) return a token.
///
/// 1000Hz is far faster than any real display, specifically to blow through `MAX_TOKENS` well
/// within the test's timeout. Deadlines are generous throughout: the clock thread's *actual*
/// tick rate under a shared/sandboxed scheduler can fall well short of the requested 1000Hz
/// (`recv_timeout`'s real granularity, CPU contention from other tests in the binary) — these
/// are correctness properties (does it cap, does it stay capped), not timing ones.
#[test]
fn real_vsync_actor_token_bucket_behavior() {
    let (engine_handle, mut engine_sched) = create_engine_actor(None);
    let ticks = Arc::new(Mutex::new(Vec::new()));
    let ticks_clone = ticks.clone();

    thread::Builder::new()
        .name("test-engine-collector".to_string())
        .spawn(move || {
            let mut collector = TickCollector { ticks: ticks_clone };
            engine_sched.run(&mut collector);
        })
        .expect("failed to spawn engine collector thread");

    let vsync = VsyncActor::spawn(1000.0, engine_handle);

    // 1. A started actor ticks at least once.
    vsync.send(Message::Control(VsyncCommand::Start)).unwrap();
    let first = wait_for_at_least(&ticks, 1, Instant::now() + Duration::from_secs(5));
    assert!(first > 0, "a started VsyncActor must tick at least once");

    // 2. Sustained demand saturates the bucket at exactly its cap, never beyond.
    let saturated = wait_for_at_least(&ticks, MAX_TOKENS, Instant::now() + Duration::from_secs(30));
    assert_eq!(
        saturated, MAX_TOKENS,
        "bucket should saturate at exactly its cap under sustained demand"
    );

    // Give the clock thread ample further opportunity; it must not exceed the cap.
    thread::sleep(Duration::from_millis(200));
    assert_eq!(
        ticks.lock().unwrap().len(),
        MAX_TOKENS,
        "must not tick again once the bucket is empty"
    );

    // 3. `RenderedResponse` does **not** replenish the bucket today — a real, slightly
    //    surprising finding, not a guess. `VsyncActor::handle_data`'s own comment says "Token
    //    management is now handled via atomic bucket"; the only thing that actually calls
    //    `return_vsync_token()` is `engine_troupe.rs`'s `handle_app_data`, reaching directly
    //    into the same-process global static — a path this test cannot reach (the function is
    //    `pub(crate)`, and the global it touches is a same-process implementation detail).
    //    Documented here as *current, about-to-change* behavior: converting the bucket to a
    //    real `Credit` means the return must become an actual message (there is no
    //    process-global left to reach into), which is expected to flip this specific
    //    assertion once that lands — the intended fix, not a regression.
    vsync
        .send(Message::Data(RenderedResponse {
            frame_number: 1,
            rendered_at: Instant::now(),
        }))
        .unwrap();
    thread::sleep(Duration::from_secs(2));
    assert_eq!(
        ticks.lock().unwrap().len(),
        MAX_TOKENS,
        "RenderedResponse must not affect the token bucket today — only engine_troupe.rs's \
         direct global-static call does, which this test cannot reach"
    );

    // 4. What engine_troupe.rs's direct global-static call could never let this test exercise:
    //    a genuine credit return, now that it travels as a real message instead. Exactly one
    //    more tick must land, not a flood — the same single-token semantics as before, finally
    //    provable from outside the crate.
    vsync
        .send(Message::Control(VsyncCommand::ReturnToken))
        .unwrap();
    let after_return =
        wait_for_at_least(&ticks, MAX_TOKENS + 1, Instant::now() + Duration::from_secs(30));
    assert_eq!(
        after_return,
        MAX_TOKENS + 1,
        "ReturnToken must unblock exactly one more tick — this is the capability the refactor \
         added: a real, externally-testable credit return"
    );

    vsync.send(Message::Control(VsyncCommand::Shutdown)).unwrap();
}
