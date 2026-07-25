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
//! (`docs/designs/pixelflow-runtime-engine-mesh-migration.md` §5 step 2) as a before/after
//! regression harness, and now exercises the real message-based `ReturnToken` that the refactor
//! added — previously untestable from outside the crate, since the only thing that returned a
//! token reached directly into a same-process global static.
//!
//! What it deliberately does *not* do: re-check "never exceeds the cap" or "`RenderedResponse`
//! doesn't return a token" by sleeping a fixed duration and rechecking a count. Those are
//! already proven deterministically, with no thread/clock/sleep involved, by `VsyncCore`'s own
//! unit tests in `vsync_actor.rs`. See `real_vsync_actor_token_bucket_behavior`'s doc comment
//! for why this file only asserts things eventually happening (via a poll-until-true with a
//! generous deadline), never things *not* happening by some wall-clock deadline.

use actor_scheduler::{Actor, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus};
use pixelflow_runtime::api::private::{EngineControl, EngineData, create_engine_actor};
use pixelflow_runtime::api::public::AppManagement;
use pixelflow_runtime::vsync_actor::{VsyncActor, VsyncCommand};
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
/// under sustained demand, and confirms `ReturnToken` unblocks exactly one more.
///
/// Deliberately does *not* re-prove "never exceeds the cap" or "`RenderedResponse` doesn't
/// return a token" here by sleeping some fixed duration and rechecking the count — that's an
/// assertion about event ordering racing against a real background clock thread, and it can
/// only ever be as good as how long you were willing to wait. Both invariants are already
/// proven deterministically, with no thread/clock/sleep in the loop, by `VsyncCore`'s own unit
/// tests in `vsync_actor.rs` (`credit_caps_ticks_at_max_tokens_with_no_clock_involved`,
/// `rendered_response_does_not_return_a_token`). This test's job is only to prove the real
/// wiring reaches that same core: a real `Start` produces a real tick, sustained real demand
/// reaches exactly the cap, and a real `ReturnToken` message unblocks a real tick — all
/// positive, "did this eventually happen" assertions via `wait_for_at_least`'s poll-until-true
/// (bounded by a generous deadline), never "did this NOT happen by the time I checked."
///
/// 1000Hz is far faster than any real display, specifically to reach `MAX_TOKENS` well within
/// the test's timeout. Deadlines are generous throughout: the clock thread's *actual* tick rate
/// under a shared/sandboxed scheduler can fall well short of the requested 1000Hz
/// (`recv_timeout`'s real granularity, CPU contention from other tests in the binary).
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

    // 2. Sustained demand reaches exactly the cap — `wait_for_at_least` returns as soon as the
    //    count crosses `MAX_TOKENS`, so an exact-equality check here already catches the bucket
    //    overshooting, with no separate "wait longer, recheck" step needed.
    let saturated = wait_for_at_least(&ticks, MAX_TOKENS, Instant::now() + Duration::from_secs(30));
    assert_eq!(
        saturated, MAX_TOKENS,
        "bucket should saturate at exactly its cap under sustained demand"
    );

    // 3. `ReturnToken` — a real message, not a same-process global mutation — unblocks exactly
    //    one more real tick. This is the capability the refactor added: previously the only
    //    thing that returned a token was `engine_troupe.rs` reaching directly into a
    //    process-global static, a path no test outside the crate could exercise.
    vsync
        .send(Message::Control(VsyncCommand::ReturnToken))
        .unwrap();
    let after_return =
        wait_for_at_least(&ticks, MAX_TOKENS + 1, Instant::now() + Duration::from_secs(30));
    assert_eq!(
        after_return,
        MAX_TOKENS + 1,
        "ReturnToken must unblock exactly one more tick"
    );

    vsync.send(Message::Control(VsyncCommand::Shutdown)).unwrap();
}
