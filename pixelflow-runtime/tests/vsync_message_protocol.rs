//! The vsync message protocol, seen from outside the crate.
//!
//! What is left here is what an integration test can say that an in-module test cannot: the
//! `impl_*_message!` conversions and the derived impls are reachable, and land in the lane
//! they claim, for a caller who only has the public types.
//!
//! Everything else this file used to hold has been deleted, along with the whole of
//! `vsync_actor_bug_hunting_tests.rs`. Both files stood up an actor **defined in the test
//! file** — a `MockVsyncActor`, a `TickRateTracker`, a `TokenTracker` — and then asserted
//! things about it. None of them ever constructed `VsyncActor`, so no change to
//! `vsync_actor.rs` could fail them. Three ways that went wrong, all of them worth naming:
//!
//! - **The mock drifted into a false spec.** It hardcoded `MAX_TOKENS = 3` (production is
//!   100) and replenished credit on `RenderedResponse` — which production deliberately does
//!   *not* do, as `rendered_response_does_not_return_a_token` in `vsync_actor.rs` pins. Four
//!   tests asserted that behavior, so the suite documented the opposite of the contract.
//! - **Cross-lane ordering was asserted as a barrier.** `shutdown_command_stops_tick_processing`
//!   sent ticks, then a shutdown, then more ticks, and asserted the later ticks were not
//!   counted. The scheduler promises best-effort priority among *simultaneously pending*
//!   messages, never that a `Control` message preempts a Management drain already in flight —
//!   `a_control_message_enqueued_mid_drain_is_not_seen_until_the_drain_ends` (actor-scheduler)
//!   is the deterministic counterexample. It broke CI, which is how this review started.
//! - **Nine tests asserted properties of the standard library.** `Duration::from_secs_f64(inf)`
//!   panics; `u64::MAX.wrapping_add(1) == 0`; `Instant::elapsed` saturates; a `1.0 / 0.0` is
//!   infinite; `mpsc::Sender::send` fails once its receiver is dropped. True, and untouchable
//!   by anything in this repository.
//!
//! The real actor is tested where it can be tested honestly: `VsyncCore` in
//! `pixelflow-runtime/src/vsync_actor.rs` steps through `step_control`/`step_management`
//! directly, with no thread, no clock, and no sleep — so the credit bound is checked by
//! counting, not by waiting.

use std::time::Instant;

use actor_scheduler::Message;
use pixelflow_runtime::vsync_actor::{RenderedResponse, VsyncCommand, VsyncManagement};

#[test]
fn a_command_converts_into_the_control_lane() {
    let msg: Message<RenderedResponse, VsyncCommand, VsyncManagement> = VsyncCommand::Start.into();
    assert!(matches!(msg, Message::Control(VsyncCommand::Start)));
}

#[test]
fn a_rendered_response_converts_into_the_data_lane() {
    let msg: Message<RenderedResponse, VsyncCommand, VsyncManagement> = RenderedResponse {
        frame_number: 1,
        rendered_at: Instant::now(),
    }
    .into();
    assert!(matches!(msg, Message::Data(_)));
}

#[test]
fn a_tick_converts_into_the_management_lane() {
    let msg: Message<RenderedResponse, VsyncCommand, VsyncManagement> =
        VsyncManagement::Tick.into();
    assert!(matches!(msg, Message::Management(VsyncManagement::Tick)));
}

#[test]
fn command_debug_output_is_the_variant_name() {
    assert_eq!(format!("{:?}", VsyncCommand::Start), "Start");
    assert_eq!(format!("{:?}", VsyncCommand::Stop), "Stop");
    assert_eq!(format!("{:?}", VsyncCommand::Shutdown), "Shutdown");
    assert_eq!(
        format!("{:?}", VsyncCommand::UpdateRefreshRate(120.0)),
        "UpdateRefreshRate(120.0)"
    );
}

/// The default matters: a `VsyncCommand` materialized by `Default` — a port's empty slot, a
/// zeroed message — must be the harmless one, not an accidental `Start`.
#[test]
fn the_default_command_is_shutdown() {
    assert!(matches!(VsyncCommand::default(), VsyncCommand::Shutdown));
}
