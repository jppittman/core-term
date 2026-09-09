//! A timer that fires on a schedule.
//!
//! Actors can't wait on time themselves: a scheduler blocks on its doorbell, so an actor that
//! wanted to wake "in 16ms" would have to block the very loop that delivers its messages. The
//! way out is the one this crate already uses everywhere else — *something else sends it a
//! message*. A [`Timer`] is that something: a thread whose only job is to wait, fire, and wait
//! again.
//!
//! ```ignore
//! let timer = Timer::spawn("vsync-clock", Schedule::Every(interval), move || {
//!     match handle.send(VsyncManagement::Tick) {
//!         Ok(()) => ControlFlow::Continue(()),
//!         // The actor is gone; there is nobody left to tick.
//!         Err(_) => ControlFlow::Break(()),
//!     }
//! });
//! ```
//!
//! The callback returns [`ControlFlow`] rather than `bool` because "keep going / stop" is
//! exactly what `ControlFlow` means, and a `-> bool` would need a doc comment explaining which
//! way round it goes.
//!
//! # Lifetime
//!
//! A timer stops when any of these happens, whichever comes first:
//!
//! - [`Timer::stop`] is called — the only one that is *synchronous*: it joins the thread, so
//!   when it returns the timer has provably fired for the last time.
//! - the callback returns [`ControlFlow::Break`].
//! - the [`Timer`] is dropped — the command channel disconnects and the thread notices.
//!
//! Dropping is the loose one: the thread may be mid-wait, so a fire already in flight can still
//! land. Use [`stop`](Timer::stop) when "no more fires after this point" has to be a fact rather
//! than a very likely outcome — a test asserting an exact fire count needs that guarantee, and
//! so does any teardown that tears down something the callback touches.

use std::ops::ControlFlow;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// When a [`Timer`] fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// Fire repeatedly, waiting this long before each fire.
    ///
    /// The wait is between fires, not between fire *starts*: a slow callback delays the next
    /// fire rather than stacking up behind it. A timer never runs its callback re-entrantly.
    Every(Duration),
    /// Fire once, after this long, then stop.
    After(Duration),
}

/// Commands to a running timer thread.
enum TimerCommand {
    /// Change the wait. Takes effect on the *next* wait, not the one already in progress.
    SetInterval(Duration),
    /// Stop the timer.
    Stop,
}

/// A handle to a running timer thread.
///
/// See the [module docs](self) for the schedule and lifetime rules.
#[derive(Debug)]
#[must_use = "dropping a Timer stops it; bind it to a variable to keep it running"]
pub struct Timer {
    tx: Sender<TimerCommand>,
    /// `None` only after [`stop`](Timer::stop) has taken it to join.
    join: Option<JoinHandle<()>>,
}

impl Timer {
    /// Spawn a timer thread that runs `on_fire` on the given schedule.
    ///
    /// `name` names the OS thread, so it shows up in debuggers and profilers.
    ///
    /// # Panics
    ///
    /// Panics if the OS refuses to spawn the thread.
    pub fn spawn<F>(name: &str, schedule: Schedule, mut on_fire: F) -> Self
    where
        F: FnMut() -> ControlFlow<()> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();

        let (mut interval, repeats) = match schedule {
            Schedule::Every(d) => (d, true),
            Schedule::After(d) => (d, false),
        };

        let join = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                loop {
                    match rx.recv_timeout(interval) {
                        Ok(TimerCommand::SetInterval(d)) => interval = d,
                        Ok(TimerCommand::Stop) => break,
                        // The wait elapsed uninterrupted: that *is* the tick.
                        Err(RecvTimeoutError::Timeout) => {
                            if on_fire().is_break() || !repeats {
                                break;
                            }
                        }
                        // The `Timer` was dropped, so nobody can stop us later either.
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .expect("failed to spawn timer thread");

        Self {
            tx,
            join: Some(join),
        }
    }

    /// Send a command, tolerating a timer thread that has already exited.
    ///
    /// Both commands are *moot* rather than failed in that state: a stopped timer has no next
    /// wait to reschedule and nothing left to stop. Since the caller could take no useful
    /// action on the error, it is handled here, once, instead of being propagated to every
    /// call site that would only discard it.
    fn command(&self, cmd: TimerCommand) {
        if self.tx.send(cmd).is_err() {
            // Already stopped — the callback broke, or a one-shot fired. Nothing to do.
        }
    }

    /// Change how long the timer waits between fires.
    ///
    /// Applies to the next wait; a wait already in progress runs out at its original length.
    /// Does nothing if the timer has already stopped — a timer outliving its usefulness is
    /// ordinary, not an error worth propagating to every caller.
    pub fn set_interval(&self, interval: Duration) {
        self.command(TimerCommand::SetInterval(interval));
    }

    /// Stop the timer and wait for its thread to exit.
    ///
    /// When this returns, the callback has provably run for the last time. That guarantee is
    /// the reason to prefer this over just dropping the timer.
    ///
    /// # Panics
    ///
    /// Panics if the timer thread panicked — i.e. if `on_fire` panicked.
    pub fn stop(mut self) {
        self.command(TimerCommand::Stop);
        // Runs whether or not the command landed, so the "provably fired for the last time"
        // guarantee holds identically for a thread that had already exited on its own.
        if let Some(join) = self.join.take() {
            join.join().expect("timer callback panicked");
        }
    }
}

impl crate::mealy::PortTarget<Duration> for Timer {
    fn try_deliver(&self, msg: Duration) -> Result<(), crate::spsc::TrySendError<Duration>> {
        self.set_interval(msg);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Long enough that a test never waits on it, short enough that a test that *wants* a fire
    /// isn't slow. Nothing asserts a fire count against elapsed time, so this only trades test
    /// latency against CPU churn — never correctness.
    const QUICK: Duration = Duration::from_millis(5);

    /// Far longer than any test's own runtime: a timer on this schedule will not fire during a
    /// test unless the test explicitly waits for it, which none do.
    const NEVER_WITHIN_THIS_TEST: Duration = Duration::from_secs(3600);

    fn counter() -> (Arc<AtomicUsize>, impl FnMut() -> ControlFlow<()>) {
        let count = Arc::new(AtomicUsize::new(0));
        let handle = count.clone();
        (count, move || {
            handle.fetch_add(1, Ordering::SeqCst);
            ControlFlow::Continue(())
        })
    }

    /// Blocks until `count` reaches `target`. No deadline: a timer that never fires must hang
    /// the test rather than race a deadline that a loaded CI runner can lose. The test harness
    /// timeout is the real backstop, and it doesn't produce false failures.
    fn wait_until_at_least(count: &AtomicUsize, target: usize) {
        while count.load(Ordering::SeqCst) < target {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn a_repeating_timer_fires_more_than_once() {
        let (count, on_fire) = counter();
        let timer = Timer::spawn("test-repeat", Schedule::Every(QUICK), on_fire);

        wait_until_at_least(&count, 2);

        timer.stop();
    }

    #[test]
    fn a_one_shot_timer_fires_exactly_once() {
        let (count, on_fire) = counter();
        let timer = Timer::spawn("test-one-shot", Schedule::After(QUICK), on_fire);

        // Wait for the fire *before* stopping: `stop` is immediate, so stopping first would
        // beat the schedule and prove nothing. Then `stop` joins — so the exact count below is
        // a fact about a thread that has provably exited, not a race against a deadline.
        wait_until_at_least(&count, 1);
        timer.stop();

        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn breaking_from_the_callback_stops_a_repeating_timer() {
        let count = Arc::new(AtomicUsize::new(0));
        let handle = count.clone();

        let timer = Timer::spawn("test-break", Schedule::Every(QUICK), move || {
            handle.fetch_add(1, Ordering::SeqCst);
            ControlFlow::Break(())
        });

        wait_until_at_least(&count, 1);
        timer.stop();

        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "Break must stop a repeating timer after its first fire"
        );
    }

    #[test]
    fn set_interval_shortens_a_timer_that_would_not_otherwise_fire() {
        let (count, on_fire) = counter();
        let timer = Timer::spawn(
            "test-set-interval",
            Schedule::Every(NEVER_WITHIN_THIS_TEST),
            on_fire,
        );

        // The timer is parked on an hour-long wait, so any fire at all proves the new interval
        // took effect — no timing assertion needed, and no way for a slow runner to fake it.
        timer.set_interval(QUICK);
        wait_until_at_least(&count, 1);

        timer.stop();
    }

    #[test]
    fn stop_is_safe_before_the_first_fire() {
        let (count, on_fire) = counter();
        let timer = Timer::spawn(
            "test-stop-early",
            Schedule::Every(NEVER_WITHIN_THIS_TEST),
            on_fire,
        );

        timer.stop();

        assert_eq!(count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn dropping_a_timer_stops_its_thread() {
        let (count, on_fire) = counter();

        {
            let _timer = Timer::spawn(
                "test-drop",
                Schedule::Every(NEVER_WITHIN_THIS_TEST),
                on_fire,
            );
        }

        // Nothing to join on a dropped timer, so this asserts the weaker of the two guarantees
        // documented in the module docs: a dropped timer that never fired still never fires.
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }
}
