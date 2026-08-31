//! The scheduler's doorbell: one word of state plus a parked thread.
//!
//! # Denotation
//!
//! A doorbell is not a queue, and `Wake`/`Shutdown` were never messages. A wake
//! is a **level** — "there is work" — which coalesces by definition: asserting
//! it twice says nothing the first assertion didn't. Shutdown is a **latch** —
//! "stop, permanently" — which cannot be lost or delayed by construction once
//! it is a sticky bit. Both live in one atomic word, so a single RMW reads the
//! whole system state and "shutdown outranks wake" is a bit test, not queue
//! ordering. The previous implementation (a `sync_channel(1)` carrying a
//! `System` enum) encoded both as queue items, which is where the capacity-1
//! contention, the blocking shutdown send, and the wake-before-and-after dance
//! around it all came from.
//!
//! # The protocol
//!
//! Producer side ([`Ring`]): make the work visible (e.g. the SPSC ring's
//! Release tail store), **then** `ring()`. Ring sets [`NOTIFIED`]; if it
//! observes [`SLEEPING`], it unparks the consumer.
//!
//! Consumer side ([`Doorbell`]): `poll()` consumes [`NOTIFIED`] with one RMW
//! and reports what it found. `wait()` polls, and on `Quiet` commits to sleep:
//! it publishes its thread handle, sets [`SLEEPING`] with an RMW, and parks
//! only if that RMW saw no pending bit.
//!
//! # Why this needs no fences
//!
//! Every transition that matters is an RMW on the one `state` word, and RMWs
//! always read the **latest** value in the word's modification order — an RMW
//! cannot act on a stale view the way a plain load can. That closes the
//! StoreLoad window the mpsc doorbell had (its `try_send` Full path was
//! loads-only, so a producer could observe "wake pending" before its own lane
//! publish was globally visible, while the consumer drained stale rings and
//! slept):
//!
//! - The consumer's sleep-commit (`fetch_or(SLEEPING)`) and a producer's
//!   `fetch_or(NOTIFIED)` are totally ordered by modification order. If the
//!   ring came first, the sleep-commit returns `NOTIFIED` and the consumer
//!   skips the park. If the sleep-commit came first, the ring returns
//!   `SLEEPING` and unparks. No interleaving parks on a set bit.
//! - Visibility of the work itself rides the same word: the AcqRel RMWs form a
//!   release sequence, so a consumer whose `poll()` RMW is ordered after a
//!   producer's `ring()` RMW also sees every write that producer made before
//!   ringing — the lane publish included.
//!
//! An unpark aimed at a thread that is not (or no longer) parked just grants a
//! park permit, producing at worst one spurious wake; both `wait()` and
//! `std::thread::park`'s contract tolerate that, and the loop re-polls.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, Thread};

/// Work is pending. Set by [`Ring::ring`], consumed by [`Doorbell::poll`].
const NOTIFIED: u32 = 1 << 0;
/// The consumer has committed to parking. Set and cleared only by
/// [`Doorbell::wait`]; producers that observe it unpark the sleeper.
const SLEEPING: u32 = 1 << 1;
/// Shutdown requested. Sticky — set once by [`Ring::shutdown`], never cleared.
const SHUTDOWN: u32 = 1 << 2;
/// Every [`Ring`] has been dropped: nothing can ever ring again. Sticky.
const ORPHANED: u32 = 1 << 3;
/// The [`Doorbell`] has been dropped: nobody is listening. Sticky.
const ABANDONED: u32 = 1 << 4;

struct Shared {
    state: AtomicU32,
    /// Live [`Ring`] handles. The last one's Drop sets [`ORPHANED`].
    rings: AtomicUsize,
    /// Where an unpark must be delivered. Written by the consumer before each
    /// sleep-commit (the consumer may migrate threads between waits), read by
    /// producers only after observing [`SLEEPING`] — which synchronizes with
    /// the registration, so the handle they read is the current one.
    sleeper: Mutex<Option<Thread>>,
}

impl Shared {
    fn unpark_sleeper(&self) {
        if let Some(t) = self
            .sleeper
            .lock()
            .expect("doorbell sleeper mutex poisoned")
            .as_ref()
        {
            t.unpark();
        }
    }
}

/// What the consumer found on its doorbell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Chime {
    /// Work is pending: drain the lanes.
    Work,
    /// Shutdown was requested. Outranks everything; sticky, so seeing it once
    /// is seeing it forever.
    Shutdown,
    /// Every [`Ring`] is gone and no work bit was set: after a final drain of
    /// whatever the lanes still buffer, nothing can ever arrive again.
    /// Reported only after pending [`Chime::Work`] has been consumed, matching
    /// the old mpsc doorbell's drain-before-disconnect order.
    Orphaned,
    /// Nothing. Returned by [`Doorbell::poll`] only — [`Doorbell::wait`]
    /// blocks instead.
    Quiet,
}

/// Producer end of a doorbell. Clone one per party that can make the consumer
/// runnable; the last Drop is what lets the consumer's `Orphaned` fire.
pub(crate) struct Ring {
    shared: Arc<Shared>,
}

/// Consumer end of a doorbell. Deliberately not `Clone`, and its methods take
/// `&mut self`: exactly one thread listens at a time (though the listener may
/// move between threads across calls).
pub(crate) struct Doorbell {
    shared: Arc<Shared>,
}

impl Doorbell {
    /// A fresh doorbell. Returns `(ring, doorbell)` — clone the [`Ring`] for
    /// each additional producer.
    pub(crate) fn new() -> (Ring, Doorbell) {
        let shared = Arc::new(Shared {
            state: AtomicU32::new(0),
            rings: AtomicUsize::new(1),
            sleeper: Mutex::new(None),
        });
        (
            Ring {
                shared: Arc::clone(&shared),
            },
            Doorbell { shared },
        )
    }

    /// Non-blocking check. Consumes a pending [`Chime::Work`].
    ///
    /// The RMW here is also the consumer's ordering edge: a caller that drains
    /// its lanes after `poll()` sees every publish made before the `ring()`
    /// this consumed (see the module doc), so drains need no fence of their
    /// own.
    pub(crate) fn poll(&mut self) -> Chime {
        let old = self.shared.state.fetch_and(!NOTIFIED, Ordering::AcqRel);
        if old & SHUTDOWN != 0 {
            Chime::Shutdown
        } else if old & NOTIFIED != 0 {
            Chime::Work
        } else if old & ORPHANED != 0 {
            Chime::Orphaned
        } else {
            Chime::Quiet
        }
    }

    /// Block until there is something to report. Never returns [`Chime::Quiet`].
    pub(crate) fn wait(&mut self) -> Chime {
        loop {
            match self.poll() {
                Chime::Quiet => {}
                chime => return chime,
            }

            // Commit to sleep. Publish where the unpark must land BEFORE
            // advertising SLEEPING — a producer only reads `sleeper` after its
            // RMW observed SLEEPING, which synchronizes with the fetch_or
            // below and therefore with this registration.
            *self
                .shared
                .sleeper
                .lock()
                .expect("doorbell sleeper mutex poisoned") = Some(thread::current());

            // The RMW reads the latest state: either nothing is pending and
            // SLEEPING is now visible to whichever ring comes next (it will
            // unpark us), or something landed since poll() and we skip the
            // park entirely. There is no window to sleep through.
            let old = self.shared.state.fetch_or(SLEEPING, Ordering::AcqRel);
            if old & (NOTIFIED | SHUTDOWN | ORPHANED) == 0 {
                thread::park();
            }
            self.shared.state.fetch_and(!SLEEPING, Ordering::AcqRel);
        }
    }
}

impl Drop for Doorbell {
    fn drop(&mut self) {
        // Lets Ring::ring / Ring::shutdown report that nobody is listening.
        self.shared.state.fetch_or(ABANDONED, Ordering::AcqRel);
    }
}

impl Ring {
    /// Assert "there is work". Never blocks; coalescing is inherent (a set bit
    /// stays one bit).
    ///
    /// Call this AFTER the work is visible (the lane's Release publish): this
    /// RMW is the producer's half of the no-stranded-message argument in the
    /// module doc.
    ///
    /// Returns `false` if the consumer is gone — ordinary for a retained
    /// [`Waker`](crate::Waker), a bug for an [`ActorHandle`](crate::ActorHandle)
    /// whose lanes just accepted a message; the caller decides.
    pub(crate) fn ring(&self) -> bool {
        let old = self.shared.state.fetch_or(NOTIFIED, Ordering::AcqRel);
        if old & SLEEPING != 0 {
            self.shared.unpark_sleeper();
        }
        old & ABANDONED == 0
    }

    /// Latch shutdown. Sticky, so delivery is guaranteed by construction —
    /// this never blocks and cannot be displaced by a pending wake, which is
    /// what the old capacity-1 queue's blocking send (and the platform-wake
    /// dance around it) existed to compensate for.
    ///
    /// Returns `false` if the consumer is already gone.
    pub(crate) fn shutdown(&self) -> bool {
        let old = self.shared.state.fetch_or(SHUTDOWN, Ordering::AcqRel);
        if old & SLEEPING != 0 {
            self.shared.unpark_sleeper();
        }
        old & ABANDONED == 0
    }
}

impl Clone for Ring {
    fn clone(&self) -> Self {
        self.shared.rings.fetch_add(1, Ordering::Relaxed);
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // Arc-style: Release on the decrement so the zero-observer's AcqRel
        // RMW on `state` is ordered after every ring this handle ever made.
        // No Acquire fence needed — the only thing touched afterwards is the
        // atomic state word itself.
        if self.shared.rings.fetch_sub(1, Ordering::Release) == 1 {
            let old = self.shared.state.fetch_or(ORPHANED, Ordering::AcqRel);
            if old & SLEEPING != 0 {
                self.shared.unpark_sleeper();
            }
        }
    }
}

impl std::fmt::Debug for Ring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ring").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Doorbell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Doorbell").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::time::Duration;

    #[test]
    fn quiet_when_nothing_happened() {
        let (_ring, mut bell) = Doorbell::new();
        assert_eq!(bell.poll(), Chime::Quiet);
    }

    #[test]
    fn ring_then_work_then_quiet() {
        let (ring, mut bell) = Doorbell::new();
        assert!(ring.ring());
        assert_eq!(bell.poll(), Chime::Work);
        assert_eq!(bell.poll(), Chime::Quiet);
    }

    #[test]
    fn rings_coalesce() {
        let (ring, mut bell) = Doorbell::new();
        assert!(ring.ring());
        assert!(ring.ring());
        assert!(ring.ring());
        assert_eq!(bell.poll(), Chime::Work);
        assert_eq!(bell.poll(), Chime::Quiet);
    }

    #[test]
    fn shutdown_is_sticky_and_outranks_work() {
        let (ring, mut bell) = Doorbell::new();
        assert!(ring.ring());
        assert!(ring.shutdown());
        assert_eq!(bell.poll(), Chime::Shutdown);
        // Sticky: still shutdown on every subsequent look.
        assert_eq!(bell.poll(), Chime::Shutdown);
        assert_eq!(bell.wait(), Chime::Shutdown);
    }

    #[test]
    fn pending_work_drains_before_orphaned() {
        let (ring, mut bell) = Doorbell::new();
        assert!(ring.ring());
        drop(ring);
        // Mirrors mpsc recv: buffered signal first, disconnect after.
        assert_eq!(bell.poll(), Chime::Work);
        assert_eq!(bell.poll(), Chime::Orphaned);
        assert_eq!(bell.poll(), Chime::Orphaned);
    }

    #[test]
    fn orphaned_only_when_last_ring_drops() {
        let (ring, mut bell) = Doorbell::new();
        let ring2 = ring.clone();
        drop(ring);
        assert_eq!(bell.poll(), Chime::Quiet);
        drop(ring2);
        assert_eq!(bell.poll(), Chime::Orphaned);
    }

    #[test]
    fn ring_reports_abandoned_consumer() {
        let (ring, bell) = Doorbell::new();
        assert!(ring.ring());
        drop(bell);
        assert!(!ring.ring());
        assert!(!ring.shutdown());
    }

    #[test]
    fn wait_blocks_until_rung() {
        let (ring, mut bell) = Doorbell::new();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            assert!(ring.ring());
        });
        assert_eq!(bell.wait(), Chime::Work);
        t.join().unwrap();
    }

    #[test]
    fn wait_wakes_on_shutdown() {
        let (ring, mut bell) = Doorbell::new();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            assert!(ring.shutdown());
        });
        assert_eq!(bell.wait(), Chime::Shutdown);
        t.join().unwrap();
    }

    #[test]
    fn wait_wakes_when_last_ring_drops() {
        let (ring, mut bell) = Doorbell::new();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            drop(ring);
        });
        assert_eq!(bell.wait(), Chime::Orphaned);
        t.join().unwrap();
    }

    /// The property the doorbell exists for: a value published before ring()
    /// is visible to the consumer that wait()/poll() releases — across many
    /// sleep/wake cycles, so both the parked and the caught-before-park paths
    /// get exercised.
    #[test]
    fn published_work_is_visible_after_wait() {
        let (ring, mut bell) = Doorbell::new();
        let published = Arc::new(AtomicU64::new(0));
        let seen_by_consumer = Arc::clone(&published);
        const ROUNDS: u64 = 10_000;

        let producer = std::thread::spawn(move || {
            for i in 1..=ROUNDS {
                // Publish, then ring — the producer contract.
                published.store(i, Ordering::Relaxed);
                assert!(ring.ring());
                if i % 64 == 0 {
                    std::thread::yield_now();
                }
            }
        });

        let mut last = 0u64;
        loop {
            match bell.wait() {
                Chime::Work => {
                    let v = seen_by_consumer.load(Ordering::Relaxed);
                    assert!(v >= last, "went backwards: {v} after {last}");
                    // v == last is legal (a coalesced re-ring); progress is
                    // guaranteed because the final publish precedes the final
                    // ring, whose Work we cannot have consumed yet if v < ROUNDS.
                    last = v;
                }
                Chime::Orphaned => break,
                other => panic!("unexpected chime: {other:?}"),
            }
        }
        assert_eq!(last, ROUNDS, "final publish lost");
        producer.join().unwrap();
    }
}
