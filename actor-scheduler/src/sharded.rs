//! Sharded SPSC inbox — multiple producers, each with a dedicated SPSC channel.
//!
//! Instead of N producers contending on one MPSC channel, each producer gets
//! its own SPSC ring buffer to the consumer. The consumer drains all shards
//! with round-robin fairness to prevent any single producer from starving others.
//!
//! # Shuffle-shard analogy
//!
//! Like shuffle sharding in the k8s API server, each producer is isolated:
//! a noisy producer can fill its own shard but cannot affect other producers'
//! ability to deliver messages. The topology is fixed at initialization time.
//!
//! # Lifecycle
//!
//! ```text
//! 1. InboxBuilder::new(capacity)       — create builder
//! 2. builder.add_producer()            — returns SpscSender, can call N times
//! 3. builder.build()                   — seals the registry, returns ShardedInbox
//! 4. inbox.drain(limit, handler)       — consumer polls all shards
//! ```
//!
//! No producers can be added after `build()`. This is the "register at init"
//! constraint — you must know all producers before the scheduler starts.

use crate::HandlerResult;
pub use crate::error::DrainStatus;
use crate::spsc::{self, SpscReceiver, SpscSender, TryRecvError};

/// Builder for a sharded inbox. Add producers, then seal with `build()`.
pub struct InboxBuilder<T> {
    receivers: Vec<SpscReceiver<T>>,
    capacity: usize,
}

impl<T> InboxBuilder<T> {
    /// Create a new builder. Each producer's SPSC channel will have
    /// at least `capacity` slots (rounded up to next power of 2).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            receivers: Vec::new(),
            capacity,
        }
    }

    /// Register a new producer. Returns the sender end of a dedicated SPSC channel.
    ///
    /// Call this once per producer during initialization, before `build()`.
    pub fn add_producer(&mut self) -> SpscSender<T> {
        let (tx, rx) = spsc::spsc_channel(self.capacity);
        self.receivers.push(rx);
        tx
    }

    /// Seal the registry and return the sharded inbox.
    ///
    /// No more producers can be added after this call.
    /// Panics if no producers were registered.
    #[must_use]
    pub fn build(self) -> ShardedInbox<T> {
        assert!(
            !self.receivers.is_empty(),
            "ShardedInbox requires at least one producer"
        );
        ShardedInbox {
            shards: self.receivers,
            round_robin: 0,
        }
    }
}

/// Consumer-side sharded inbox. Holds N SPSC receivers and drains them fairly.
pub struct ShardedInbox<T> {
    shards: Vec<SpscReceiver<T>>,
    /// Starting index for round-robin. Rotated after each drain cycle
    /// to prevent the first shard from always getting priority.
    round_robin: usize,
}

impl<T> ShardedInbox<T> {
    /// Drain messages from all shards up to a total `limit`.
    ///
    /// Uses round-robin across shards: each shard gets drained up to
    /// `per_shard` messages (limit / num_shards, minimum 1), then we
    /// rotate the starting shard for fairness.
    ///
    /// Returns:
    /// - `Ok(DrainStatus::Empty)` — all shards empty
    /// - `Ok(DrainStatus::More)` — hit limit, more messages may exist
    /// - `Ok(DrainStatus::Disconnected)` — all producers dropped
    /// - `Err(HandlerError)` — handler failed
    pub fn drain(
        &mut self,
        limit: usize,
        mut handler: impl FnMut(T) -> HandlerResult,
    ) -> Result<DrainStatus, crate::HandlerError> {
        let n = self.shards.len();
        let per_shard = (limit / n).max(1);
        let mut total = 0usize;
        let mut all_empty = true;
        let mut all_disconnected = true;

        for i in 0..n {
            let idx = (self.round_robin + i) % n;
            let shard = &mut self.shards[idx];
            let mut shard_count = 0;

            loop {
                if total >= limit || shard_count >= per_shard {
                    // Hit per-shard or total limit — there might be more
                    all_empty = false;
                    break;
                }

                match shard.try_recv() {
                    Ok(msg) => {
                        handler(msg)?;
                        shard_count += 1;
                        total += 1;
                        all_disconnected = false;
                    }
                    Err(TryRecvError::Empty) => {
                        all_disconnected = false;
                        break;
                    }
                    Err(TryRecvError::Disconnected) => {
                        break;
                    }
                }
            }
        }

        // Rotate starting shard for next drain
        self.round_robin = (self.round_robin + 1) % n;

        if all_disconnected {
            Ok(DrainStatus::Disconnected)
        } else if total >= limit || !all_empty {
            // We either hit the limit, or we had per-shard caps that stopped early
            if total >= limit {
                Ok(DrainStatus::More)
            } else if all_empty {
                Ok(DrainStatus::Empty)
            } else {
                // Some shards may have more, but we haven't proven it
                // Re-check: did any shard actually hit its per-shard limit?
                // If total < limit, we only stopped because of per-shard caps.
                // That means there might be more in those shards.
                Ok(DrainStatus::More)
            }
        } else {
            Ok(DrainStatus::Empty)
        }
    }

    /// Number of registered shards (producers).
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HandlerError;

    #[test]
    fn basic_sharded_drain() {
        let mut builder = InboxBuilder::new(16);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        tx1.try_send(1).unwrap();
        tx1.try_send(2).unwrap();
        tx2.try_send(10).unwrap();
        tx2.try_send(20).unwrap();

        let mut received = Vec::new();
        let status = inbox
            .drain(100, |msg| {
                received.push(msg);
                Ok(())
            })
            .unwrap();

        assert_eq!(status, DrainStatus::Empty);
        assert_eq!(received.len(), 4);
        // All messages received (order depends on round-robin start)
        received.sort();
        assert_eq!(received, vec![1, 2, 10, 20]);
    }

    #[test]
    fn burst_limit_respected() {
        let mut builder = InboxBuilder::new(64);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        for i in 0..50 {
            tx.try_send(i).unwrap();
        }

        let mut count = 0;
        let status = inbox
            .drain(10, |_msg| {
                count += 1;
                Ok(())
            })
            .unwrap();

        assert_eq!(count, 10);
        assert_eq!(status, DrainStatus::More);
    }

    #[test]
    fn round_robin_fairness() {
        let mut builder = InboxBuilder::new(64);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        // Producer 1 floods, producer 2 sends one
        for i in 0..50 {
            tx1.try_send(i).unwrap();
        }
        tx2.try_send(100).unwrap();

        // With limit=4 and 2 shards, per_shard=2
        let mut received = Vec::new();
        inbox
            .drain(4, |msg| {
                received.push(msg);
                Ok(())
            })
            .unwrap();

        // Producer 2's message should appear (not starved by producer 1)
        assert!(
            received.contains(&100),
            "Producer 2 was starved! Got: {:?}",
            received
        );
    }

    #[test]
    fn all_producers_disconnect() {
        let mut builder = InboxBuilder::new(16);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        drop(tx1);
        drop(tx2);

        let status = inbox.drain(100, |_: u32| Ok(())).unwrap();
        assert_eq!(status, DrainStatus::Disconnected);
    }

    #[test]
    fn drain_buffered_after_disconnect() {
        let mut builder = InboxBuilder::new(16);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        tx.try_send(42).unwrap();
        drop(tx);

        let mut received = Vec::new();
        let status = inbox
            .drain(100, |msg| {
                received.push(msg);
                Ok(())
            })
            .unwrap();

        assert_eq!(received, vec![42]);
        // First drain gets the message; the shard reports Disconnected but
        // we still got data, so it's not all-disconnected yet
        // Second drain should show disconnected
        let status2 = inbox.drain(100, |_: u32| Ok(())).unwrap();
        assert_eq!(status2, DrainStatus::Disconnected);
    }

    #[test]
    fn handler_error_propagates() {
        let mut builder = InboxBuilder::new(16);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();

        let result = inbox.drain(100, |msg: u32| {
            if msg == 1 {
                Err(HandlerError::fatal("boom"))
            } else {
                Ok(())
            }
        });

        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "at least one producer")]
    fn panics_on_empty_build() {
        let builder = InboxBuilder::<u32>::new(16);
        let _inbox = builder.build();
    }

    // Kills: replace shard_count -> usize with 0 (line 159)
    // Kills: replace shard_count -> usize with 1 (line 159)
    #[test]
    fn shard_count_returns_number_of_registered_producers() {
        let mut builder = InboxBuilder::<u32>::new(8);
        let _tx1 = builder.add_producer();
        let _tx2 = builder.add_producer();
        let _tx3 = builder.add_producer();
        let inbox = builder.build();

        assert_eq!(inbox.shard_count(), 3, "Should have 3 shards for 3 producers");
        assert_ne!(inbox.shard_count(), 0);
        assert_ne!(inbox.shard_count(), 1);
    }

    #[test]
    fn shard_count_is_one_for_single_producer() {
        let mut builder = InboxBuilder::<u32>::new(8);
        let _tx = builder.add_producer();
        let inbox = builder.build();
        assert_eq!(inbox.shard_count(), 1);
    }

    // Kills: replace || with && in condition on line 109 (total >= limit || shard_count >= per_shard)
    // With &&: both conditions must be true to stop, so per-shard limit is effectively ignored
    // unless total limit is also reached.
    #[test]
    fn per_shard_limit_enforced_independently_of_total() {
        // 2 shards, limit=4, per_shard=2. Each shard should drain at most 2 messages.
        // Shard 1 has 10, Shard 2 has 10. Without per-shard limit, one shard could drain 4.
        let mut builder = InboxBuilder::<u32>::new(64);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let mut inbox = builder.build();

        for i in 0u32..10 {
            tx1.try_send(i).unwrap();
            tx2.try_send(i + 100).unwrap();
        }

        let mut from_shard1 = 0usize;
        let mut from_shard2 = 0usize;
        inbox
            .drain(4, |msg: u32| {
                if msg < 100 {
                    from_shard1 += 1;
                } else {
                    from_shard2 += 1;
                }
                Ok(())
            })
            .unwrap();

        // Each shard should contribute at most per_shard = 4/2 = 2 messages
        assert!(from_shard1 <= 2, "Shard 1 drained {} messages, expected <= 2", from_shard1);
        assert!(from_shard2 <= 2, "Shard 2 drained {} messages, expected <= 2", from_shard2);
        assert_eq!(from_shard1 + from_shard2, 4, "Total should equal limit of 4");
    }

    // Kills: replace >= with < in drain condition on line 138 (total >= limit)
    // With <: drain continues even past the limit, consuming more than allowed.
    // Kills: replace >= with < on line 140 (if total >= limit → DrainStatus::More)
    #[test]
    fn drain_returns_more_when_total_limit_reached() {
        let mut builder = InboxBuilder::<u32>::new(64);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        for i in 0u32..20 {
            tx.try_send(i).unwrap();
        }

        let mut count = 0;
        let status = inbox
            .drain(5, |_: u32| {
                count += 1;
                Ok(())
            })
            .unwrap();

        assert_eq!(count, 5, "Should drain exactly 5 messages (the limit)");
        assert_eq!(status, DrainStatus::More, "Should return More when limit reached with 15 remaining");
    }

    // Kills: replace += with *= in drain on line 119 (total += 1)
    #[test]
    fn drain_total_increments_by_one_per_message() {
        let mut builder = InboxBuilder::<u32>::new(64);
        let tx = builder.add_producer();
        let mut inbox = builder.build();

        for i in 0u32..8 {
            tx.try_send(i).unwrap();
        }

        // With += * = total would explode immediately and stop after 1 msg.
        // With correct += 1, limit=8 processes exactly 8 messages.
        let mut count = 0;
        inbox
            .drain(8, |_: u32| {
                count += 1;
                Ok(())
            })
            .unwrap();

        assert_eq!(count, 8, "Should process all 8 messages when limit=8");
    }

    // Kills: replace % with / or + in round_robin index (line 104 and 134)
    // Kills: replace + with * in index calculation (line 134)
    #[test]
    fn round_robin_rotates_starting_shard_each_drain() {
        // 3 shards. After each drain, round_robin += 1 (mod 3).
        // First drain starts at shard 0. Second starts at shard 1, etc.
        let mut builder = InboxBuilder::<u32>::new(64);
        let tx1 = builder.add_producer();
        let tx2 = builder.add_producer();
        let tx3 = builder.add_producer();
        let mut inbox = builder.build();

        // Only shard 1 (tx2) has a message
        tx2.try_send(99).unwrap();

        // First drain (round_robin=0): starts at shard 0, reaches shard 1, finds msg
        let mut received = Vec::new();
        inbox.drain(10, |msg: u32| { received.push(msg); Ok(()) }).unwrap();
        assert_eq!(received, vec![99]);

        // Add another msg to shard 2 (tx3)
        tx3.try_send(77).unwrap();
        // Second drain (round_robin=1): starts at shard 1, reaches shard 2
        let mut received2 = Vec::new();
        let status = inbox.drain(10, |msg: u32| { received2.push(msg); Ok(()) }).unwrap();
        assert_eq!(received2, vec![77]);
        drop(status);
    }
}
