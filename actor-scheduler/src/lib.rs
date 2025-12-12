//! Priority Channel - A multi-priority message passing system
//!
//! This crate provides a message scheduler with three priority levels:
//! - **Control**: Highest priority, unlimited drain
//! - **Management**: Medium priority, unlimited drain
//! - **Data**: Lowest priority, burst-limited with backpressure
//!
//! # Architecture
//!
//! The scheduler uses a "doorbell" pattern where:
//! 1. The receiver blocks on the Control channel
//! 2. Data messages send a Wake signal to unblock the receiver
//! 3. Priority processing drains Control → Management → Data
//!
//! # Example
//!
//! ```rust
//! use actor_scheduler::{ActorScheduler, Message, SchedulerHandler};
//!
//! struct MyHandler;
//!
//! impl SchedulerHandler<String, String, String> for MyHandler {
//!     fn handle_data(&mut self, msg: String) {
//!         println!("Data: {}", msg);
//!     }
//!     fn handle_control(&mut self, msg: String) {
//!         println!("Control: {}", msg);
//!     }
//!     fn handle_management(&mut self, msg: String) {
//!         println!("Management: {}", msg);
//!     }
//!     fn park(&mut self) {}
//! }
//!
//! let (tx, mut rx) = ActorScheduler::<String, String, String>::new(10, 100);
//!
//! // Spawn receiver thread
//! std::thread::spawn(move || {
//!     let mut handler = MyHandler;
//!     rx.run(&mut handler);
//! });
//!
//! // Send messages from any thread
//! tx.send(Message::Data("low priority data".to_string())).unwrap();
//! tx.send(Message::Control("high priority control".to_string())).unwrap();
//! ```

mod error;

pub use error::SendError;

use std::sync::{
    Arc,
    mpsc::{self, Receiver, SyncSender, TryRecvError},
};
use std::time::{Duration, Instant};

/// Configuration for priority scheduler.
#[derive(Debug, Clone, Copy)]
pub struct PriorityConfig {
    /// Maximum data messages to process per wake cycle (burst limit)
    pub burst_limit: usize,
    /// Size of bounded data buffer (backpressure threshold)
    pub high_priority_buffer: usize,
    /// Size of control/management buffers
    pub control_buffer: usize,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            burst_limit: 1024,
            high_priority_buffer: 128,
            control_buffer: 128,
        }
    }
}

/// The types of messages supported by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message<D, C, M> {
    Data(D),
    Control(C), // No wrapper - just C directly
    Management(M),
}

/// Implement From for a Control message type.
#[macro_export]
macro_rules! impl_control_message {
    ($ty:ty) => {
        impl<D, M> From<$ty> for $crate::Message<D, $ty, M> {
            fn from(msg: $ty) -> Self {
                $crate::Message::Control(msg)
            }
        }
    };
}

/// Implement From for a Data message type.
#[macro_export]
macro_rules! impl_data_message {
    ($ty:ty) => {
        impl<C, M> From<$ty> for $crate::Message<$ty, C, M> {
            fn from(msg: $ty) -> Self {
                $crate::Message::Data(msg)
            }
        }
    };
}

/// Implement From for a Management message type.
#[macro_export]
macro_rules! impl_management_message {
    ($ty:ty) => {
        impl<D, C> From<$ty> for $crate::Message<D, C, $ty> {
            fn from(msg: $ty) -> Self {
                $crate::Message::Management(msg)
            }
        }
    };
}

/// The Actor trait - implement this to define your actor's behavior.
///
/// Actors process messages from three priority lanes:
/// - **Data** (D): High-throughput data messages
/// - **Control** (C): Time-critical control messages
/// - **Management** (M): Lifecycle and configuration messages
pub trait Actor<D, C, M> {
    /// Handle a data message.
    fn handle_data(&mut self, msg: D);

    /// Handle a control message.
    /// The scheduler will exit when all senders are dropped.
    fn handle_control(&mut self, msg: C);

    /// Handle a management message.
    fn handle_management(&mut self, msg: M);

    /// Opportunity to handle tasks not tied to a particular event
    /// called after every work cycle
    fn park(&mut self);
}

/// Legacy alias for backward compatibility
#[deprecated(since = "0.2.0", note = "Use `Actor` instead")]
pub use Actor as SchedulerHandler;

/// Create a new actor with the given configuration.
///
/// Convenience function for creating an actor scheduler and handle.
///
/// # Arguments
/// * `data_buffer_size` - Size of bounded data buffer
/// * `wake_handler` - Optional wake handler for platform event loops
pub fn create_actor<D, C, M>(
    data_buffer_size: usize,
    wake_handler: Option<Arc<dyn WakeHandler>>,
) -> (ActorHandle<D, C, M>, ActorScheduler<D, C, M>) {
    let config = PriorityConfig {
        high_priority_buffer: data_buffer_size,
        ..Default::default()
    };
    ActorScheduler::new_with_config(config, wake_handler)
}

/// Trait for waking a blocked actor scheduler.
///
/// Implement this trait for platform-specific wake mechanisms (e.g., NSEvent on macOS).
/// When messages are sent, the wake handler is called to ensure the scheduler
/// processes them immediately, even if blocked on a platform event loop.
pub trait WakeHandler: Send + Sync {
    /// Wake the scheduler from a blocked state.
    ///
    /// Called automatically when Data/Management/Control messages are sent.
    /// Platform implementations might send events to wake up event loops,
    /// while the default implementation sends a Wake message through the control channel.
    fn wake(&self);
}

/// Minimum backoff duration when no messages are available
const MIN_BACKOFF: Duration = Duration::from_micros(10);

/// Maximum backoff duration when no messages are available
const MAX_BACKOFF: Duration = Duration::from_millis(1);

/// Calculate exponential backoff with jitter.
///
/// Uses a simple exponential backoff strategy with added jitter to prevent
/// thundering herd problems when multiple actors wake simultaneously.
///
/// # Arguments
/// * `attempt` - The backoff attempt count (0 = first backoff)
///
/// # Returns
/// A duration to sleep, with exponential growth and random jitter
fn backoff_with_jitter(attempt: u32) -> Duration {
    // Exponential backoff: min(MIN * 2^attempt, MAX)
    let base_micros = MIN_BACKOFF.as_micros() as u64;
    let max_micros = MAX_BACKOFF.as_micros() as u64;

    // Calculate 2^attempt with saturation
    let multiplier = 2u64.saturating_pow(attempt);
    let backoff_micros = base_micros.saturating_mul(multiplier);
    let capped_micros = backoff_micros.min(max_micros);

    // Add jitter: random value between [0.5 * backoff, 1.0 * backoff]
    // Using Instant hash for "randomness" (good enough for backoff jitter)
    let now = Instant::now();
    let hash = (now.elapsed().as_nanos() as u64).wrapping_mul(0x9e3779b97f4a7c15); // fibonacci hash
    let jitter_pct = 50 + (hash % 50); // 50-99%
    let jittered_micros = (capped_micros * jitter_pct) / 100;

    Duration::from_micros(jittered_micros)
}

/// A unified sender handle that routes messages to the scheduler with priority lanes.
pub struct ActorHandle<D, C, M> {
    // Doorbell channel (buffer: 1) - highest priority wake signal
    tx_doorbell: SyncSender<()>,
    // All lanes are bounded for backpressure
    tx_data: SyncSender<D>,
    tx_control: SyncSender<C>, // No wrapper - just C directly
    tx_mgmt: SyncSender<M>,
    // Optional custom wake handler for platform-specific wake mechanisms
    wake_handler: Option<Arc<dyn WakeHandler>>,
}

// Manual Debug implementation - wake_handler is opaque (trait object)
impl<D, C, M> std::fmt::Debug for ActorHandle<D, C, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActorHandle")
            .field("has_wake_handler", &self.wake_handler.is_some())
            .finish_non_exhaustive()
    }
}

// Manual Clone implementation - we don't require D, C, M to be Clone
// because we're just cloning the channel senders, not the messages.
impl<D, C, M> Clone for ActorHandle<D, C, M> {
    fn clone(&self) -> Self {
        Self {
            tx_doorbell: self.tx_doorbell.clone(),
            tx_data: self.tx_data.clone(),
            tx_control: self.tx_control.clone(),
            tx_mgmt: self.tx_mgmt.clone(),
            wake_handler: self.wake_handler.clone(),
        }
    }
}

/// Send with retry and exponential backoff + jitter for fairness.
///
/// Used for control and management lanes to prevent thundering herd when
/// multiple senders compete for buffer space.
fn send_with_backoff<T>(tx: &SyncSender<T>, mut msg: T) -> Result<(), SendError> {
    use std::sync::mpsc::TrySendError;

    let mut attempt = 0;
    loop {
        match tx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(returned_msg)) => {
                // Channel full - backoff with jitter for fairness
                let backoff = backoff_with_jitter(attempt);
                std::thread::sleep(backoff);
                attempt = attempt.saturating_add(1);
                msg = returned_msg; // Restore message for retry
            }
            Err(err) => {
                // Disconnected - convert to our error type
                return Err(err.into());
            }
        }
    }
}

impl<D, C, M> ActorHandle<D, C, M> {
    /// Sends a message to the appropriate priority lane and wakes the scheduler.
    ///
    /// Accepts any type that implements `IntoMessage` for this handle's message types.
    /// Use the `impl_control_message!`, `impl_data_message!`, or `impl_management_message!`
    /// macros to mark your message types.
    ///
    /// # Blocking Behavior
    /// - `Data`: Blocking send (backpressure when buffer full)
    /// - `Control`: Retry with exponential backoff + jitter for fairness
    /// - `Management`: Retry with exponential backoff + jitter for fairness
    ///
    /// Backoff on control/management prevents thundering herd when multiple
    /// senders compete for these lanes.
    ///
    /// # Errors
    /// Returns `Err` only if the receiver has been dropped.
    pub fn send<T: Into<Message<D, C, M>>>(&self, msg: T) -> Result<(), SendError> {
        let msg = msg.into();
        self.send_message(msg)
    }

    fn send_message(&self, msg: Message<D, C, M>) -> Result<(), SendError> {
        match msg {
            Message::Data(d) => {
                // Data lane: regular blocking send
                self.tx_data.send(d)?;
                self.wake();
            }
            Message::Control(ctrl_msg) => {
                // Control lane: retry with backoff for fairness
                send_with_backoff(&self.tx_control, ctrl_msg)?;
                self.wake();
            }
            Message::Management(m) => {
                // Management lane: retry with backoff for fairness
                send_with_backoff(&self.tx_mgmt, m)?;
                self.wake();
            }
        };
        Ok(())
    }

    /// Wake the scheduler to process messages.
    ///
    /// If a custom wake handler is configured, calls it first to wake the platform
    /// event loop (e.g., sending NSEvent on macOS). Then sends a doorbell signal to
    /// unblock ActorScheduler.run().
    ///
    /// Doorbell uses try_send (drops if full) - safe because one pending wake is sufficient.
    fn wake(&self) {
        // First: wake platform event loop if custom waker present
        // This ensures the receiver can start draining channels immediately
        if let Some(waker) = &self.wake_handler {
            waker.wake();
        }

        // Then: try to send doorbell (drop if full - that's fine)
        // Only need one pending wake signal to process all messages
        let _ = self.tx_doorbell.try_send(());
    }
}

/// The receiver side that implements the priority scheduling logic.
pub struct ActorScheduler<D, C, M> {
    rx_doorbell: Receiver<()>, // Highest priority - wake signal
    rx_data: Receiver<D>,
    rx_control: Receiver<C>, // No wrapper - just C directly
    rx_mgmt: Receiver<M>,
    config: PriorityConfig,
}

impl<D, C, M> ActorScheduler<D, C, M> {
    /// Create a new scheduler channel with priority lanes.
    ///
    /// # Arguments
    /// * `data_burst_limit` - Maximum data messages to process per wake cycle
    /// * `data_buffer_size` - Size of bounded data buffer (backpressure threshold)
    ///
    /// # Returns
    /// Returns `(sender, receiver)` tuple. The sender can be cloned and shared.
    pub fn new(data_burst_limit: usize, data_buffer_size: usize) -> (ActorHandle<D, C, M>, Self) {
        let config = PriorityConfig {
            burst_limit: data_burst_limit,
            high_priority_buffer: data_buffer_size,
            ..Default::default()
        };
        Self::new_with_config(config, None)
    }

    /// Create a new scheduler channel with priority lanes and a custom wake actor.
    ///
    /// This variant allows platform-specific wake mechanisms (e.g., NSEvent on macOS)
    /// to be used in addition to the default control channel wake signal.
    ///
    /// # Arguments
    /// * `data_burst_limit` - Maximum data messages to process per wake cycle
    /// * `data_buffer_size` - Size of bounded data buffer (backpressure threshold)
    /// * `wake_handler` - Optional custom wake handler for platform event loops
    ///
    /// # Returns
    /// Returns `(sender, receiver)` tuple. The sender can be cloned and shared.
    pub fn new_with_wake_handler(
        data_burst_limit: usize,
        data_buffer_size: usize,
        wake_handler: Option<Arc<dyn WakeHandler>>,
    ) -> (ActorHandle<D, C, M>, Self) {
        let config = PriorityConfig {
            burst_limit: data_burst_limit,
            high_priority_buffer: data_buffer_size,
            ..Default::default()
        };
        Self::new_with_config(config, wake_handler)
    }

    /// Create new scheduler with detailed config
    pub fn new_with_config(
        config: PriorityConfig,
        wake_handler: Option<Arc<dyn WakeHandler>>,
    ) -> (ActorHandle<D, C, M>, Self) {
        let (tx_doorbell, rx_doorbell) = mpsc::sync_channel(1); // Buffer size 1
        let (tx_data, rx_data) = mpsc::sync_channel(config.high_priority_buffer);
        let (tx_control, rx_control) = mpsc::sync_channel(config.control_buffer);
        let (tx_mgmt, rx_mgmt) = mpsc::sync_channel(config.control_buffer);

        let sender = ActorHandle {
            tx_doorbell,
            tx_data,
            tx_control,
            tx_mgmt,
            wake_handler,
        };

        let receiver = ActorScheduler {
            rx_doorbell,
            rx_data,
            rx_control,
            rx_mgmt,
            config,
        };

        (sender, receiver)
    }

    /// The Main Scheduler Loop.
    /// Blocks on the Doorbell channel. Prioritizes Control > Management > Data.
    ///
    /// # Arguments
    /// * `actor` - Implementation of `Actor` trait
    ///
    /// This method runs forever until all senders are dropped.
    pub fn run<A>(&mut self, actor: &mut A)
    where
        A: Actor<D, C, M>,
    {
        loop {
            // 1. Block on Doorbell (Highest Priority)
            match self.rx_doorbell.recv() {
                Ok(()) => {}
                Err(_) => return, // All senders disconnected
            }

            // 2. Priority Processing Loop
            let mut keep_working = true;

            while keep_working {
                keep_working = false;

                // A. Control (Highest Priority - Unlimited Drain)
                while let Ok(ctrl_msg) = self.rx_control.try_recv() {
                    actor.handle_control(ctrl_msg);
                    keep_working = true;
                }

                // B. Management (Medium Priority - Unlimited Drain)
                while let Ok(msg) = self.rx_mgmt.try_recv() {
                    actor.handle_management(msg);
                    keep_working = true;
                }

                // C. Data (Low Priority - Burst Limited)
                let mut data_count = 0;
                while data_count < self.config.burst_limit {
                    match self.rx_data.try_recv() {
                        Ok(msg) => {
                            actor.handle_data(msg);
                            data_count += 1;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }

                if data_count >= self.config.burst_limit {
                    keep_working = true;
                }
            }
            actor.park()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    struct TestHandler {
        log: Arc<Mutex<Vec<String>>>,
    }

    impl SchedulerHandler<String, String, String> for TestHandler {
        fn handle_data(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Data: {}", msg));
        }
        fn handle_control(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Ctrl: {}", msg));
        }
        fn handle_management(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Mgmt: {}", msg));
        }
        fn park(&mut self) {}
    }

    #[test]
    fn test_priority_ordering() {
        let (tx, mut rx) = ActorScheduler::new(2, 10);
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        let handle = thread::spawn(move || {
            let mut handler = TestHandler { log: log_clone };
            rx.run(&mut handler);
        });

        // Send messages in mixed order
        tx.send(Message::Data("1".to_string())).unwrap();
        tx.send(Message::Management("M".to_string())).unwrap();
        tx.send(Message::Data("2".to_string())).unwrap();
        tx.send(Message::Control("C".to_string())).unwrap();
        tx.send(Message::Data("3".to_string())).unwrap();

        thread::sleep(Duration::from_millis(50));

        // Drop sender to close channels and stop run()
        drop(tx);
        handle.join().unwrap();

        let messages = log.lock().unwrap();
        assert!(messages.len() > 0, "Should have processed messages");

        // Control should be processed before lower priority messages
        let ctrl_idx = messages.iter().position(|s| s.contains("Ctrl")).unwrap();
        let data1_idx = messages.iter().position(|s| s.contains("Data: 1")).unwrap();

        assert!(
            ctrl_idx < data1_idx,
            "Control should be processed before Data that was sent earlier"
        );
    }

    #[test]
    fn test_backpressure() {
        let (tx, mut rx) = ActorScheduler::new(2, 1); // Buffer size 1, burst limit 2
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        thread::spawn(move || {
            let mut handler = TestHandler { log: log_clone };
            rx.run(&mut handler);
        });

        let tx_clone = tx.clone();
        let send_thread = thread::spawn(move || {
            tx_clone.send(Message::Data("1".to_string())).unwrap();
            tx_clone.send(Message::Data("2".to_string())).unwrap(); // Should block
            tx_clone.send(Message::Data("3".to_string())).unwrap();
        });

        thread::sleep(Duration::from_millis(100));
        drop(tx);
        send_thread.join().unwrap();

        thread::sleep(Duration::from_millis(50));
        let messages = log.lock().unwrap();
        assert_eq!(messages.len(), 3, "All messages should be processed");
    }

    #[test]
    fn test_trait_handler() {
        struct CountingHandler {
            data_count: usize,
            ctrl_count: usize,
            mgmt_count: usize,
        }

        impl SchedulerHandler<i32, String, bool> for CountingHandler {
            fn handle_data(&mut self, _: i32) {
                self.data_count += 1;
            }
            fn handle_control(&mut self, _: String) {
                self.ctrl_count += 1;
            }
            fn handle_management(&mut self, _: bool) {
                self.mgmt_count += 1;
            }
            fn park(&mut self) {}
        }

        let (tx, mut rx) = ActorScheduler::new(10, 100);

        let handle = thread::spawn(move || {
            let mut handler = CountingHandler {
                data_count: 0,
                ctrl_count: 0,
                mgmt_count: 0,
            };
            rx.run(&mut handler);
            handler
        });

        tx.send(Message::Data(1)).unwrap();
        tx.send(Message::Data(2)).unwrap();
        tx.send(Message::Control("test".to_string())).unwrap();
        tx.send(Message::Management(true)).unwrap();

        thread::sleep(Duration::from_millis(50));
        drop(tx);

        let handler = handle.join().unwrap();
        assert_eq!(handler.data_count, 2);
        assert_eq!(handler.ctrl_count, 1);
        assert_eq!(handler.mgmt_count, 1);
    }
}
