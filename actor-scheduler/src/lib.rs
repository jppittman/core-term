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
//! use actor_scheduler::{spawn, Actor, Message};
//!
//! struct MyHandler;
//!
//! impl Actor<String, String, String> for MyHandler {
//!     fn handle_data(&mut self, msg: String) {
//!         println!("Data: {}", msg);
//!     }
//!     fn handle_control(&mut self, msg: String) {
//!         println!("Control: {}", msg);
//!     }
//!     fn handle_management(&mut self, msg: String) {
//!         println!("Management: {}", msg);
//!     }
//! }
//!
//! // Spawn actor in dedicated thread
//! let handle = spawn(MyHandler);
//!
//! // Send messages from any thread
//! handle.send(Message::Data("low priority data".to_string())).unwrap();
//! handle.send(Message::Control("high priority control".to_string())).unwrap();
//! ```

mod error;

pub use error::SendError;

// Re-export derive macro when feature is enabled (currently disabled)
// #[cfg(feature = "derive")]
// pub use actor_scheduler_derive::ActorManifest;

use std::sync::{
    mpsc::{self, Receiver, SyncSender, TryRecvError},
    Arc, Weak,
};
use std::time::{Duration, Instant};

/// The types of messages supported by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message<D, C, M> {
    Data(D),
    Control(C),
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

/// Maximum capacity for Control and Management lanes
const CONTROL_MGMT_BUFFER_SIZE: usize = 128;

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
    // Wrapped in Arc to allow Weak references for the "Metronome" pattern
    tx_doorbell: Arc<SyncSender<()>>,
    // All lanes are bounded for backpressure
    tx_data: Arc<SyncSender<D>>,
    tx_control: Arc<SyncSender<C>>,
    tx_mgmt: Arc<SyncSender<M>>,
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

impl<D, C, M> ActorHandle<D, C, M> {
    /// Create a weak reference to this handle.
    ///
    /// Weak handles don't prevent the actor from shutting down.
    /// Call `upgrade()` to convert back to strong reference.
    ///
    /// This is critical for avoiding reference cycles. For example:
    /// - EnginePlatform holds strong ActorHandle → Engine
    /// - Engine owns Driver (cloned)
    /// - Driver holds **weak** ActorHandle → Engine (no cycle!)
    pub fn downgrade(&self) -> WeakActorHandle<D, C, M> {
        WeakActorHandle {
            tx_doorbell: Arc::downgrade(&self.tx_doorbell),
            tx_data: Arc::downgrade(&self.tx_data),
            tx_control: Arc::downgrade(&self.tx_control),
            tx_mgmt: Arc::downgrade(&self.tx_mgmt),
            wake_handler: self.wake_handler.clone(),
        }
    }
}

/// A weak reference to an ActorHandle that doesn't prevent actor shutdown.
///
/// Useful for actors that need to send messages but shouldn't keep other actors alive.
/// Example: VSync timer holds weak ref to engine - when engine shuts down, VSync detects it.
pub struct WeakActorHandle<D, C, M> {
    tx_doorbell: Weak<SyncSender<()>>,
    tx_data: Weak<SyncSender<D>>,
    tx_control: Weak<SyncSender<C>>,
    tx_mgmt: Weak<SyncSender<M>>,
    wake_handler: Option<Arc<dyn WakeHandler>>,
}

// Manual Clone implementation for WeakActorHandle
impl<D, C, M> Clone for WeakActorHandle<D, C, M> {
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

// Manual Debug implementation for WeakActorHandle
impl<D, C, M> std::fmt::Debug for WeakActorHandle<D, C, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakActorHandle")
            .field("has_wake_handler", &self.wake_handler.is_some())
            .finish_non_exhaustive()
    }
}

impl<D, C, M> WeakActorHandle<D, C, M> {
    /// Attempt to upgrade to a strong ActorHandle.
    ///
    /// Returns `None` if all strong references have been dropped (actor shut down).
    pub fn upgrade(&self) -> Option<ActorHandle<D, C, M>> {
        Some(ActorHandle {
            tx_doorbell: self.tx_doorbell.upgrade()?,
            tx_data: self.tx_data.upgrade()?,
            tx_control: self.tx_control.upgrade()?,
            tx_mgmt: self.tx_mgmt.upgrade()?,
            wake_handler: self.wake_handler.clone(),
        })
    }

    /// Send a message to the actor (upgrades weak handle first).
    ///
    /// Returns an error if the actor has shut down (upgrade fails) or if the send fails.
    /// This is the recommended way to send from a weak handle - it handles the upgrade/send dance.
    pub fn send(&self, msg: Message<D, C, M>) -> Result<(), SendError> {
        let handle = self.upgrade().ok_or(SendError)?;
        handle.send(msg)
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

/// Spawn an actor in a dedicated thread with priority message lanes.
///
/// This is the recommended way to create actors. The actor is owned by the scheduler,
/// which runs in a dedicated thread. Messages are processed with three priority levels:
/// Control (highest) > Management (medium) > Data (lowest, burst-limited).
///
/// # Arguments
/// * `actor` - The actor to spawn (consumed)
///
/// # Returns
/// An `ActorHandle` that can be cloned and used to send messages to the actor.
///
/// # Example
/// ```
/// use actor_scheduler::{spawn, Actor, Message};
///
/// struct MyActor;
/// impl Actor<String, String, String> for MyActor {
///     fn handle_data(&mut self, msg: String) { println!("Data: {}", msg); }
///     fn handle_control(&mut self, msg: String) { println!("Ctrl: {}", msg); }
///     fn handle_management(&mut self, msg: String) { println!("Mgmt: {}", msg); }
/// }
///
/// let handle = spawn(MyActor);
/// handle.send(Message::Data("hello".to_string())).unwrap();
/// ```
pub fn spawn<A, D, C, M>(actor: A) -> ActorHandle<D, C, M>
where
    A: Actor<D, C, M> + Send + 'static,
    D: Send + 'static,
    C: Send + 'static,
    M: Send + 'static,
{
    spawn_with_config(actor, 1024, 1024, None)
}

/// Spawn an actor with custom configuration.
///
/// See `spawn()` for details. This variant allows customizing buffer sizes and wake handler.
///
/// # Arguments
/// * `actor` - The actor to spawn (consumed)
/// * `data_burst_limit` - Maximum data messages per wake cycle (default: 1024)
/// * `data_buffer_size` - Data channel buffer size (default: 1024)
/// * `wake_handler` - Optional platform-specific wake handler
///
/// # Returns
/// An `ActorHandle` that can be cloned and used to send messages to the actor.
pub fn spawn_with_config<A, D, C, M>(
    actor: A,
    data_burst_limit: usize,
    data_buffer_size: usize,
    wake_handler: Option<Arc<dyn WakeHandler>>,
) -> ActorHandle<D, C, M>
where
    A: Actor<D, C, M> + Send + 'static,
    D: Send + 'static,
    C: Send + 'static,
    M: Send + 'static,
{
    use std::thread;

    // A. Create channels
    let (tx_doorbell, rx_doorbell) = mpsc::sync_channel(1);
    let (tx_data, rx_data) = mpsc::sync_channel(data_buffer_size);
    let (tx_control, rx_control) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);
    let (tx_mgmt, rx_mgmt) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);

    // B. Create handle (senders wrapped in Arc)
    let handle = ActorHandle {
        tx_doorbell: Arc::new(tx_doorbell),
        tx_data: Arc::new(tx_data),
        tx_control: Arc::new(tx_control),
        tx_mgmt: Arc::new(tx_mgmt),
        wake_handler,
    };

    // C. Create scheduler (owns actor)
    let scheduler = ActorScheduler {
        rx_doorbell,
        rx_data,
        rx_control,
        rx_mgmt,
        actor,
        data_burst_limit,
    };

    // D. Spawn thread
    thread::spawn(move || {
        scheduler.run();
    });

    handle
}

/// The receiver side that implements the priority scheduling logic.
///
/// Owns the actor and processes messages on its behalf.
pub struct ActorScheduler<D, C, M, A>
where
    A: Actor<D, C, M>,
{
    rx_doorbell: Receiver<()>,  // Highest priority - wake signal
    rx_data: Receiver<D>,
    rx_control: Receiver<C>,    // No wrapper - just C directly
    rx_mgmt: Receiver<M>,
    actor: A,  // OWNED ACTOR
    data_burst_limit: usize,
}

impl<D, C, M, A> ActorScheduler<D, C, M, A>
where
    A: Actor<D, C, M>,
{
    /// The Main Scheduler Loop.
    /// Blocks on the Doorbell channel. Prioritizes Control > Management > Data.
    ///
    /// This method runs forever until all senders are dropped.
    pub fn run(mut self) {
        loop {
            // 1. Block on Doorbell (Highest Priority)
            match self.rx_doorbell.recv() {
                Ok(()) => {},
                Err(_) => return, // All senders disconnected
            }

            // 2. Priority Processing Loop
            let mut keep_working = true;

            while keep_working {
                keep_working = false;

                // A. Control (Highest Priority - Unlimited Drain)
                while let Ok(ctrl_msg) = self.rx_control.try_recv() {
                    self.actor.handle_control(ctrl_msg);
                    keep_working = true;
                }

                // B. Management (Medium Priority - Unlimited Drain)
                while let Ok(msg) = self.rx_mgmt.try_recv() {
                    self.actor.handle_management(msg);
                    keep_working = true;
                }

                // C. Data (Low Priority - Burst Limited)
                let mut data_count = 0;
                while data_count < self.data_burst_limit {
                    match self.rx_data.try_recv() {
                        Ok(msg) => {
                            self.actor.handle_data(msg);
                            data_count += 1;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }

                if data_count >= self.data_burst_limit {
                    keep_working = true;
                }
            }
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

    impl Actor<String, String, String> for TestHandler {
        fn handle_data(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Data: {}", msg));
        }
        fn handle_control(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Ctrl: {}", msg));
        }
        fn handle_management(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Mgmt: {}", msg));
        }
    }

    #[test]
    fn test_priority_ordering() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        let handler = TestHandler { log: log_clone };
        let tx = spawn_with_config(handler, 2, 10, None);

        // Send messages in mixed order
        tx.send(Message::Data("1".to_string())).unwrap();
        tx.send(Message::Management("M".to_string())).unwrap();
        tx.send(Message::Data("2".to_string())).unwrap();
        tx.send(Message::Control("C".to_string())).unwrap();
        tx.send(Message::Data("3".to_string())).unwrap();

        thread::sleep(Duration::from_millis(50));

        // Drop sender to close channels and stop run()
        drop(tx);

        thread::sleep(Duration::from_millis(50));

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
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        let handler = TestHandler { log: log_clone };
        let tx = spawn_with_config(handler, 2, 1, None); // Buffer size 1, burst limit 2

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
            data_count: Arc<Mutex<usize>>,
            ctrl_count: Arc<Mutex<usize>>,
            mgmt_count: Arc<Mutex<usize>>,
        }

        impl Actor<i32, String, bool> for CountingHandler {
            fn handle_data(&mut self, _: i32) {
                *self.data_count.lock().unwrap() += 1;
            }
            fn handle_control(&mut self, _: String) {
                *self.ctrl_count.lock().unwrap() += 1;
            }
            fn handle_management(&mut self, _: bool) {
                *self.mgmt_count.lock().unwrap() += 1;
            }
        }

        let data_count = Arc::new(Mutex::new(0));
        let ctrl_count = Arc::new(Mutex::new(0));
        let mgmt_count = Arc::new(Mutex::new(0));

        let handler = CountingHandler {
            data_count: data_count.clone(),
            ctrl_count: ctrl_count.clone(),
            mgmt_count: mgmt_count.clone(),
        };

        let tx = spawn_with_config(handler, 10, 100, None);

        tx.send(Message::Data(1)).unwrap();
        tx.send(Message::Data(2)).unwrap();
        tx.send(Message::Control("test".to_string())).unwrap();
        tx.send(Message::Management(true)).unwrap();

        thread::sleep(Duration::from_millis(50));
        drop(tx);

        thread::sleep(Duration::from_millis(50));

        assert_eq!(*data_count.lock().unwrap(), 2);
        assert_eq!(*ctrl_count.lock().unwrap(), 1);
        assert_eq!(*mgmt_count.lock().unwrap(), 1);
    }
}
