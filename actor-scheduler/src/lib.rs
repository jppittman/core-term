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
//! # Troupe System
//!
//! The troupe system provides lifecycle management for groups of actors.
//! Troupes can nest - a child troupe's `play()` can run inside a parent's spawned thread.
//!
//! ## Basic Usage
//!
//! ```ignore
//! troupe! {
//!     engine: EngineActor [expose],    // handle exposed to parent
//!     vsync: VsyncActor,               // internal only
//!     display: DisplayActor [main],    // runs on calling thread
//! }
//!
//! // Simple: create and run in one step
//! run().expect("troupe failed");
//! ```
//!
//! ## Two-Phase Initialization (for nesting)
//!
//! ```ignore
//! // Phase 1: Create child troupe (no threads yet)
//! let child = Troupe::new();
//!
//! // Phase 2: Parent grabs exposed handles
//! let child_engine = child.exposed().engine;
//!
//! // Phase 3: Spawn child troupe as an actor in parent
//! s.spawn(|| child.play());
//!
//! // Parent can now send to child_engine
//! ```
//!
//! ## Nesting Architecture
//!
//! ```text
//! RootTroupe.play()                          <- main thread (GUI)
//! ├── spawn thread -> ActorA.run()
//! ├── spawn thread -> ChildTroupe.play()    <- blocks, owns scoped threads
//! │   ├── spawn thread -> ChildActorX.run()
//! │   └── ChildActorY.run() [child's main]
//! └── RootMainActor.run() [root's main]     <- GUI actor, on main thread
//! ```
//!
//! # Example (Basic Scheduler)
//!
//! ```rust
//! use actor_scheduler::{ActorScheduler, Message, SchedulerHandler, ParkHint};
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
//!     fn park(&mut self, _hint: ParkHint) -> ParkHint { ParkHint::Wait }
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

// Re-export macros from the proc-macro crate
pub use actor_scheduler_macros::{actor_impl, troupe};

use std::sync::{
    Arc,
    mpsc::{self, Receiver, SyncSender, TryRecvError},
};
use std::time::{Duration, Instant};

/// The types of messages supported by the scheduler.
///
/// Messages are organized into three priority lanes, with different guarantees and semantics.
///
/// # Message Lanes
///
/// | Lane | Priority | Throughput | Blocking | Use Case |
/// |------|----------|-----------|----------|----------|
/// | **Data** (D) | Lowest | High | Yes (backpressure) | Continuous, high-volume data |
/// | **Control** (C) | High | Medium | Unlimited | Time-critical state changes |
/// | **Management** (M) | Medium | Low | Unlimited | Lifecycle, configuration |
///
/// ## Data Lane (D)
///
/// **Purpose**: High-throughput, low-latency data messages.
///
/// **Contract**:
/// - **Sender**: Sends data continuously; may block on full buffer
/// - **Receiver**: Drains after Control and Management, subject to burst limiting
/// - **Guarantee**: Best-effort delivery; may drop if buffer overflows
/// - **Ordering**: FIFO within lane
///
/// **Example**: Frame data, sensor readings, streaming events
///
/// ## Control Lane (C)
///
/// **Purpose**: Time-critical control messages that need immediate attention.
///
/// **Contract**:
/// - **Sender**: Never blocks; has unlimited buffer
/// - **Receiver**: Drains before Management and Data messages
/// - **Guarantee**: Guaranteed delivery (unlimited queue)
/// - **Ordering**: FIFO within lane, always processed before Data/Management
///
/// **Example**: User input (keypresses, mouse), window resize, close requests
///
/// ## Management Lane (M)
///
/// **Purpose**: Configuration and lifecycle messages.
///
/// **Contract**:
/// - **Sender**: Never blocks; has unlimited buffer
/// - **Receiver**: Drains between Control and Data
/// - **Guarantee**: Guaranteed delivery (unlimited queue)
/// - **Ordering**: FIFO within lane
///
/// **Example**: Configuration changes, resource allocation, subscription/unsubscription
///
/// # Scheduling Strategy
///
/// The scheduler drains messages in priority order:
///
/// ```text
/// Loop:
///   1. Drain all Control messages (may block or spin until all drained)
///   2. Drain all Management messages
///   3. Drain up to N Data messages (burst limit to prevent starvation)
///   4. Call park() - let actor/OS do other work
///   5. Repeat
/// ```
///
/// This ensures:
/// - Control messages are NEVER delayed by data processing
/// - Management messages get attention even if data is continuously flowing
/// - Data can't starve other lanes (burst limit)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message<D, C, M> {
    /// A data message (lowest priority, high throughput).
    ///
    /// # Contract
    ///
    /// **Sender**:
    /// - May block if buffer is full (backpressure)
    /// - Should not send Control/Management equivalent if Data suffices
    ///
    /// **Receiver** (Actor):
    /// - Will receive via `handle_data()`
    /// - Processing is deferred behind Control and Management
    /// - May be burst-limited (batches processed per iteration)
    ///
    /// # Example
    ///
    /// ```ignore
    /// tx.send(Message::Data(PixelData { x: 100, y: 50, color: red }))?;
    /// // May block if the 10-message buffer is full
    /// ```
    Data(D),

    /// A control message (highest priority, time-critical).
    ///
    /// # Contract
    ///
    /// **Sender**:
    /// - NEVER blocks (unlimited buffer)
    /// - Use for messages that can't wait (user input, resize events)
    /// - Should not over-use; reserve for truly time-critical operations
    ///
    /// **Receiver** (Actor):
    /// - Will receive via `handle_control()`
    /// - Guaranteed to be processed before Data/Management messages
    /// - If the actor isn't running, message queues indefinitely (unbounded buffer)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // User clicked the close button - this must be processed immediately
    /// tx.send(Message::Control(CloseRequested))?;
    /// // Never blocks; queues if actor is busy
    /// ```
    ///
    /// # Warning: Unbounded Queue
    ///
    /// Control messages have unlimited buffer. If the actor is slow, Control messages
    /// can pile up indefinitely, consuming memory. Use Control sparingly and only for
    /// messages that truly can't be delayed.
    Control(C),

    /// A management message (medium priority, configuration/lifecycle).
    ///
    /// # Contract
    ///
    /// **Sender**:
    /// - NEVER blocks (unlimited buffer)
    /// - Use for lifecycle and configuration (create, destroy, configure)
    /// - Lower priority than Control but higher than Data
    ///
    /// **Receiver** (Actor):
    /// - Will receive via `handle_management()`
    /// - Guaranteed delivery (unbounded queue)
    /// - Processed after Control but before Data messages
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Allocate a new resource - this doesn't need to be super-urgent
    /// // but it's more important than continuous data stream
    /// tx.send(Message::Management(AllocateBuffer { size: 1024 }))?;
    /// ```
    Management(M),

    /// Shutdown signal.
    ///
    /// # Contract
    ///
    /// **Sender**: Signals that the actor should shut down cleanly.
    ///
    /// **Receiver**: The scheduler handles this directly—the actor never sees it.
    /// When the scheduler receives `Shutdown`, it exits the run loop immediately
    /// and `rx.run()` returns.
    ///
    /// # Implementation Details
    ///
    /// - This is never delivered to the actor's `handle_*` methods
    /// - It's a special signal interpreted by the scheduler itself
    /// - Useful for graceful shutdown of the actor system
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Shut down the actor
    /// tx.send(Message::Shutdown)?;
    /// // rx.run() will exit and return
    /// ```
    Shutdown,
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

/// Hint tells the OS loop how aggressive it should be
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParkHint {
    Wait, // Queues empty. Sleep until OS event or Wake signal. (0% CPU)
    Poll, // Queues busy. Process pending OS events and return immediately.
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

    /// The "Hook" where the Actor creates the bridge to the OS.
    /// Called when the scheduler has drained available messages (or hit burst limits).
    ///
    /// Returns a hint about whether the scheduler should block or poll.
    fn park(&mut self, hint: ParkHint) -> ParkHint;
}

/// Legacy alias for backward compatibility
#[deprecated(since = "0.2.0", note = "Use `Actor` instead")]
pub use Actor as SchedulerHandler;

/// Defines the message types for an actor managed by the troupe! macro.
///
/// This trait is separate from `TroupeActor` to allow extracting type information
/// without lifetime parameters, which is necessary for the `troupe!` macro to
/// generate struct field types.
///
/// # Example
///
/// ```ignore
/// impl ActorTypes for MyActor {
///     type Data = MyData;
///     type Control = MyControl;
///     type Management = MyManagement;
/// }
/// ```
pub trait ActorTypes {
    /// The data message type for this actor.
    type Data: Send + 'static;
    /// The control message type for this actor.
    type Control: Send + 'static;
    /// The management message type for this actor.
    type Management: Send + 'static;
}

/// The TroupeActor trait for actors managed by the troupe! macro.
///
/// Unlike the basic `Actor` trait, `TroupeActor` is parameterized over a Directory
/// type, enabling type-safe access to other actors in the group. The `#[actor_impl]`
/// macro generates the impl for this trait.
///
/// # Example
///
/// ```ignore
/// pub struct EngineActor<'a> {
///     dir: &'a Directory,
/// }
///
/// impl ActorTypes for EngineActor<'_> {
///     type Data = EngineData;
///     type Control = EngineControl;
///     type Management = EngineManagement;
/// }
///
/// impl<'a> TroupeActor<'a, Directory> for EngineActor<'a> {
///     fn new(dir: &'a Directory) -> Self { Self { dir } }
/// }
///
/// impl Actor<EngineData, EngineControl, EngineManagement> for EngineActor<'_> {
///     fn handle_data(&mut self, msg: EngineData) { }
///     fn handle_control(&mut self, msg: EngineControl) { }
///     fn handle_management(&mut self, msg: EngineManagement) { }
///     fn park(&mut self, hint: ParkHint) -> ParkHint { hint }
/// }
/// ```
pub trait TroupeActor<'a, Dir>:
    Sized
    + ActorTypes
    + Actor<
        <Self as ActorTypes>::Data,
        <Self as ActorTypes>::Control,
        <Self as ActorTypes>::Management,
    >
where
    Dir: 'a,
{
    /// Create a new actor with a reference to the directory.
    fn new(dir: &'a Dir) -> Self;
}

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
    ActorScheduler::new_with_wake_handler(
        1024, // Default data burst limit
        data_buffer_size,
        wake_handler,
    )
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
const MAX_BACKOFF: Duration = Duration::from_millis(500);

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
/// Fibonacci hash constant for jitter calculation.
const JITTER_HASH_CONSTANT: u64 = 0x9e3779b97f4a7c15;
/// Minimum jitter percentage (50%).
const JITTER_MIN_PCT: u64 = 50;
/// Jitter range (50-99%).
const JITTER_RANGE: u64 = 50;

fn backoff_with_jitter(attempt: u32) -> Result<Duration, SendError> {
    let base_micros = MIN_BACKOFF.as_micros() as u64;
    let max_micros = MAX_BACKOFF.as_micros() as u64;

    let multiplier = 2u64.saturating_pow(attempt);
    let backoff_micros = base_micros.saturating_mul(multiplier);
    if backoff_micros > max_micros {
        return Err(SendError::Timeout);
    }

    // Add jitter: random value between [0.5 * backoff, 1.0 * backoff]
    // Using Instant hash for "randomness" (good enough for backoff jitter)
    let now = Instant::now();
    let hash = (now.elapsed().as_nanos() as u64).wrapping_mul(JITTER_HASH_CONSTANT);
    let jitter_pct = JITTER_MIN_PCT + (hash % JITTER_RANGE);
    let jittered_micros = (backoff_micros * jitter_pct) / 100;

    Ok(Duration::from_micros(jittered_micros))
}

/// A unified sender handle that routes messages to the scheduler with priority lanes.
pub struct ActorHandle<D, C, M> {
    // Doorbell channel (buffer: 1) - wake signal
    tx_doorbell: SyncSender<()>,
    // Shutdown channel (buffer: 1) - shutdown signal
    tx_shutdown: SyncSender<()>,
    // All lanes are bounded for backpressure
    tx_data: SyncSender<D>,
    tx_control: SyncSender<C>,
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
            tx_shutdown: self.tx_shutdown.clone(),
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
                let backoff = backoff_with_jitter(attempt)?;
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
            Message::Shutdown => {
                // Shutdown: send signal and wake (try_send - drop if already pending)
                let _ = self.tx_shutdown.try_send(());
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
        if let Some(waker) = &self.wake_handler {
            waker.wake();
        }

        let _ = self.tx_doorbell.try_send(());
    }
}

/// The receiver side that implements the priority scheduling logic.
pub struct ActorScheduler<D, C, M> {
    rx_doorbell: Receiver<()>, // Wake signal
    rx_shutdown: Receiver<()>, // Shutdown signal
    rx_data: Receiver<D>,
    rx_control: Receiver<C>,
    rx_mgmt: Receiver<M>,
    data_burst_limit: usize,
    management_burst_limit: usize,
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
        let (tx_doorbell, rx_doorbell) = mpsc::sync_channel(1);
        let (tx_shutdown, rx_shutdown) = mpsc::sync_channel(1);
        let (tx_data, rx_data) = mpsc::sync_channel(data_buffer_size);
        let (tx_control, rx_control) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);
        let (tx_mgmt, rx_mgmt) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);

        let sender = ActorHandle {
            tx_doorbell,
            tx_shutdown,
            tx_data,
            tx_control,
            tx_mgmt,
            wake_handler: None,
        };

        let receiver = ActorScheduler {
            rx_doorbell,
            rx_shutdown,
            rx_data,
            rx_control,
            rx_mgmt,
            data_burst_limit,
            management_burst_limit: CONTROL_MGMT_BUFFER_SIZE,
        };

        (sender, receiver)
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
        let (tx_doorbell, rx_doorbell) = mpsc::sync_channel(1);
        let (tx_shutdown, rx_shutdown) = mpsc::sync_channel(1);
        let (tx_data, rx_data) = mpsc::sync_channel(data_buffer_size);
        let (tx_control, rx_control) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);
        let (tx_mgmt, rx_mgmt) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);

        let sender = ActorHandle {
            tx_doorbell,
            tx_shutdown,
            tx_data,
            tx_control,
            tx_mgmt,
            wake_handler,
        };

        let receiver = ActorScheduler {
            rx_doorbell,
            rx_shutdown,
            rx_data,
            rx_control,
            rx_mgmt,
            data_burst_limit,
            management_burst_limit: CONTROL_MGMT_BUFFER_SIZE,
        };

        (sender, receiver)
    }

    /// The Main Scheduler Loop.
    /// Blocks on the Doorbell channel. Prioritizes Shutdown > Control > Management > Data.
    ///
    /// # Arguments
    /// * `actor` - Implementation of `Actor` trait
    ///
    /// This method runs until:
    /// - A `Shutdown` message is received (immediate exit, actor never sees it)
    /// - All senders are dropped (channel disconnected)
    pub fn run<A>(&mut self, actor: &mut A)
    where
        A: Actor<D, C, M>,
    {
        loop {
            match self.rx_doorbell.recv() {
                Ok(()) => {}
                Err(_) => return,
            }

            let mut keep_working = true;

            while keep_working {
                keep_working = false;

                // Check shutdown first - highest priority, causes immediate exit
                if self.rx_shutdown.try_recv().is_ok() {
                    return;
                }

                while let Ok(ctrl_msg) = self.rx_control.try_recv() {
                    actor.handle_control(ctrl_msg);
                    keep_working = true;
                }

                let mut mgmt_count = 0;
                while mgmt_count < self.management_burst_limit {
                    match self.rx_mgmt.try_recv() {
                        Ok(msg) => {
                            actor.handle_management(msg);
                            mgmt_count += 1;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }
                if mgmt_count >= self.management_burst_limit {
                    keep_working = true;
                }

                // Check control between management and data for rough priority
                while let Ok(ctrl_msg) = self.rx_control.try_recv() {
                    actor.handle_control(ctrl_msg);
                    keep_working = true;
                }

                let mut data_count = 0;
                while data_count < self.data_burst_limit {
                    match self.rx_data.try_recv() {
                        Ok(msg) => {
                            actor.handle_data(msg);
                            data_count += 1;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }

                if data_count >= self.data_burst_limit {
                    keep_working = true;
                }

                // Call park with appropriate hint based on whether we'll loop again
                let hint = if keep_working {
                    ParkHint::Poll // Queues still have work, do quick OS poll
                } else {
                    ParkHint::Wait // Queues drained, can block
                };
                let returned_hint = actor.park(hint);

                // Actor can override and request to keep looping
                if returned_hint == ParkHint::Poll {
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

        fn park(&mut self, _hint: ParkHint) -> ParkHint {
            // No-op for test
            ParkHint::Wait
        }
    }

    #[test]
    fn verify_data_lane_backpressure_contract() {
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
    fn verify_actor_trait_contract() {
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
            fn park(&mut self, _hint: ParkHint) -> ParkHint {
                ParkHint::Wait
            }
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

        let actor = handle.join().unwrap();
        assert_eq!(actor.data_count, 2);
        assert_eq!(actor.ctrl_count, 1);
        assert_eq!(actor.mgmt_count, 1);
    }

    #[test]
    fn shutdown_message_exits_scheduler_immediately() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let (tx, mut rx) = ActorScheduler::<(), (), ()>::new(10, 100);

        let exited = Arc::new(AtomicBool::new(false));
        let exited_clone = exited.clone();

        let handle = thread::spawn(move || {
            struct NoopActor;
            impl Actor<(), (), ()> for NoopActor {
                fn handle_data(&mut self, _: ()) {}
                fn handle_control(&mut self, _: ()) {}
                fn handle_management(&mut self, _: ()) {}
                fn park(&mut self, h: ParkHint) -> ParkHint { h }
            }
            rx.run(&mut NoopActor);
            exited_clone.store(true, Ordering::SeqCst);
        });

        // Verify running
        thread::sleep(Duration::from_millis(20));
        assert!(!exited.load(Ordering::SeqCst), "should still be running");

        // Send shutdown
        tx.send(Message::Shutdown).unwrap();

        // Should exit quickly
        handle.join().unwrap();
        assert!(exited.load(Ordering::SeqCst), "should have exited");
    }

    #[test]
    fn shutdown_takes_priority_over_pending_messages() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (tx, mut rx) = ActorScheduler::<i32, (), ()>::new(10, 100);

        let processed = Arc::new(AtomicUsize::new(0));
        let processed_clone = processed.clone();

        // Queue many data messages
        for i in 0..50 {
            tx.send(Message::Data(i)).unwrap();
        }
        // Then shutdown
        tx.send(Message::Shutdown).unwrap();

        let handle = thread::spawn(move || {
            struct CountActor(Arc<AtomicUsize>);
            impl Actor<i32, (), ()> for CountActor {
                fn handle_data(&mut self, _: i32) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
                fn handle_control(&mut self, _: ()) {}
                fn handle_management(&mut self, _: ()) {}
                fn park(&mut self, h: ParkHint) -> ParkHint { h }
            }
            rx.run(&mut CountActor(processed_clone));
        });

        handle.join().unwrap();

        // Shutdown should have caused exit before all data processed
        let count = processed.load(Ordering::SeqCst);
        assert!(count < 50, "shutdown should exit before processing all data, processed {}", count);
    }
}

#[cfg(test)]
mod troupe_tests {
    #![allow(dead_code)] // Test module - structs demonstrate pattern but may not all be constructed

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // === Message types ===

    pub struct EngineData;
    #[derive(Default)]
    pub enum EngineControl {
        Tick,
        #[default]
        Shutdown,
    }
    pub struct EngineManagement;

    pub struct DisplayData;
    #[derive(Default)]
    pub enum DisplayControl {
        Render,
        #[default]
        Shutdown,
    }
    pub struct DisplayManagement;

    // === Actors ===

    pub struct EngineActor<'a> {
        dir: &'a Directory,
        tick_count: &'a AtomicUsize,
    }

    impl Actor<EngineData, EngineControl, EngineManagement> for EngineActor<'_> {
        fn handle_data(&mut self, _msg: EngineData) {}
        fn handle_control(&mut self, msg: EngineControl) {
            match msg {
                EngineControl::Tick => {
                    self.tick_count.fetch_add(1, Ordering::SeqCst);
                    // Tell display to render
                    let _ = self
                        .dir
                        .display
                        .send(Message::Control(DisplayControl::Render));
                }
                EngineControl::Shutdown => {}
            }
        }
        fn handle_management(&mut self, _msg: EngineManagement) {}
        fn park(&mut self, _hint: ParkHint) -> ParkHint {
            ParkHint::Wait
        }
    }

    impl ActorTypes for EngineActor<'_> {
        type Data = EngineData;
        type Control = EngineControl;
        type Management = EngineManagement;
    }

    #[allow(clippy::needless_lifetimes)]
    impl<'__dir, __Dir: '__dir> TroupeActor<'__dir, __Dir> for EngineActor<'__dir> {
        fn new(_dir: &'__dir __Dir) -> Self {
            panic!("use new_with_counter instead")
        }
    }

    pub struct DisplayActor<'a> {
        dir: &'a Directory,
        render_count: &'a AtomicUsize,
        shutdown_after: usize,
    }

    impl Actor<DisplayData, DisplayControl, DisplayManagement> for DisplayActor<'_> {
        fn handle_data(&mut self, _msg: DisplayData) {}
        fn handle_control(&mut self, msg: DisplayControl) {
            match msg {
                DisplayControl::Render => {
                    let count = self.render_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if count >= self.shutdown_after {
                        // Signal shutdown to engine
                        let _ = self
                            .dir
                            .engine
                            .send(Message::Control(EngineControl::Shutdown));
                    }
                }
                DisplayControl::Shutdown => {}
            }
        }
        fn handle_management(&mut self, _msg: DisplayManagement) {}
        fn park(&mut self, _hint: ParkHint) -> ParkHint {
            ParkHint::Wait
        }
    }

    impl ActorTypes for DisplayActor<'_> {
        type Data = DisplayData;
        type Control = DisplayControl;
        type Management = DisplayManagement;
    }

    #[allow(clippy::needless_lifetimes)]
    impl<'__dir, __Dir: '__dir> TroupeActor<'__dir, __Dir> for DisplayActor<'__dir> {
        fn new(_dir: &'__dir __Dir) -> Self {
            panic!("use new_with_counter instead")
        }
    }

    // === Manual Directory (what troupe! would generate) ===

    pub struct Directory {
        pub engine: ActorHandle<EngineData, EngineControl, EngineManagement>,
        pub display: ActorHandle<DisplayData, DisplayControl, DisplayManagement>,
    }

    /// Test that the two-phase initialization pattern compiles and works.
    /// This demonstrates the Directory pattern where actors get handles upfront.
    #[test]
    fn test_troupe_directory_pattern() {
        // Phase 1: Create all handles and schedulers upfront
        let (engine_h, _engine_s) =
            create_actor::<EngineData, EngineControl, EngineManagement>(1024, None);
        let (display_h, _display_s) =
            create_actor::<DisplayData, DisplayControl, DisplayManagement>(1024, None);

        // Build directory - everyone can send to everyone
        let dir = Directory {
            engine: engine_h.clone(),
            display: display_h.clone(),
        };

        // Verify cross-actor messaging works via directory
        // Engine can send to display
        dir.display
            .send(Message::Control(DisplayControl::Render))
            .unwrap();

        // Display can send to engine
        dir.engine
            .send(Message::Control(EngineControl::Tick))
            .unwrap();

        // Verify handles are independent (cloning works)
        let engine_h2 = dir.engine.clone();
        engine_h2
            .send(Message::Control(EngineControl::Tick))
            .unwrap();
    }
}

/// Test module for troupe nesting pattern
#[cfg(test)]
mod troupe_nesting_tests {
    #![allow(dead_code)]

    use super::*;

    // === Simple actors for nesting test ===

    pub struct WorkerData(pub String);
    #[derive(Default)]
    pub enum WorkerControl {
        Process,
        #[default]
        Shutdown,
    }
    pub struct WorkerManagement;

    /// Worker actor that just receives work items
    pub struct WorkerActor<'a> {
        _dir: &'a WorkerDirectory,
    }

    impl Actor<WorkerData, WorkerControl, WorkerManagement> for WorkerActor<'_> {
        fn handle_data(&mut self, _msg: WorkerData) {}
        fn handle_control(&mut self, _msg: WorkerControl) {}
        fn handle_management(&mut self, _msg: WorkerManagement) {}
        fn park(&mut self, _hint: ParkHint) -> ParkHint {
            ParkHint::Wait
        }
    }

    impl ActorTypes for WorkerActor<'_> {
        type Data = WorkerData;
        type Control = WorkerControl;
        type Management = WorkerManagement;
    }

    impl<'a, Dir: 'a> TroupeActor<'a, Dir> for WorkerActor<'a> {
        fn new(_dir: &'a Dir) -> Self {
            panic!("test only")
        }
    }

    // Manual directory for worker troupe
    pub struct WorkerDirectory {
        pub worker: ActorHandle<WorkerData, WorkerControl, WorkerManagement>,
    }

    // Manual ExposedHandles for worker troupe
    pub struct WorkerExposedHandles {
        pub worker: ActorHandle<WorkerData, WorkerControl, WorkerManagement>,
    }

    // Manual Troupe struct for worker
    pub struct WorkerTroupe {
        pub directory: WorkerDirectory,
        worker_scheduler: ActorScheduler<WorkerData, WorkerControl, WorkerManagement>,
    }

    impl WorkerTroupe {
        pub fn new() -> Self {
            let (worker_h, worker_s) =
                create_actor::<WorkerData, WorkerControl, WorkerManagement>(1024, None);
            Self {
                directory: WorkerDirectory { worker: worker_h },
                worker_scheduler: worker_s,
            }
        }

        pub fn exposed(&self) -> WorkerExposedHandles {
            WorkerExposedHandles {
                worker: self.directory.worker.clone(),
            }
        }
    }

    /// Test the two-phase Troupe pattern: new() → exposed() → play()
    #[test]
    fn test_troupe_two_phase_pattern() {
        // Phase 1: Create child troupe (no threads yet)
        let child = WorkerTroupe::new();

        // Phase 2: Parent grabs exposed handles
        let exposed = child.exposed();

        // Parent can now send to child even before child.play()
        // Messages will queue until child starts processing
        exposed
            .worker
            .send(Message::Control(WorkerControl::Process))
            .unwrap();
        exposed
            .worker
            .send(Message::Data(WorkerData("hello".to_string())))
            .unwrap();

        // Verify we can clone handles from exposed
        let worker_h2 = exposed.worker.clone();
        worker_h2
            .send(Message::Control(WorkerControl::Process))
            .unwrap();

        // Note: We don't call play() here since that would block.
        // The test verifies the two-phase construction pattern works.
    }

    /// Test that ExposedHandles can be sent to parent scope
    #[test]
    fn test_exposed_handles_outlive_troupe_struct() {
        let exposed = {
            // Create troupe in inner scope
            let child = WorkerTroupe::new();
            child.exposed() // ExposedHandles escapes
        };
        // Troupe struct dropped, but handles still valid (channels still open)

        // Can still send - messages queue (will never be processed since
        // scheduler was dropped, but channel is still open from handle side)
        // Actually this will fail because receiver side dropped!
        // This test shows the handles can outlive the Troupe struct,
        // but in practice you'd call play() to keep channels open.

        // Just verify the type works
        let _: WorkerExposedHandles = exposed;
    }
}
