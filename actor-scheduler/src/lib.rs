//! Priority Channel - A multi-priority message passing system
//!
//! This crate provides a message scheduler with three priority levels:
//! - **Control**: Highest priority, burst-limited to prevent starvation
//! - **Management**: Medium priority, burst-limited
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
//! use actor_scheduler::{ActorScheduler, Message, SchedulerHandler, ActorStatus, SystemStatus, HandlerResult, HandlerError};
//!
//! struct MyHandler;
//!
//! impl SchedulerHandler<String, String, String> for MyHandler {
//!     fn handle_data(&mut self, msg: String) -> HandlerResult {
//!         println!("Data: {}", msg);
//!         Ok(())
//!     }
//!     fn handle_control(&mut self, msg: String) -> HandlerResult {
//!         println!("Control: {}", msg);
//!         Ok(())
//!     }
//!     fn handle_management(&mut self, msg: String) -> HandlerResult {
//!         println!("Management: {}", msg);
//!         Ok(())
//!     }
//!     fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> { Ok(ActorStatus::Idle) }
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

use error::DrainStatus;
pub use error::{HandlerError, HandlerResult, SendError};

// Re-export macros from the proc-macro crate
pub use actor_scheduler_macros::{actor_impl, troupe};

use std::sync::{
    Arc,
    mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
};
use std::time::Duration;

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
/// - **Sender**: Retries with exponential backoff if buffer full (prevents slow-loris attacks)
/// - **Receiver**: Drains before Management and Data, with burst limit to prevent starvation
/// - **Guarantee**: Best-effort priority delivery (bounded buffer with backoff)
/// - **Ordering**: FIFO within lane, typically processed before Data/Management
///
/// **Example**: User input (keypresses, mouse), window resize, close requests
///
/// ## Management Lane (M)
///
/// **Purpose**: Configuration and lifecycle messages.
///
/// **Contract**:
/// - **Sender**: Retries with exponential backoff if buffer full
/// - **Receiver**: Drains between Control and Data, with burst limiting
/// - **Guarantee**: Best-effort delivery (bounded buffer with backoff)
/// - **Ordering**: FIFO within lane
///
/// **Example**: Configuration changes, resource allocation, subscription/unsubscription
///
/// # Scheduling Strategy
///
/// The scheduler drains messages in priority order with burst limits:
///
/// ```text
/// Loop:
///   1. Drain Control messages (capped at burst limit)
///   2. Drain Management messages (capped at burst limit)
///   3. Drain Control messages again (priority recheck)
///   4. Drain Data messages (capped at burst limit)
///   5. Call park() - let actor/OS do other work
///   6. Repeat
/// ```
///
/// This provides best-effort priority with starvation protection:
/// - Control messages typically process before Data/Management
/// - All lanes are burst-limited to prevent monopolization
/// - No cross-lane ordering guarantees - only best-effort priority
/// - Protection against slow-loris attacks (poorly-behaved senders can't drown channels)
///
/// Configurable shutdown behavior per actor via `ShutdownMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShutdownMode {
    /// Exit immediately, drop all pending messages (default, current behavior)
    #[default]
    Immediate,

    /// Drain control+management lanes, drop data
    /// Use for actors where control/management cleanup is critical
    DrainControl,

    /// Process all pending messages before exit (with timeout fallback)
    /// Use for actors that must process all messages (e.g., logging, persistence)
    DrainAll { timeout: std::time::Duration },
}

#[derive(Debug)]
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
    /// - Retries with exponential backoff if buffer is full
    /// - Use for messages that need priority (user input, resize events)
    /// - Backoff prevents poorly-behaved senders from monopolizing the channel
    ///
    /// **Receiver** (Actor):
    /// - Will receive via `handle_control()`
    /// - Best-effort priority processing before Data/Management messages
    /// - Draining is burst-limited to prevent starvation of other lanes
    ///
    /// # Example
    ///
    /// ```ignore
    /// // User clicked the close button - this should be processed with priority
    /// tx.send(Message::Control(CloseRequested))?;
    /// // Retries with backoff if buffer full
    /// ```
    ///
    /// # Backpressure
    ///
    /// Control messages use a bounded buffer with exponential backoff on retry.
    /// If senders overwhelm the receiver, they will experience increasing delays.
    /// This prevents poorly-behaved senders from monopolizing the control channel.
    Control(C),

    /// A management message (medium priority, configuration/lifecycle).
    ///
    /// # Contract
    ///
    /// **Sender**:
    /// - Retries with exponential backoff if buffer is full
    /// - Use for lifecycle and configuration (create, destroy, configure)
    /// - Lower priority than Control but higher than Data
    ///
    /// **Receiver** (Actor):
    /// - Will receive via `handle_management()`
    /// - Best-effort delivery (bounded buffer with backoff)
    /// - Typically processed after Control but before Data messages
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

/// Actor status returned from park() to hint the scheduler about blocking behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorStatus {
    Idle, // Actor has no unfinished work. Scheduler can block. (0% CPU)
    Busy, // Actor has unfinished work (yielding). Scheduler should poll.
}

/// Status provided to the actor's park method indicating the state of the scheduler's queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemStatus {
    Idle, // Scheduler queues are empty
    Busy, // Scheduler queues have more work (burst limit reached)
}

/// The Actor trait - implement this to define your actor's behavior.
///
/// Actors process messages from three priority lanes:
/// - **Data** (D): High-throughput data messages
/// - **Control** (C): Time-critical control messages
/// - **Management** (M): Lifecycle and configuration messages
pub trait Actor<D, C, M> {
    /// Handle a data message.
    ///
    /// Returns `Ok(())` on success, or a `HandlerError` on failure.
    /// - `HandlerError::Temporary`: Scheduler logs and continues
    /// - `HandlerError::Fatal`: Scheduler initiates shutdown
    fn handle_data(&mut self, msg: D) -> HandlerResult;

    /// Handle a control message.
    ///
    /// Returns `Ok(())` on success, or a `HandlerError` on failure.
    /// - `HandlerError::Temporary`: Scheduler logs and continues
    /// - `HandlerError::Fatal`: Scheduler initiates shutdown
    fn handle_control(&mut self, msg: C) -> HandlerResult;

    /// Handle a management message.
    ///
    /// Returns `Ok(())` on success, or a `HandlerError` on failure.
    /// - `HandlerError::Temporary`: Scheduler logs and continues
    /// - `HandlerError::Fatal`: Scheduler initiates shutdown
    fn handle_management(&mut self, msg: M) -> HandlerResult;

    /// The "Hook" where the Actor creates the bridge to the OS.
    /// Called when the scheduler has drained available messages (or hit burst limits).
    ///
    /// Returns actor status: Busy if yielding with unfinished work, Idle if done.
    /// Can return `HandlerError::Fatal` to trigger shutdown.
    fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError>;
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
///     fn park(&mut self, status: SystemStatus) -> ActorStatus { ActorStatus::Idle }
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
#[must_use]
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
/// Smaller buffer forces faster detection of overload scenarios
const CONTROL_MGMT_BUFFER_SIZE: usize = 32;

/// Minimum backoff duration when control/management channels are full
/// High enough to prevent oscillation where senders retry faster than receiver can drain
const MIN_BACKOFF: Duration = Duration::from_millis(1);

/// Maximum backoff duration when control/management channels are full
/// Large enough to guarantee the control channel can be fully drained (10x buffer size)
/// At ~1µs per message, 320 messages = ~320µs, so 5s is extremely generous
const MAX_BACKOFF: Duration = Duration::from_secs(5);

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
    // Use wall clock time for actual randomness (prevents thundering herd)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0));

    // Mix nanoseconds with attempt number for better distribution across threads
    let hash = (now.as_nanos() as u64 ^ (attempt as u64).wrapping_mul(0x517cc1b727220a95))
        .wrapping_mul(JITTER_HASH_CONSTANT);

    let jitter_pct = JITTER_MIN_PCT + (hash % JITTER_RANGE);
    let jittered_micros = (backoff_micros * jitter_pct) / 100;

    Ok(Duration::from_micros(jittered_micros))
}

/// System messages - combines wake and shutdown into one channel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum System {
    /// Wake the scheduler to process messages
    Wake,
    /// Shutdown the scheduler
    Shutdown,
}

/// A unified sender handle that routes messages to the scheduler with priority lanes.
pub struct ActorHandle<D, C, M> {
    // Doorbell channel (buffer: 1) - wake and shutdown signals
    tx_doorbell: SyncSender<System>,
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
            tx_data: self.tx_data.clone(),
            tx_control: self.tx_control.clone(),
            tx_mgmt: self.tx_mgmt.clone(),
            wake_handler: self.wake_handler.clone(),
        }
    }
}

/// Number of immediate retries (spin) before yielding
/// At ~10-20ns per spin, 100 spins = ~1-2µs (less than context switch cost)
const SPIN_ATTEMPTS: u32 = 100;

/// Number of yield attempts before escalating to sleep
/// After hot spinning, cooperatively yield to let receiver process
const YIELD_ATTEMPTS: u32 = 20;

/// Send with retry and exponential backoff + jitter for fairness.
///
/// Backoff strategy:
/// 1. Spin (immediate retry) for first few attempts
/// 2. Yield (cooperative) for next few attempts
/// 3. Sleep (blocking) with exponential backoff for remaining attempts
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
                // Restore message for retry
                msg = returned_msg;

                // Backoff strategy: spin → yield → sleep
                if attempt < SPIN_ATTEMPTS {
                    // Phase 1: Spin (immediate retry, hot loop)
                    // No sleep/yield - just retry immediately
                } else if attempt < SPIN_ATTEMPTS + YIELD_ATTEMPTS {
                    // Phase 2: Yield (cooperative, let other threads run)
                    std::thread::yield_now();
                } else {
                    // Phase 3: Sleep (exponential backoff with jitter)
                    #[cfg(debug_assertions)]
                    if attempt % 10 == 0 {
                        eprintln!(
                            "[ActorScheduler] Priority channel full, backing off (attempt {})",
                            attempt
                        );
                    }

                    let sleep_attempt = attempt - (SPIN_ATTEMPTS + YIELD_ATTEMPTS);
                    let backoff = backoff_with_jitter(sleep_attempt)?;
                    std::thread::sleep(backoff);
                }

                attempt = attempt.saturating_add(1);
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
                // Try send first to avoid blocking if possible
                if let Err(std::sync::mpsc::TrySendError::Full(returned_d)) =
                    self.tx_data.try_send(d)
                {
                    #[cfg(debug_assertions)]
                    eprintln!("[ActorScheduler] Data channel full, blocking send...");

                    self.tx_data.send(returned_d)?;
                }
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
                // Shutdown: blocking send to guarantee delivery
                // Actor must be running before calling this (doorbell will be drained)
                self.tx_doorbell.send(System::Shutdown)?;

                // Also call custom wake handler if present
                if let Some(waker) = &self.wake_handler {
                    waker.wake();
                }
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
        match self.tx_doorbell.try_send(System::Wake) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                // Doorbell is bounded(1) - if full, a wake is already pending
            }
            Err(TrySendError::Disconnected(_)) => {
                panic!("Doorbell receiver disconnected - scheduler dropped unexpectedly");
            }
        }
    }
}

/// The receiver side that implements the priority scheduling logic.
pub struct ActorScheduler<D, C, M> {
    rx_doorbell: Receiver<System>, // Wake and shutdown signals
    rx_data: Receiver<D>,
    rx_control: Receiver<C>,
    rx_mgmt: Receiver<M>,
    data_burst_limit: usize,
    management_burst_limit: usize,
    control_burst_limit: usize,
    shutdown_mode: ShutdownMode,
}

/// System status after processing messages
enum SchedulerLoopStatus {
    /// More work available, keep polling
    Working,
    /// Queues drained, can block
    Idle,
}

impl<D, C, M> ActorScheduler<D, C, M> {
    /// Drain a channel up to a limit, applying a handler to each message.
    ///
    /// Returns:
    /// - `Ok(DrainStatus::Empty)` - Channel drained, no more messages
    /// - `Ok(DrainStatus::More)` - Hit burst limit, more messages may exist
    /// - `Ok(DrainStatus::Disconnected)` - Channel disconnected (normal shutdown)
    /// - `Err(HandlerError)` - Handler failed, propagate error up
    fn drain_channel<T>(
        rx: &Receiver<T>,
        limit: usize,
        mut handler: impl FnMut(T) -> HandlerResult,
    ) -> Result<DrainStatus, HandlerError> {
        let mut count = 0;

        loop {
            if count >= limit {
                return Ok(DrainStatus::More);
            }

            match rx.try_recv() {
                Ok(msg) => {
                    handler(msg)?;
                    count += 1;
                }
                Err(TryRecvError::Empty) => {
                    return Ok(DrainStatus::Empty);
                }
                Err(TryRecvError::Disconnected) => {
                    return Ok(DrainStatus::Disconnected);
                }
            }
        }
    }

    /// Drain control and management channels without limit, ignoring data.
    ///
    /// Used for `ShutdownMode::DrainControl` to process critical cleanup messages
    /// while dropping lower-priority data messages.
    fn drain_control_and_management<A>(&mut self, actor: &mut A) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        // Drain control completely
        while let DrainStatus::More = Self::drain_channel(&self.rx_control, usize::MAX, |msg| {
            actor.handle_control(msg)
        })? {}

        // Drain management completely
        while let DrainStatus::More = Self::drain_channel(&self.rx_mgmt, usize::MAX, |msg| {
            actor.handle_management(msg)
        })? {}

        Ok(())
    }

    /// Drain all channels (control, management, data) with timeout fallback.
    ///
    /// Used for `ShutdownMode::DrainAll` to process all pending messages before shutdown.
    /// If the timeout is exceeded, remaining messages are dropped.
    fn drain_all_with_timeout<A>(
        &mut self,
        actor: &mut A,
        timeout: std::time::Duration,
    ) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        use std::time::Instant;

        let deadline = Instant::now() + timeout;
        let batch_size = 10;

        loop {
            let control_status = Self::drain_channel(&self.rx_control, batch_size, |msg| {
                actor.handle_control(msg)
            })?;
            if Instant::now() >= deadline {
                return Ok(());
            }

            let mgmt_status = Self::drain_channel(&self.rx_mgmt, batch_size, |msg| {
                actor.handle_management(msg)
            })?;
            if Instant::now() >= deadline {
                return Ok(());
            }

            let data_status =
                Self::drain_channel(&self.rx_data, batch_size, |msg| actor.handle_data(msg))?;
            if Instant::now() >= deadline {
                return Ok(());
            }

            // Done when all channels are empty or disconnected
            let all_done = !matches!(control_status, DrainStatus::More)
                && !matches!(mgmt_status, DrainStatus::More)
                && !matches!(data_status, DrainStatus::More);

            if all_done {
                return Ok(());
            }
        }
    }

    /// Process messages from all priority lanes, return status.
    ///
    /// Returns:
    /// - `Ok(Some(status))` - Processed messages, continue with given status
    /// - `Ok(None)` - All channels disconnected, normal shutdown
    /// - `Err(HandlerError)` - Handler failed
    #[inline]
    fn handle_wake<A>(&mut self, actor: &mut A) -> Result<Option<SchedulerLoopStatus>, HandlerError>
    where
        A: Actor<D, C, M>,
    {
        // Drain Control → Mgmt → Control → Data
        // Control budget is split evenly between the two control runs to prevent double priority
        let half_control = self.control_burst_limit / 2;

        let control1 = Self::drain_channel(&self.rx_control, half_control, |msg| {
            actor.handle_control(msg)
        })?;

        let mgmt = Self::drain_channel(&self.rx_mgmt, self.management_burst_limit, |msg| {
            actor.handle_management(msg)
        })?;

        let control2 = Self::drain_channel(&self.rx_control, half_control, |msg| {
            actor.handle_control(msg)
        })?;

        let data = Self::drain_channel(&self.rx_data, self.data_burst_limit, |msg| {
            actor.handle_data(msg)
        })?;

        // All disconnected = normal shutdown
        if matches!(
            (&control1, &mgmt, &control2, &data),
            (
                DrainStatus::Disconnected,
                DrainStatus::Disconnected,
                DrainStatus::Disconnected,
                DrainStatus::Disconnected
            )
        ) {
            return Ok(None);
        }

        // Any channel hit burst limit = more work available
        let more_work = matches!(control1, DrainStatus::More)
            || matches!(mgmt, DrainStatus::More)
            || matches!(control2, DrainStatus::More)
            || matches!(data, DrainStatus::More);

        let system_status = if more_work {
            SystemStatus::Busy
        } else {
            SystemStatus::Idle
        };

        let returned_hint = actor.park(system_status)?;

        let status = if more_work || returned_hint == ActorStatus::Busy {
            SchedulerLoopStatus::Working
        } else {
            SchedulerLoopStatus::Idle
        };

        Ok(Some(status))
    }

    #[cold]
    fn handle_shutdown<A>(&mut self, actor: &mut A) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        match self.shutdown_mode {
            ShutdownMode::Immediate => Ok(()),
            ShutdownMode::DrainControl => self.drain_control_and_management(actor),
            ShutdownMode::DrainAll { timeout } => self.drain_all_with_timeout(actor, timeout),
        }
    }

    /// Create a new scheduler channel with priority lanes.
    ///
    /// # Arguments
    /// * `data_burst_limit` - Maximum data messages to process per wake cycle
    /// * `data_buffer_size` - Size of bounded data buffer (backpressure threshold).
    ///   Must be >= 1. A buffer of 0 creates a rendezvous channel that will deadlock
    ///   if the receiver isn't actively receiving.
    ///
    /// # Panics
    /// Panics if `data_buffer_size` is 0.
    ///
    /// # Returns
    /// Returns `(sender, receiver)` tuple. The sender can be cloned and shared.
    #[must_use]
    pub fn new(data_burst_limit: usize, data_buffer_size: usize) -> (ActorHandle<D, C, M>, Self) {
        assert!(
            data_buffer_size > 0,
            "data_buffer_size must be >= 1, got {}",
            data_buffer_size
        );

        let (tx_doorbell, rx_doorbell) = mpsc::sync_channel(1);
        let (tx_data, rx_data) = mpsc::sync_channel(data_buffer_size);
        let (tx_control, rx_control) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);
        let (tx_mgmt, rx_mgmt) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);

        let sender = ActorHandle {
            tx_doorbell,
            tx_data,
            tx_control,
            tx_mgmt,
            wake_handler: None,
        };

        let receiver = ActorScheduler {
            rx_doorbell,
            rx_data,
            rx_control,
            rx_mgmt,
            data_burst_limit,
            management_burst_limit: CONTROL_MGMT_BUFFER_SIZE,
            control_burst_limit: CONTROL_MGMT_BUFFER_SIZE * 10,
            shutdown_mode: ShutdownMode::default(),
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
    /// * `data_buffer_size` - Size of bounded data buffer (backpressure threshold).
    ///   Must be >= 1.
    /// * `wake_handler` - Optional custom wake handler for platform event loops
    ///
    /// # Panics
    /// Panics if `data_buffer_size` is 0.
    ///
    /// # Returns
    /// Returns `(sender, receiver)` tuple. The sender can be cloned and shared.
    #[must_use]
    pub fn new_with_wake_handler(
        data_burst_limit: usize,
        data_buffer_size: usize,
        wake_handler: Option<Arc<dyn WakeHandler>>,
    ) -> (ActorHandle<D, C, M>, Self) {
        assert!(
            data_buffer_size > 0,
            "data_buffer_size must be >= 1, got {}",
            data_buffer_size
        );

        let (tx_doorbell, rx_doorbell) = mpsc::sync_channel(1);
        let (tx_data, rx_data) = mpsc::sync_channel(data_buffer_size);
        let (tx_control, rx_control) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);
        let (tx_mgmt, rx_mgmt) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);

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
            data_burst_limit,
            management_burst_limit: CONTROL_MGMT_BUFFER_SIZE,
            control_burst_limit: CONTROL_MGMT_BUFFER_SIZE * 10,
            shutdown_mode: ShutdownMode::default(),
        };

        (sender, receiver)
    }

    /// Create a new scheduler channel with configurable shutdown behavior.
    ///
    /// This variant allows specifying how the actor should behave on shutdown:
    /// - `Immediate`: Drop all pending messages (default)
    /// - `DrainControl`: Process control+management, drop data
    /// - `DrainAll`: Process all messages with timeout fallback
    ///
    /// # Arguments
    /// * `data_burst_limit` - Maximum data messages to process per wake cycle
    /// * `data_buffer_size` - Size of bounded data buffer (backpressure threshold).
    ///   Must be >= 1.
    /// * `shutdown_mode` - Shutdown behavior (see `ShutdownMode`)
    ///
    /// # Panics
    /// Panics if `data_buffer_size` is 0.
    ///
    /// # Returns
    /// Returns `(sender, receiver)` tuple. The sender can be cloned and shared.
    #[must_use]
    pub fn new_with_shutdown_mode(
        data_burst_limit: usize,
        data_buffer_size: usize,
        shutdown_mode: ShutdownMode,
    ) -> (ActorHandle<D, C, M>, Self) {
        assert!(
            data_buffer_size > 0,
            "data_buffer_size must be >= 1, got {}",
            data_buffer_size
        );

        let (tx_doorbell, rx_doorbell) = mpsc::sync_channel(1);
        let (tx_data, rx_data) = mpsc::sync_channel(data_buffer_size);
        let (tx_control, rx_control) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);
        let (tx_mgmt, rx_mgmt) = mpsc::sync_channel(CONTROL_MGMT_BUFFER_SIZE);

        let sender = ActorHandle {
            tx_doorbell,
            tx_data,
            tx_control,
            tx_mgmt,
            wake_handler: None,
        };

        let receiver = ActorScheduler {
            rx_doorbell,
            rx_data,
            rx_control,
            rx_mgmt,
            data_burst_limit,
            management_burst_limit: CONTROL_MGMT_BUFFER_SIZE,
            control_burst_limit: CONTROL_MGMT_BUFFER_SIZE * 10,
            shutdown_mode,
        };

        (sender, receiver)
    }

    /// The Main Scheduler Loop.
    /// Blocks on the Doorbell channel. Prioritizes Shutdown > Control > Management > Data.
    ///
    /// # Arguments
    /// * `actor` - Implementation of `Actor` trait
    ///
    /// # Exit conditions
    /// - `Shutdown` message received → drains per shutdown_mode, returns
    /// - All senders dropped → returns
    /// - Handler returns `Recoverable` error → returns (supervisor can restart)
    /// - Handler returns `Fatal` error → panics
    pub fn run<A>(&mut self, actor: &mut A)
    where
        A: Actor<D, C, M>,
    {
        if let Err(e) = self.run_inner(actor) {
            match e {
                HandlerError::Recoverable(_) => {
                    // Return normally - supervisor can restart
                }
                HandlerError::Fatal(msg) => {
                    panic!("Actor fatal error: {}", msg);
                }
            }
        }
    }

    fn run_inner<A>(&mut self, actor: &mut A) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        use std::sync::mpsc::TryRecvError;

        let mut working = false;

        loop {
            let signal = if working {
                self.rx_doorbell.try_recv()
            } else {
                self.rx_doorbell
                    .recv()
                    .map_err(|_| TryRecvError::Disconnected)
            };

            match signal {
                Ok(System::Shutdown) => {
                    self.handle_shutdown(actor)?;
                    return Ok(());
                }
                Ok(System::Wake) | Err(TryRecvError::Empty) => {
                    match self.handle_wake(actor)? {
                        Some(status) => {
                            working = matches!(status, SchedulerLoopStatus::Working);
                        }
                        None => return Ok(()), // All channels disconnected
                    }
                }
                Err(TryRecvError::Disconnected) => return Ok(()),
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
        fn handle_data(&mut self, msg: String) -> HandlerResult {
            self.log.lock().unwrap().push(format!("Data: {}", msg));
            Ok(())
        }
        fn handle_control(&mut self, msg: String) -> HandlerResult {
            self.log.lock().unwrap().push(format!("Ctrl: {}", msg));
            Ok(())
        }
        fn handle_management(&mut self, msg: String) -> HandlerResult {
            self.log.lock().unwrap().push(format!("Mgmt: {}", msg));
            Ok(())
        }

        fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
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
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                self.data_count += 1;
                Ok(())
            }
            fn handle_control(&mut self, _: String) -> HandlerResult {
                self.ctrl_count += 1;
                Ok(())
            }
            fn handle_management(&mut self, _: bool) -> HandlerResult {
                self.mgmt_count += 1;
                Ok(())
            }
            fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
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
                fn handle_data(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn handle_control(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn handle_management(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Idle)
                }
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
        fn handle_data(&mut self, _msg: EngineData) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, msg: EngineControl) -> HandlerResult {
            match msg {
                EngineControl::Tick => {
                    self.tick_count.fetch_add(1, Ordering::SeqCst);
                    self.dir
                        .display
                        .send(Message::Control(DisplayControl::Render))
                        .expect("Failed to send render command to display actor");
                }
                EngineControl::Shutdown => {}
            }
            Ok(())
        }
        fn handle_management(&mut self, _msg: EngineManagement) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
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
        fn handle_data(&mut self, _msg: DisplayData) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, msg: DisplayControl) -> HandlerResult {
            match msg {
                DisplayControl::Render => {
                    let count = self.render_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if count >= self.shutdown_after {
                        self.dir
                            .engine
                            .send(Message::Control(EngineControl::Shutdown))
                            .expect("Failed to send shutdown to engine");
                    }
                }
                DisplayControl::Shutdown => {}
            }
            Ok(())
        }
        fn handle_management(&mut self, _msg: DisplayManagement) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
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

    /// Adversarial test: Malicious control sender trying to starve data lane
    /// Uses CONTINUOUS flooding to ensure burst limiting works during active attack
    #[test]
    fn adversarial_control_flood_vs_data() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let (tx, mut rx) = ActorScheduler::<i32, (), ()>::new(100, 100);

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

        // Receiver thread
        let receiver_handle = thread::spawn(move || {
            struct TestActor {
                control_count: Arc<AtomicUsize>,
                data_count: Arc<AtomicUsize>,
            }
            impl Actor<i32, (), ()> for TestActor {
                fn handle_control(&mut self, _: ()) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_data(&mut self, _: i32) -> HandlerResult {
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_management(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy) // Keep spinning to maximize throughput
                }
            }

            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Malicious control sender: CONTINUOUS flood
        let tx_control = tx.clone();
        let stop_flag = stop_flooding.clone();
        let control_sender = thread::spawn(move || {
            let mut sent = 0;
            while !stop_flag.load(Ordering::Relaxed) {
                if tx_control.send(Message::Control(())).is_ok() {
                    sent += 1;
                }
            }
            sent
        });

        // Give flooder time to start
        thread::sleep(Duration::from_millis(20));

        // Well-behaved data sender sends DURING the flood
        let tx_data = tx.clone();
        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                let _ = tx_data.send(Message::Data(i));
            }
        });

        data_sender.join().unwrap();

        // Let it run a bit more
        thread::sleep(Duration::from_millis(50));

        // Stop the flood
        stop_flooding.store(true, Ordering::Relaxed);
        let control_sent = control_sender.join().unwrap();

        // Give receiver time to drain
        thread::sleep(Duration::from_millis(50));
        drop(tx);
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Control flood vs data - Control sent: {}, processed: {}, Data processed: {}/100",
            control_sent, control_count, data_count
        );

        // CRITICAL: Data must get through DURING continuous control flood
        assert!(
            data_count > 0,
            "Data lane was completely starved during continuous control flood"
        );

        // With burst limiting, we should process a good portion of data
        assert!(
            data_count > 50,
            "Burst limiting too weak - only {}/100 data processed during flood",
            data_count
        );
    }

    /// Adversarial test: Multiple bad actors teaming up to flood control
    /// Uses CONTINUOUS flooding from multiple threads to test collusion resistance
    #[test]
    fn adversarial_multiple_control_flooders() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let (tx, mut rx) = ActorScheduler::<i32, (), ()>::new(100, 100);

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

        // Receiver thread
        let receiver_handle = thread::spawn(move || {
            struct TestActor {
                control_count: Arc<AtomicUsize>,
                data_count: Arc<AtomicUsize>,
            }
            impl Actor<i32, (), ()> for TestActor {
                fn handle_control(&mut self, _: ()) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_data(&mut self, _: i32) -> HandlerResult {
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_management(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }

            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Spawn 5 malicious control senders - CONTINUOUS flooding
        let mut control_handles = vec![];
        for _ in 0..5 {
            let tx_clone = tx.clone();
            let stop_flag = stop_flooding.clone();
            let handle = thread::spawn(move || {
                let mut sent = 0;
                while !stop_flag.load(Ordering::Relaxed) {
                    if tx_clone.send(Message::Control(())).is_ok() {
                        sent += 1;
                    }
                }
                sent
            });
            control_handles.push(handle);
        }

        // Give flooders time to start
        thread::sleep(Duration::from_millis(20));

        // Well-behaved data sender sends DURING the coordinated attack
        let tx_data = tx.clone();
        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                let _ = tx_data.send(Message::Data(i));
            }
        });

        data_sender.join().unwrap();

        // Let it run a bit more
        thread::sleep(Duration::from_millis(50));

        // Stop all flooders
        stop_flooding.store(true, Ordering::Relaxed);
        let mut total_control_sent = 0;
        for handle in control_handles {
            total_control_sent += handle.join().unwrap();
        }

        // Give receiver time to drain
        thread::sleep(Duration::from_millis(50));
        drop(tx);
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Multiple attackers - Control sent: {}, processed: {}, Data: {}/100",
            total_control_sent, control_count, data_count
        );

        // CRITICAL: Even with 5 coordinated attackers, data must get through
        assert!(
            data_count > 0,
            "Data lane completely starved by coordinated control attack"
        );

        // With burst limiting, we should process a good portion despite 5 attackers
        assert!(
            data_count > 50,
            "Burst limiting too weak against coordinated attack - only {}/100 data processed",
            data_count
        );
    }

    /// Adversarial test: Continuous control flood with concurrent data
    /// Validates that burst limiting allows data through DURING active control flooding
    #[test]
    fn adversarial_continuous_control_flood() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let (tx, mut rx) = ActorScheduler::<i32, (), ()>::new(100, 100);

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

        // Receiver thread
        let receiver_handle = thread::spawn(move || {
            struct TestActor {
                control_count: Arc<AtomicUsize>,
                data_count: Arc<AtomicUsize>,
            }
            impl Actor<i32, (), ()> for TestActor {
                fn handle_control(&mut self, _: ()) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_data(&mut self, _: i32) -> HandlerResult {
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                fn handle_management(&mut self, _: ()) -> HandlerResult {
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }

            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Continuous control flooder - doesn't stop until flag is set
        let tx_control = tx.clone();
        let stop_flag = stop_flooding.clone();
        let control_flooder = thread::spawn(move || {
            let mut sent = 0;
            while !stop_flag.load(Ordering::Relaxed) {
                if tx_control.send(Message::Control(())).is_ok() {
                    sent += 1;
                }
            }
            sent
        });

        // Give flooder time to start flooding
        thread::sleep(Duration::from_millis(50));

        // Now send data messages WHILE control is flooding
        let tx_data = tx.clone();
        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                let _ = tx_data.send(Message::Data(i));
            }
        });

        // Wait for data to be sent
        data_sender.join().unwrap();

        // Let receiver process for a bit more
        thread::sleep(Duration::from_millis(100));

        // Stop the flooder
        stop_flooding.store(true, Ordering::Relaxed);
        let control_sent = control_flooder.join().unwrap();

        // Give receiver time to drain
        thread::sleep(Duration::from_millis(50));
        drop(tx);
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Continuous flood - Control sent: {}, processed: {}, Data processed: {}/100",
            control_sent, control_count, data_count
        );

        // CRITICAL: Data must get through even during continuous control flooding
        // This proves burst limiting prevents control monopolization
        assert!(
            data_count > 0,
            "Burst limiting FAILED - data was starved during continuous control flood"
        );

        // We should process a significant portion of data despite the flood
        assert!(
            data_count > 50,
            "Burst limiting is too weak - only {}/100 data messages processed",
            data_count
        );
    }

    /// Adversarial test: Slow receiver with many aggressive senders
    /// Validates that backoff system prevents message loss even under heavy load
    #[test]
    fn adversarial_slow_receiver_resilience() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        // Small buffers to trigger backpressure
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 10);

        let control_processed = Arc::new(AtomicUsize::new(0));
        let mgmt_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));

        let cp = control_processed.clone();
        let mp = mgmt_processed.clone();
        let dp = data_processed.clone();

        // Slow receiver that processes steadily
        let receiver_handle = thread::spawn(move || {
            struct SlowActor {
                control_count: Arc<AtomicUsize>,
                mgmt_count: Arc<AtomicUsize>,
                data_count: Arc<AtomicUsize>,
            }
            impl Actor<i32, i32, i32> for SlowActor {
                fn handle_control(&mut self, _: i32) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(2));
                    Ok(())
                }
                fn handle_data(&mut self, _: i32) -> HandlerResult {
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(2));
                    Ok(())
                }
                fn handle_management(&mut self, _: i32) -> HandlerResult {
                    self.mgmt_count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(2));
                    Ok(())
                }
                fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }

            let mut actor = SlowActor {
                control_count: cp,
                mgmt_count: mp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Spawn multiple aggressive senders
        let mut sender_handles = vec![];
        for sender_id in 0..3 {
            let tx_clone = tx.clone();
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let msg_val = sender_id * 1000 + i;
                    // Flood all lanes - backoff should prevent loss
                    let _ = tx_clone.send(Message::Control(msg_val));
                    let _ = tx_clone.send(Message::Management(msg_val));
                }
            });
            sender_handles.push(handle);
        }

        for handle in sender_handles {
            handle.join().unwrap();
        }

        // Give receiver time to drain
        thread::sleep(Duration::from_millis(1000));
        drop(tx);
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let mgmt_count = mgmt_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Slow receiver resilience - Control: {}, Mgmt: {}, Data: {}",
            control_count, mgmt_count, data_count
        );

        // Backoff system should allow all messages to eventually get through
        // 3 senders * 100 messages each = 300 per lane
        assert_eq!(
            control_count, 300,
            "Backoff should allow all control messages through"
        );
        assert_eq!(
            mgmt_count, 300,
            "Backoff should allow all management messages through"
        );
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
        fn handle_data(&mut self, _msg: WorkerData) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _msg: WorkerControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _msg: WorkerManagement) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
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

#[cfg(test)]
mod shutdown_tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;
    use std::time::Duration;

    struct CountingActor {
        data_count: Arc<AtomicUsize>,
        control_count: Arc<AtomicUsize>,
        mgmt_count: Arc<AtomicUsize>,
    }

    impl Actor<i32, (), ()> for CountingActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn handle_control(&mut self, _: ()) -> HandlerResult {
            self.control_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn handle_management(&mut self, _: ()) -> HandlerResult {
            self.mgmt_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            match status {
                SystemStatus::Idle => Ok(ActorStatus::Idle),
                SystemStatus::Busy => Ok(ActorStatus::Busy),
            }
        }
    }

    #[test]
    fn test_shutdown_immediate_exits_quickly_under_flood() {
        let (tx, mut rx) =
            ActorScheduler::new_with_shutdown_mode(100, 100, ShutdownMode::Immediate);

        let data_count = Arc::new(AtomicUsize::new(0));
        let control_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        let actor_data = data_count.clone();
        let actor_control = control_count.clone();
        let actor_mgmt = mgmt_count.clone();

        let actor_handle = thread::spawn(move || {
            let mut actor = CountingActor {
                data_count: actor_data,
                control_count: actor_control,
                mgmt_count: actor_mgmt,
            };
            rx.run(&mut actor);
        });

        // Flood with data messages
        for i in 0..1000 {
            let _ = tx.send(Message::Data(i));
        }

        // Give time for messages to queue
        thread::sleep(Duration::from_millis(10));

        // Shutdown should return quickly even with backlog
        let shutdown_start = std::time::Instant::now();
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();
        let shutdown_duration = shutdown_start.elapsed();

        // Should shutdown within 100ms (fast, not waiting for drain)
        assert!(
            shutdown_duration < Duration::from_millis(100),
            "Immediate shutdown should exit quickly, took {:?}",
            shutdown_duration
        );
    }

    #[test]
    fn test_shutdown_drain_control_processes_control_and_mgmt() {
        let (tx, mut rx) =
            ActorScheduler::new_with_shutdown_mode(100, 100, ShutdownMode::DrainControl);

        let data_count = Arc::new(AtomicUsize::new(0));
        let control_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        let actor_data = data_count.clone();
        let actor_control = control_count.clone();
        let actor_mgmt = mgmt_count.clone();

        let actor_handle = thread::spawn(move || {
            let mut actor = CountingActor {
                data_count: actor_data,
                control_count: actor_control,
                mgmt_count: actor_mgmt,
            };
            rx.run(&mut actor);
        });

        // Send messages
        for i in 0..50 {
            tx.send(Message::Data(i)).unwrap();
        }
        for _ in 0..50 {
            tx.send(Message::Control(())).unwrap();
        }
        for _ in 0..50 {
            tx.send(Message::Management(())).unwrap();
        }

        // Give time for some to queue
        thread::sleep(Duration::from_millis(10));

        // Shutdown - should drain control+mgmt
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();

        // All control+mgmt should be processed, data may be dropped
        let control = control_count.load(Ordering::Relaxed);
        let mgmt = mgmt_count.load(Ordering::Relaxed);
        let data = data_count.load(Ordering::Relaxed);

        assert_eq!(control, 50, "All control messages should be processed");
        assert_eq!(mgmt, 50, "All management messages should be processed");
        // Data might be partially processed or dropped
        assert!(data <= 50, "Data messages may be dropped");
    }

    #[test]
    fn test_shutdown_drain_all_processes_everything() {
        let (tx, mut rx) = ActorScheduler::new_with_shutdown_mode(
            100,
            100,
            ShutdownMode::DrainAll {
                timeout: Duration::from_secs(1),
            },
        );

        let data_count = Arc::new(AtomicUsize::new(0));
        let control_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        let actor_data = data_count.clone();
        let actor_control = control_count.clone();
        let actor_mgmt = mgmt_count.clone();

        let actor_handle = thread::spawn(move || {
            let mut actor = CountingActor {
                data_count: actor_data,
                control_count: actor_control,
                mgmt_count: actor_mgmt,
            };
            rx.run(&mut actor);
        });

        // Send 100 of each type
        for i in 0..100 {
            tx.send(Message::Data(i)).unwrap();
            tx.send(Message::Control(())).unwrap();
            tx.send(Message::Management(())).unwrap();
        }

        // Give time for messages to queue
        thread::sleep(Duration::from_millis(50));

        // Shutdown - should drain all
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();

        // All messages should be processed
        assert_eq!(data_count.load(Ordering::Relaxed), 100);
        assert_eq!(control_count.load(Ordering::Relaxed), 100);
        assert_eq!(mgmt_count.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_shutdown_drain_all_timeout_fallback() {
        let (tx, mut rx) = ActorScheduler::new_with_shutdown_mode(
            10,   // Small burst limit to check shutdown frequently
            1000, // Large buffer to avoid blocking sends
            ShutdownMode::DrainAll {
                timeout: Duration::from_millis(200), // Increased timeout from 50ms to 200ms
            },
        );

        let data_count = Arc::new(AtomicUsize::new(0));
        let control_count = Arc::new(AtomicUsize::new(0));
        let mgmt_count = Arc::new(AtomicUsize::new(0));

        let actor_data = data_count.clone();
        let actor_control = control_count.clone();
        let actor_mgmt = mgmt_count.clone();

        // Slow actor that sleeps on each message
        struct SlowActor {
            data_count: Arc<AtomicUsize>,
            control_count: Arc<AtomicUsize>,
            mgmt_count: Arc<AtomicUsize>,
        }

        impl Actor<i32, (), ()> for SlowActor {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                thread::sleep(Duration::from_millis(1)); // Slow!
                self.data_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }

            fn handle_control(&mut self, _: ()) -> HandlerResult {
                self.control_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }

            fn handle_management(&mut self, _: ()) -> HandlerResult {
                self.mgmt_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }

            fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                match status {
                    SystemStatus::Idle => Ok(ActorStatus::Idle),
                    SystemStatus::Busy => Ok(ActorStatus::Busy),
                }
            }
        }

        let actor_handle = thread::spawn(move || {
            let mut actor = SlowActor {
                data_count: actor_data,
                control_count: actor_control,
                mgmt_count: actor_mgmt,
            };
            rx.run(&mut actor);
        });

        // Send 400 data messages (would take 400ms to process fully)
        for i in 0..400 {
            tx.send(Message::Data(i)).unwrap();
        }

        // Give actor time to start processing but not finish
        thread::sleep(Duration::from_millis(5));

        // Shutdown with 200ms timeout - should timeout before processing all 400
        let shutdown_start = std::time::Instant::now();
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();
        let shutdown_duration = shutdown_start.elapsed();

        // Shutdown should respect timeout (~200ms + overhead for normal run loop batch)
        assert!(
            shutdown_duration < Duration::from_millis(450),
            "Timeout should limit shutdown duration, took {:?}",
            shutdown_duration
        );

        // Should have processed SOME but definitely not all 400
        let processed = data_count.load(Ordering::Relaxed);
        assert!(
            processed < 400,
            "Timeout should prevent processing all messages, processed {}",
            processed
        );
        assert!(
            processed > 10,
            "Should process at least some messages before timeout"
        );
    }
}
