//! Priority Channel - A multi-priority message passing system
//!
//! This crate provides a message scheduler with three priority levels:
//! - **Control**: Highest priority, burst-limited to prevent starvation
//! - **Management**: Medium priority, burst-limited
//! - **Data**: Lowest priority, burst-limited with backpressure
//!
//! # Architecture
//!
//! The scheduler uses a "doorbell" pattern (see `doorbell.rs`) where:
//! 1. Senders publish into per-producer SPSC lanes, then ring the doorbell
//! 2. The receiver parks on the doorbell when idle; a ring (level) wakes it,
//!    a shutdown (latch) stops it
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
//! use actor_scheduler::{ActorScheduler, Message, Actor, ActorStatus, SystemStatus, HandlerResult, HandlerError};
//!
//! struct MyHandler;
//!
//! impl Actor<String, String, String> for MyHandler {
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
//!     fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> { Ok(ActorStatus::Idle) }
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

pub mod actors;
mod doorbell;
mod error;
pub mod host;
pub mod mealy;
mod params;
pub mod sharded;
pub mod spsc;

use error::DrainStatus;
pub use error::{HandlerError, HandlerResult, SendError};
pub use host::{Green, GreenSender, GreenThread, Host, HostOut, RunSweep, green_channel};
pub use params::SchedulerParams;
pub use spsc::TrySendError;

// Re-export macros from the proc-macro crate
pub use actor_scheduler_macros::{ports, troupe};

use doorbell::{Chime, Doorbell, Ring};
use sharded::{InboxBuilder, ShardedInbox};
use spsc::SpscSender;
use std::sync::Arc;
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
///   5. Call handle_os() - let actor/OS do other work
///   6. Repeat
/// ```
///
/// This provides best-effort priority with starvation protection:
/// - Control messages typically process before Data/Management
/// - All lanes are burst-limited to prevent monopolization
/// - No cross-lane ordering guarantees - only best-effort priority
/// - Protection against slow-loris attacks (poorly-behaved senders can't drown channels)
///
/// Exit immediately on shutdown, dropping all pending messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShutdownMode {
    #[default]
    Immediate,
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

/// Actor status returned from handle_os() to hint the scheduler about blocking behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorStatus {
    Idle, // Actor has no unfinished work. Scheduler can block. (0% CPU)
    Busy, // Actor has unfinished work (yielding). Scheduler should poll.
}

/// Status provided to the actor's handle_os method indicating the state of the scheduler's queues.
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
    /// Returns `Ok(())` on success. An `Err` panics the scheduler thread — there is no
    /// recoverable outcome to opt into.
    fn handle_data(&mut self, msg: D) -> HandlerResult;

    /// Handle a control message.
    ///
    /// Returns `Ok(())` on success. An `Err` panics the scheduler thread.
    fn handle_control(&mut self, msg: C) -> HandlerResult;

    /// Handle a management message.
    ///
    /// Returns `Ok(())` on success. An `Err` panics the scheduler thread.
    fn handle_management(&mut self, msg: M) -> HandlerResult;

    /// The "Hook" where the Actor creates the bridge to the OS.
    /// Called when the scheduler has drained available messages (or hit burst limits).
    ///
    /// Returns actor status: Busy if yielding with unfinished work, Idle if done.
    /// An `Err` panics the scheduler thread.
    fn handle_os(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError>;
}

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
/// type, enabling type-safe access to other actors in the group.
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
///     fn handle_os(&mut self, status: SystemStatus) -> ActorStatus { ActorStatus::Idle }
/// }
/// ```
pub trait TroupeActor<Dir>:
    Sized
    + ActorTypes
    + Actor<
        <Self as ActorTypes>::Data,
        <Self as ActorTypes>::Control,
        <Self as ActorTypes>::Management,
    >
{
    /// Create a new actor from its directory of handles.
    ///
    /// With SPSC channels, each actor OWNS its directory instance.
    /// The directory contains dedicated SPSC handles to every other actor.
    fn new(dir: Dir) -> Self;
}

/// Builder for multi-producer actor channels.
///
/// Each call to [`add_producer`](ActorBuilder::add_producer) creates a dedicated
/// SPSC channel per lane. Call [`build`](ActorBuilder::build) to seal the registry
/// and get the [`ActorScheduler`].
///
/// # Lifecycle
///
/// ```text
/// 1. ActorBuilder::new(buffer_size, waker)
/// 2. builder.add_producer()  → ActorHandle  (repeat N times)
/// 3. builder.build()         → ActorScheduler (seals registry)
/// ```
///
/// # Example
///
/// ```ignore
/// let mut builder = ActorBuilder::<Data, Control, Mgmt>::new(1024, None);
/// let handle_a = builder.add_producer();  // Actor A's dedicated channels
/// let handle_b = builder.add_producer();  // Actor B's dedicated channels
/// let mut scheduler = builder.build();    // Seals — no more producers
/// ```
pub struct ActorBuilder<D, C, M> {
    doorbell_ring: Ring,
    doorbell: Option<Doorbell>,
    data_inbox: InboxBuilder<D>,
    control_inbox: InboxBuilder<C>,
    mgmt_inbox: InboxBuilder<M>,
    wake_handler: Option<Arc<dyn WakeHandler>>,
    params: SchedulerParams,
}

impl<D, C, M> ActorBuilder<D, C, M> {
    /// Create a new builder with default scheduler parameters.
    ///
    /// # Arguments
    /// * `data_buffer_size` - Per-producer SPSC buffer size for the data lane
    /// * `wake_handler` - Optional platform wake handler (e.g., macOS Cocoa waker)
    #[must_use]
    pub fn new(data_buffer_size: usize, wake_handler: Option<Arc<dyn WakeHandler>>) -> Self {
        Self::new_with_params(data_buffer_size, wake_handler, SchedulerParams::DEFAULT)
    }

    /// Create a new builder with explicit tuning parameters.
    #[must_use]
    pub fn new_with_params(
        data_buffer_size: usize,
        wake_handler: Option<Arc<dyn WakeHandler>>,
        params: SchedulerParams,
    ) -> Self {
        assert!(
            data_buffer_size > 0,
            "data_buffer_size must be >= 1, got {}",
            data_buffer_size
        );
        params.validate();

        let (doorbell_ring, bell) = Doorbell::new();

        Self {
            doorbell_ring,
            doorbell: Some(bell),
            data_inbox: InboxBuilder::new(data_buffer_size),
            control_inbox: InboxBuilder::new(params.control_mgmt_buffer_size),
            mgmt_inbox: InboxBuilder::new(params.control_mgmt_buffer_size),
            wake_handler,
            params,
        }
    }

    /// Register a new producer. Returns a unique [`ActorHandle`] with dedicated
    /// SPSC channels to this actor's three priority lanes.
    ///
    /// Call this once per producer during initialization, before [`build`](Self::build).
    pub fn add_producer(&mut self) -> ActorHandle<D, C, M> {
        ActorHandle {
            doorbell: self.doorbell_ring.clone(),
            tx_data: self.data_inbox.add_producer(),
            tx_control: self.control_inbox.add_producer(),
            tx_mgmt: self.mgmt_inbox.add_producer(),
            wake_handler: self.wake_handler.clone(),
            params: self.params,
        }
    }

    /// Seal the registry and return the scheduler.
    ///
    /// Uses default burst limits from [`SchedulerParams`].
    /// No more producers can be added after this call.
    #[must_use]
    pub fn build(self) -> ActorScheduler<D, C, M> {
        let burst = self.params.default_data_burst_limit;
        self.build_with_burst(burst, ShutdownMode::default())
    }

    /// Seal the registry with explicit burst limit and shutdown mode.
    #[must_use]
    pub fn build_with_burst(
        self,
        data_burst_limit: usize,
        shutdown_mode: ShutdownMode,
    ) -> ActorScheduler<D, C, M> {
        ActorScheduler {
            doorbell: self.doorbell.expect("ActorBuilder::build called twice"),
            rx_data: self.data_inbox.build(),
            rx_control: self.control_inbox.build(),
            rx_mgmt: self.mgmt_inbox.build(),
            data_burst_limit,
            management_burst_limit: self.params.management_burst_limit(),
            control_burst_limit: self.params.control_burst_limit(),
            shutdown_mode,
        }
    }

    /// Seal the registry as a [`DedicatedThread`] instead of a classic [`ActorScheduler`].
    ///
    /// Same producer-facing contract as [`build`](Self::build)/[`build_with_burst`](Self::build_with_burst)
    /// — every [`ActorHandle`] already handed out by [`add_producer`](Self::add_producer) keeps
    /// working unchanged, because it feeds the same three sharded lanes either way. What
    /// differs is the consumer: a [`Transducer`](mealy::Transducer) plus its
    /// [`Wiring`](mealy::Wiring) instead of an [`Actor`] impl, run through [`Node`](mealy::Node)'s
    /// dispatch discipline instead of `handle_data`/`handle_control`/`handle_management`.
    ///
    /// This is the load-bearing bridge for placement-as-a-driver-choice (design doc §5): the
    /// same lanes an `ActorHandle` feeds can be read by either an `ActorScheduler` (this
    /// module) or a `Node` (`mealy.rs`), because [`sharded::ShardedInbox`] implements both
    /// `ShardedInbox::drain` (for the former) and [`mealy::Inbox`] (for the latter).
    #[must_use]
    pub fn build_dedicated_thread<T, W>(
        self,
        actor: T,
        wiring: W,
    ) -> DedicatedThread<T, W, ShardedInbox<D>, ShardedInbox<C>, ShardedInbox<M>>
    where
        T: mealy::Transducer<Data = D, Control = C, Management = M>,
        W: mealy::Wiring<Out = T::Out>,
    {
        // One doorbell visit's worth of lane work, mirroring the classic scheduler's
        // per-wake burst budget (two control half-bursts + management + data).
        let sweep_burst = (self.params.control_burst_limit()
            + self.params.management_burst_limit()
            + self.params.default_data_burst_limit)
            .max(1);
        let node = mealy::Node::new_with_lanes(
            actor,
            mealy::Lanes {
                control: self.control_inbox.build(),
                management: self.mgmt_inbox.build(),
                data: self.data_inbox.build(),
            },
            wiring,
            self.params,
        );
        DedicatedThread {
            node,
            sweep_burst,
            doorbell: self
                .doorbell
                .expect("ActorBuilder::build_dedicated_thread called twice"),
        }
    }
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

/// Fibonacci hash constant for jitter calculation.
const JITTER_HASH_CONSTANT: u64 = 0x9e3779b97f4a7c15;

/// Mixes the attempt number into the wall-clock nanoseconds before hashing,
/// so consecutive attempts on the same thread still land in different buckets.
const ATTEMPT_MIX_CONSTANT: u64 = 0x517cc1b727220a95;

/// Calculate exponential backoff with jitter.
///
/// Uses a simple exponential backoff strategy with added jitter to prevent
/// thundering herd problems when multiple actors wake simultaneously.
fn backoff_with_jitter(attempt: u32, params: &SchedulerParams) -> Result<Duration, SendError> {
    let base_micros = params.min_backoff.as_micros() as u64;
    let max_micros = params.max_backoff.as_micros() as u64;

    let multiplier = 2u64.saturating_pow(attempt);
    let backoff_micros = base_micros.saturating_mul(multiplier);
    if backoff_micros > max_micros {
        return Err(SendError::Timeout);
    }

    // Add jitter: random value between [min_pct%, (min_pct+range_pct)%] of backoff
    // Use wall clock time for actual randomness (prevents thundering herd)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0));

    // Mix nanoseconds with attempt number for better distribution across threads
    let hash = (now.as_nanos() as u64 ^ (attempt as u64).wrapping_mul(ATTEMPT_MIX_CONSTANT))
        .wrapping_mul(JITTER_HASH_CONSTANT);

    let jitter_pct = params.jitter_min_pct + (hash % params.jitter_range_pct);
    let jittered_micros = (backoff_micros * jitter_pct) / 100;

    Ok(Duration::from_micros(jittered_micros))
}

/// A unified sender handle that routes messages to the scheduler with priority lanes.
///
/// Each handle owns dedicated SPSC channels (one per lane) to the target actor.
/// Not `Clone` — use [`ActorBuilder::add_producer`] to create additional handles.
/// This eliminates all send-side contention: each producer gets its own wait-free path.
pub struct ActorHandle<D, C, M> {
    // Doorbell producer end - wake (level) and shutdown (latch) signals
    doorbell: Ring,
    // Each lane is a dedicated SPSC channel (one producer per handle)
    tx_data: SpscSender<D>,
    tx_control: SpscSender<C>,
    tx_mgmt: SpscSender<M>,
    // Optional custom wake handler for platform-specific wake mechanisms
    wake_handler: Option<Arc<dyn WakeHandler>>,
    // Tunable parameters for backoff/retry behavior
    params: SchedulerParams,
}

// Manual Debug implementation - wake_handler is opaque (trait object)
impl<D, C, M> std::fmt::Debug for ActorHandle<D, C, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActorHandle")
            .field("has_wake_handler", &self.wake_handler.is_some())
            .finish_non_exhaustive()
    }
}

/// Send with retry and exponential backoff + jitter for fairness.
///
/// Backoff strategy:
/// 1. Spin (immediate retry) for first `params.spin_attempts`
/// 2. Yield (cooperative) for next `params.yield_attempts`
/// 3. Sleep (blocking) with exponential backoff for remaining attempts
///
/// Used for control and management lanes to prevent thundering herd when
/// multiple senders compete for buffer space.
pub(crate) fn send_with_backoff<T>(
    tx: &SpscSender<T>,
    mut msg: T,
    params: &SchedulerParams,
) -> Result<(), SendError> {
    let mut attempt = 0u32;
    loop {
        match tx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(spsc::TrySendError::Full(returned_msg)) => {
                // Restore message for retry
                msg = returned_msg;

                // Backoff strategy: spin → yield → sleep
                if attempt < params.spin_attempts {
                    // Phase 1: Spin (immediate retry, hot loop)
                    // No sleep/yield - just retry immediately
                } else if attempt < params.spin_attempts + params.yield_attempts {
                    // Phase 2: Yield (cooperative, let other threads run)
                    std::thread::yield_now();
                } else {
                    // Phase 3: Sleep (exponential backoff with jitter)
                    #[cfg(debug_assertions)]
                    if attempt.is_multiple_of(10) {
                        eprintln!(
                            "[ActorScheduler] Priority channel full, backing off (attempt {})",
                            attempt
                        );
                    }

                    let sleep_attempt = attempt - (params.spin_attempts + params.yield_attempts);
                    let backoff = backoff_with_jitter(sleep_attempt, params)?;
                    std::thread::sleep(backoff);
                }

                attempt = attempt.saturating_add(1);
            }
            Err(spsc::TrySendError::Disconnected(_)) => {
                return Err(SendError::Disconnected);
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
            Message::Data(mut d) => {
                // Data lane: spin-yield until space available (backpressure)
                loop {
                    match self.tx_data.try_send(d) {
                        Ok(()) => break,
                        Err(spsc::TrySendError::Full(returned_d)) => {
                            d = returned_d;
                            std::thread::yield_now();
                        }
                        Err(spsc::TrySendError::Disconnected(_)) => {
                            return Err(SendError::Disconnected);
                        }
                    }
                }
                self.wake();
            }
            Message::Control(ctrl_msg) => {
                // Control lane: retry with backoff for fairness
                send_with_backoff(&self.tx_control, ctrl_msg, &self.params)?;
                self.wake();
            }
            Message::Management(m) => {
                // Management lane: retry with backoff for fairness
                send_with_backoff(&self.tx_mgmt, m, &self.params)?;
                self.wake();
            }
            Message::Shutdown => {
                // Shutdown is a sticky latch on the doorbell, so delivery is guaranteed by
                // construction — no blocking send, and no wake-on-both-sides dance to keep a
                // capacity-1 slot drainable. Latch first, THEN fire the platform waker: a
                // consumer blocked inside an OS wait (`step_os`/`handle_os`) that the waker
                // interrupts must find the latch already set, or it would re-enter the wait
                // with nothing left to re-wake it.
                if !self.doorbell.shutdown() {
                    return Err(SendError::Disconnected);
                }
                if let Some(waker) = &self.wake_handler {
                    waker.wake();
                }
            }
        };
        Ok(())
    }

    /// Non-blocking send: the mechanism `mealy::send_port` uses to deliver a port addressed to
    /// this handle (design doc §3.2). Tries the message's lane exactly once — no spin, no
    /// backoff — then rings the doorbell on success via the same [`Self::wake`] path
    /// [`Self::send`] uses. On `Full` or `Disconnected` the message is handed back inside the
    /// error, unsent.
    ///
    /// `pub(crate)`, not public: **the scheduler owns send policy, a caller names a target.**
    /// `mealy::send_port` (via [`mealy::PortTarget`](crate::mealy::PortTarget)) is the one public
    /// path to this handle from a `Wiring::flush` — it is what parks the payload on refusal
    /// rather than leaving the choice (drop, park, panic) to whoever called `try_send`. Flush
    /// runs inside `Node::poll` inside a `Host::sweep`, on a thread shared by every green actor
    /// that host owns — blocking there freezes all of them, which is why a green actor may not
    /// block at all and this cannot be [`Self::send`]. A handler pushing into another actor's
    /// inbox from outside a flush should reach for [`Self::send`] instead: its bounded backoff
    /// stops only the caller, and a loud timeout beats a silently dropped or spun-on message.
    ///
    /// # Errors
    /// Returns [`TrySendError::Full`] if the target lane's ring is full, or
    /// [`TrySendError::Disconnected`] if the receiver is gone. Either way the message comes
    /// back rewrapped in its original [`Message`] variant.
    pub(crate) fn try_send<T: Into<Message<D, C, M>>>(
        &self,
        msg: T,
    ) -> Result<(), spsc::TrySendError<Message<D, C, M>>> {
        use spsc::TrySendError;

        match msg.into() {
            Message::Data(d) => match self.tx_data.try_send(d) {
                Ok(()) => {
                    self.wake();
                    Ok(())
                }
                Err(TrySendError::Full(d)) => Err(TrySendError::Full(Message::Data(d))),
                Err(TrySendError::Disconnected(d)) => {
                    Err(TrySendError::Disconnected(Message::Data(d)))
                }
            },
            Message::Control(c) => match self.tx_control.try_send(c) {
                Ok(()) => {
                    self.wake();
                    Ok(())
                }
                Err(TrySendError::Full(c)) => Err(TrySendError::Full(Message::Control(c))),
                Err(TrySendError::Disconnected(c)) => {
                    Err(TrySendError::Disconnected(Message::Control(c)))
                }
            },
            Message::Management(m) => match self.tx_mgmt.try_send(m) {
                Ok(()) => {
                    self.wake();
                    Ok(())
                }
                Err(TrySendError::Full(m)) => Err(TrySendError::Full(Message::Management(m))),
                Err(TrySendError::Disconnected(m)) => {
                    Err(TrySendError::Disconnected(Message::Management(m)))
                }
            },
            // Shutdown travels on the doorbell, not a lane ring, so there is no `Full` of its
            // own to report non-blockingly — forward to the same delivery `send` uses and
            // report success, matching the spec's carve-out for the one variant with no ring
            // to be full.
            Message::Shutdown => {
                // Always reported as delivered: shutdown is a sticky latch that never blocks,
                // and `send`'s own `Message::Shutdown` arm only ever fails when the doorbell
                // is abandoned — meaning the scheduler is already gone, which is not this
                // call's problem to report a second way.
                match self.send_message(Message::Shutdown) {
                    Ok(()) | Err(_) => {}
                }
                Ok(())
            }
        }
    }

    /// Wake the scheduler to process messages.
    ///
    /// Rings the doorbell first — its RMW is what orders the lane publish
    /// before the scheduler's sleep/wake decision (see `doorbell.rs`; no fence
    /// needed) and coalescing is inherent, a set bit stays one bit. Then the
    /// custom wake handler, if any, wakes the platform event loop (e.g.,
    /// sending NSEvent on macOS): state first, mechanisms second, so a
    /// consumer the mechanism rouses always finds the state already set.
    fn wake(&self) {
        if !self.doorbell.ring() {
            panic!("Doorbell abandoned - scheduler dropped unexpectedly");
        }
        if let Some(waker) = &self.wake_handler {
            waker.wake();
        }
    }
}

/// The port sender's view of an `ActorHandle`: a `Wiring::flush` builds the `Message` itself
/// (it already knows which lane a port belongs on) and hands it to `mealy::send_port`, rather
/// than this crate reconstructing an arbitrary port type from a `Message` on the retry path —
/// see `mealy::send_port`'s doc for why that direction of conversion is the one worth avoiding.
impl<D, C, M> mealy::PortTarget<Message<D, C, M>> for ActorHandle<D, C, M> {
    fn try_deliver(
        &self,
        msg: Message<D, C, M>,
    ) -> Result<(), spsc::TrySendError<Message<D, C, M>>> {
        self.try_send(msg)
    }
}

/// Rings an actor's doorbell without sending it a message.
///
/// The scheduler blocks on its doorbell when idle, so anything that makes an actor runnable
/// *without* going through its lanes has to ring that bell itself. The green tier is the
/// case that needs it: a producer pushes straight into a green actor's inbox
/// ([`GreenSender`](crate::host::GreenSender)) and then wakes the host that owns it.
///
/// This is the same contract as a `Waker` in a futures runtime — "there is work for you now"
/// — and it is deliberately *not* `ActorHandle::send`: a wake carries no payload and cannot
/// back up, because the doorbell is a level, not a queue — asserting it twice is one bit.
///
/// # Ordering
///
/// **Make the work visible, then wake.** Waking first admits a lost wakeup: the host can
/// wake, find nothing, and go back to sleep before the message lands.
#[derive(Clone)]
pub struct Waker {
    doorbell: Ring,
    wake_handler: Option<Arc<dyn WakeHandler>>,
}

impl Waker {
    /// Signal the actor that it has work.
    ///
    /// Never blocks and never fails. The doorbell is a level, so repeated wakes coalesce
    /// into one bit; the ring's RMW orders the caller's inbox push before the scheduler's
    /// sleep/wake decision (see `doorbell.rs`), so no fence is needed here. Unlike
    /// `ActorHandle::wake`, a waker outliving its scheduler is ordinary — a green actor
    /// can be fed after its host is gone. There is nobody to wake, so an abandoned
    /// doorbell is ignored rather than a panic.
    pub fn wake(&self) {
        let _ = self.doorbell.ring();
        if let Some(waker) = &self.wake_handler {
            waker.wake();
        }
    }
}

impl std::fmt::Debug for Waker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Waker")
            .field("has_wake_handler", &self.wake_handler.is_some())
            .finish_non_exhaustive()
    }
}

impl<D, C, M> ActorHandle<D, C, M> {
    /// A [`Waker`] for this actor's scheduler.
    ///
    /// Hand one to anything that can make this actor runnable without sending it a message.
    #[must_use]
    pub fn waker(&self) -> Waker {
        Waker {
            doorbell: self.doorbell.clone(),
            wake_handler: self.wake_handler.clone(),
        }
    }
}

/// The receiver side that implements the priority scheduling logic.
///
/// Internally uses [`ShardedInbox`] per lane: each registered producer has
/// a dedicated SPSC ring buffer, and the scheduler drains all shards with
/// round-robin fairness. The MPSC doorbell channel is kept for wake/shutdown signals.
pub struct ActorScheduler<D, C, M> {
    doorbell: Doorbell, // Wake (level) and shutdown (latch) signals
    rx_data: ShardedInbox<D>,
    rx_control: ShardedInbox<C>,
    rx_mgmt: ShardedInbox<M>,
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
        // No fence needed before the drains: the doorbell poll/wait that led
        // here is an RMW, which both orders every publish made before the
        // consumed ring() (see doorbell.rs) and — via the sleep-commit RMW —
        // makes it impossible to block while a published message is stranded.

        // Drain Control → Mgmt → Control → Data
        // Control budget is split evenly between the two control runs to prevent double priority.
        // Floor at 1: integer halving would otherwise zero the budget when the
        // limit is 1, and a zero-budget drain can never make progress.
        let half_control = (self.control_burst_limit / 2).max(1);

        let control1 = self
            .rx_control
            .drain(half_control, |msg| actor.handle_control(msg))?;

        let mgmt = self.rx_mgmt.drain(self.management_burst_limit, |msg| {
            actor.handle_management(msg)
        })?;

        let control2 = self
            .rx_control
            .drain(half_control, |msg| actor.handle_control(msg))?;

        let data = self
            .rx_data
            .drain(self.data_burst_limit, |msg| actor.handle_data(msg))?;

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

        let returned_hint = actor.handle_os(system_status)?;

        let status = if more_work || returned_hint == ActorStatus::Busy {
            SchedulerLoopStatus::Working
        } else {
            SchedulerLoopStatus::Idle
        };

        Ok(Some(status))
    }

    #[cold]
    fn handle_shutdown<A>(&mut self, _actor: &mut A) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        match self.shutdown_mode {
            ShutdownMode::Immediate => Ok(()),
        }
    }

    /// Create a new scheduler with a single producer.
    ///
    /// Convenience method for the common case of one sender. Returns
    /// `(handle, scheduler)`. For multiple producers, use [`ActorBuilder`].
    ///
    /// # Arguments
    /// * `data_burst_limit` - Maximum data messages to process per wake cycle
    /// * `data_buffer_size` - Size of bounded data buffer (backpressure threshold).
    ///
    /// # Panics
    /// Panics if `data_buffer_size` is 0.
    #[must_use]
    pub fn new(data_burst_limit: usize, data_buffer_size: usize) -> (ActorHandle<D, C, M>, Self) {
        Self::new_with_params(data_burst_limit, data_buffer_size, SchedulerParams::DEFAULT)
    }

    /// Create a new scheduler with explicit tuning parameters and a single producer.
    #[must_use]
    pub fn new_with_params(
        data_burst_limit: usize,
        data_buffer_size: usize,
        params: SchedulerParams,
    ) -> (ActorHandle<D, C, M>, Self) {
        let mut builder = ActorBuilder::new_with_params(data_buffer_size, None, params);
        let handle = builder.add_producer();
        let scheduler = builder.build_with_burst(data_burst_limit, ShutdownMode::default());
        (handle, scheduler)
    }

    /// Create a new scheduler with a custom wake handler and a single producer.
    #[must_use]
    pub fn new_with_wake_handler(
        data_burst_limit: usize,
        data_buffer_size: usize,
        wake_handler: Option<Arc<dyn WakeHandler>>,
    ) -> (ActorHandle<D, C, M>, Self) {
        let mut builder = ActorBuilder::new(data_buffer_size, wake_handler);
        let handle = builder.add_producer();
        let scheduler = builder.build_with_burst(data_burst_limit, ShutdownMode::default());
        (handle, scheduler)
    }

    /// The main scheduler loop.
    ///
    /// Blocks on the doorbell channel. Drains priority lanes in order:
    /// Shutdown > Control > Management > Data.
    ///
    /// Returns when `Message::Shutdown` is received or every sender handle is dropped. A
    /// handler `Err` panics the scheduler thread instead of returning — every build profile
    /// sets `panic = "abort"`, so there is no recoverable outcome to report back to a caller;
    /// fail fast, fail loudly.
    pub fn run<A>(&mut self, actor: &mut A)
    where
        A: Actor<D, C, M>,
    {
        if let Err(e) = self.run_inner(actor) {
            e.panic();
        }
    }

    /// Single non-blocking drain cycle, for driving an actor synchronously
    /// without a dedicated thread.
    ///
    /// Cooperative multiplexing of multiple actors on one thread is the green
    /// tier's job (`Host` in `host.rs`; see
    /// `docs/designs/actor-scheduler-mealy-transducer.md`), not this method's.
    /// Its remaining real callers are test fixtures that need to step an
    /// actor's message loop by hand; it is expected to leave the public API
    /// once those callers migrate to `Host`.
    ///
    /// Unlike [`run`], `poll_once()` never blocks:
    /// - If the doorbell is empty, it still attempts one drain pass (the actor
    ///   may have work from a previous `Working` state).
    /// - Returns `true` once the actor is finished and must not be polled again; `false` to
    ///   keep polling. A handler `Err` panics, same as [`run`].
    ///
    /// # Caller responsibility
    ///
    /// The caller must continue calling `poll_once()` after a `Disconnected`
    /// doorbell until it returns `true` — buffered SPSC messages need draining.
    pub fn poll_once<A>(&mut self, actor: &mut A) -> bool
    where
        A: Actor<D, C, M>,
    {
        match self.doorbell.poll() {
            Chime::Shutdown => {
                if let Err(e) = self.handle_shutdown(actor) {
                    e.panic();
                }
                true
            }

            Chime::Work | Chime::Quiet => match self.handle_wake(actor) {
                Ok(Some(_)) => false, // still running
                Ok(None) => true,     // all disconnected
                Err(e) => e.panic(),
            },

            Chime::Orphaned => {
                // All handles dropped — drain one batch, report done when empty
                match self.handle_wake(actor) {
                    Ok(Some(_)) => false, // more buffered work; caller polls again
                    Ok(None) => true,
                    Err(e) => e.panic(),
                }
            }
        }
    }

    fn run_inner<A>(&mut self, actor: &mut A) -> Result<(), HandlerError>
    where
        A: Actor<D, C, M>,
    {
        let mut working = false;

        loop {
            let chime = if working {
                self.doorbell.poll()
            } else {
                self.doorbell.wait()
            };

            match chime {
                Chime::Shutdown => {
                    self.handle_shutdown(actor)?;
                    return Ok(());
                }
                Chime::Work | Chime::Quiet => {
                    match self.handle_wake(actor)? {
                        Some(status) => {
                            working = matches!(status, SchedulerLoopStatus::Working);
                        }
                        None => return Ok(()), // All channels disconnected
                    }
                }
                Chime::Orphaned => {
                    // Doorbell orphaned — all handles and wakers dropped.
                    // SPSC shards may still have buffered messages.
                    // Drain until all shards report Disconnected.
                    loop {
                        match self.handle_wake(actor)? {
                            Some(_) => {} // keep draining
                            None => return Ok(()),
                        }
                    }
                }
            }
        }
    }
}

/// Runs one [`mealy::Node`] on a dedicated OS thread, blocking on a doorbell.
///
/// The mealy-tier mirror of [`host::GreenThread`]: same doorbell, same wake/shutdown vocabulary,
/// same "stay awake rather than risk a lost wakeup" reasoning — but driving one actor's own
/// thread instead of hosting many inside someone else's `handle_os`. Lives here rather than in
/// `host.rs` so the doorbell-loop discipline it must mirror ([`ActorScheduler::run_inner`])
/// stays on the same page as the mirror.
///
/// Built by [`ActorBuilder::build_dedicated_thread`], never directly: the doorbell receiver
/// and the three sharded lanes must come from the same builder as the [`ActorHandle`]s that
/// feed them, and `ActorBuilder` is the only thing that owns that invariant.
pub struct DedicatedThread<
    T,
    W,
    RD,
    RC = mealy::NoLane<<T as mealy::Transducer>::Control>,
    RM = mealy::NoLane<<T as mealy::Transducer>::Management>,
> where
    T: mealy::Transducer,
    W: mealy::Wiring<Out = T::Out>,
{
    node: mealy::Node<T, W, RD, RC, RM>,
    /// Lane polls allowed per sweep before the driver revisits the doorbell — the same role
    /// as the classic scheduler's per-wake burst limits. Without it, a continuously-fed lane
    /// would keep the sweep loop from ever seeing a latched `Shutdown` or giving `step_os` a
    /// turn.
    sweep_burst: usize,
    doorbell: Doorbell,
}

impl<T, W, RD, RC, RM> DedicatedThread<T, W, RD, RC, RM>
where
    T: mealy::Transducer,
    W: mealy::Wiring<Out = T::Out>,
    RD: mealy::Inbox<Item = T::Data>,
    RC: mealy::Inbox<Item = T::Control>,
    RM: mealy::Inbox<Item = T::Management>,
{
    /// Drain a bounded burst of lane work, then give `step_os` a turn. One "sweep" of the
    /// doorbell loop below.
    ///
    /// Draining is repeated polls — `Node::poll`'s own Control/Management/Data cycle already
    /// rotates across calls — but bounded by `sweep_burst`, for the same reason the classic
    /// scheduler bounds each wake's drain: an unbounded "until quiet" loop under a
    /// continuously-fed lane never revisits the doorbell, so a queued `Shutdown` is never
    /// observed and `step_os` never runs. Budget spent with work remaining is just `Busy`.
    ///
    /// Lane completion (`Halted` — every lane disconnected) is *not* terminal here: an OS
    /// bridge's external source outlives the last `ActorHandle` by design, and a retained
    /// [`Waker`] can still ring the doorbell long after every lane is dead. The sweep only
    /// reports whether there is work; *terminality belongs to the doorbell* — [`run`] completes
    /// when the doorbell disconnects (no handle and no waker left anywhere) and a sweep finds
    /// nothing to do.
    ///
    /// A wiring target that is gone panics inside `Node` itself (`mealy.rs`) the instant a
    /// flush discovers it — there is no supervisor for this driver to hand the problem to, so
    /// failing fast at the point of discovery is the whole policy.
    ///
    /// Returns the busy hint for the doorbell loop's `working` flag.
    fn sweep(&mut self) -> bool {
        let mut polls = 0usize;
        let lane_step = loop {
            match self.node.poll() {
                mealy::Step::Ran => {
                    polls += 1;
                    if polls >= self.sweep_burst {
                        break mealy::Step::Ran;
                    }
                }
                other => break other,
            }
        };
        let lanes_completed = matches!(lane_step, mealy::Step::Halted);

        // Idle means the lap found every lane genuinely empty. Ran means the budget ran out
        // with lane work remaining; Blocked means a parked outbox is deferring lane work
        // rather than lacking it — both are the `Busy` the classic scheduler reports to
        // `handle_os` in the same positions.
        let system_status = if lane_step == mealy::Step::Idle || lanes_completed {
            SystemStatus::Idle
        } else {
            SystemStatus::Busy
        };

        let (os_step, os_actor_status) = self.node.poll_os(system_status);
        let os_busy =
            matches!(os_actor_status, ActorStatus::Busy) || os_step == mealy::Step::Blocked;

        let lanes_busy = !lanes_completed && lane_step != mealy::Step::Idle;
        lanes_busy || os_busy
    }

    /// Run this node until it halts or is shut down.
    ///
    /// Mirrors [`ActorScheduler::run_inner`]'s doorbell loop exactly: block on the doorbell
    /// unless the last sweep left work behind, in which case poll it non-blockingly instead
    /// (the same `working` flag). `Message::Shutdown` still stops it immediately, because
    /// shutdown travels the doorbell rather than a lane, same as every other actor here.
    ///
    /// A `Blocked` sweep is reported busy rather than idle — a parked outbox resolves by a
    /// consumer draining, which sleeping on this doorbell cannot wait for (ring-not-full
    /// waking — a consumer ringing this thread's doorbell the instant it drains — is not wired
    /// yet), and there is no risk of *missing* a wake this way, only of spending CPU until it
    /// clears. A gone wiring target panics inside `Node` rather than reaching this loop at all.
    ///
    /// Completion is the doorbell's to declare: it returns on `Shutdown`, or when the doorbell
    /// disconnects — meaning no `ActorHandle` *and* no [`Waker`] exists anywhere, so nothing can
    /// ever wake this node again — and a final drain finds nothing left. Dead lanes alone are
    /// not completion: an idle OS bridge whose waker is still held somewhere sleeps at 0% CPU
    /// instead of exiting, because that waker is someone's declared intent to come back.
    pub fn run(&mut self) {
        let mut working = false;

        loop {
            let chime = if working {
                self.doorbell.poll()
            } else {
                self.doorbell.wait()
            };

            match chime {
                Chime::Shutdown => return,
                Chime::Work | Chime::Quiet => working = self.sweep(),
                Chime::Orphaned => {
                    // All handles and wakers dropped. Keep sweeping — buffered shard
                    // messages, a parked outbox, or step_os's own work can all outlive the
                    // last `ActorHandle` — until a sweep reports genuinely nothing left to do.
                    while self.sweep() {}
                    return;
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

        fn handle_os(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    // Regression: control_burst_limit of 1 used to halve to a zero drain
    // budget (1 / 2 == 0), so the control lane could never make progress —
    // and a zero-limit drain misreported the lane as Disconnected.
    #[test]
    fn control_lane_progresses_with_burst_limit_one() {
        let params = SchedulerParams {
            control_mgmt_buffer_size: 1,
            control_burst_multiplier: 1, // control_burst_limit == 1
            ..SchedulerParams::DEFAULT
        };
        assert_eq!(params.control_burst_limit(), 1);

        let (tx, mut rx) = ActorScheduler::<String, String, String>::new_with_params(4, 4, params);
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        let scheduler = thread::spawn(move || {
            let mut handler = TestHandler { log: log_clone };
            rx.run(&mut handler);
        });

        tx.send(Message::Control("ping".to_string())).unwrap();

        // Must be observed BEFORE shutdown: shutdown-time draining would mask
        // a broken steady-state control path.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let processed = loop {
            if log.lock().unwrap().contains(&"Ctrl: ping".to_string()) {
                break true;
            }
            if std::time::Instant::now() >= deadline {
                break false;
            }
            thread::sleep(Duration::from_millis(5));
        };

        tx.send(Message::Shutdown).unwrap();
        scheduler.join().unwrap();

        assert!(
            processed,
            "control message was never processed with control_burst_limit == 1"
        );
    }

    // Regression: with a data burst limit of 0, the scheduler must neither
    // spin forever (drain(0) reporting More every cycle) nor exit early by
    // misreporting the lane as Disconnected and dropping queued messages.
    // The drain clamps its budget to 1, so it makes progress and terminates.
    #[test]
    fn zero_data_burst_limit_processes_messages_and_exits() {
        let (tx, mut rx) = ActorScheduler::<String, String, String>::new(0, 10);
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        let scheduler = thread::spawn(move || {
            let mut handler = TestHandler { log: log_clone };
            rx.run(&mut handler);
        });

        tx.send(Message::Data("a".to_string())).unwrap();
        tx.send(Message::Data("b".to_string())).unwrap();
        drop(tx);

        // Bounded join: the scheduler must exit once all handles are gone.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !scheduler.is_finished() && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            scheduler.is_finished(),
            "scheduler must terminate with a zero data burst limit"
        );
        scheduler.join().unwrap();

        let log = log.lock().unwrap();
        assert!(
            log.contains(&"Data: a".to_string()) && log.contains(&"Data: b".to_string()),
            "queued messages must not be dropped on exit; got {:?}",
            *log
        );
    }

    #[test]
    fn verify_data_lane_backpressure_contract() {
        let (tx, mut rx) = ActorScheduler::new(2, 1);
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        thread::spawn(move || {
            let mut handler = TestHandler { log: log_clone };
            rx.run(&mut handler);
        });

        // Send from this thread — the 3rd message may spin-yield on backpressure
        let send_thread = thread::spawn(move || {
            tx.send(Message::Data("1".to_string())).unwrap();
            tx.send(Message::Data("2".to_string())).unwrap();
            tx.send(Message::Data("3".to_string())).unwrap();
        });

        send_thread.join().unwrap();
        thread::sleep(Duration::from_millis(100));
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

        impl Actor<i32, String, bool> for CountingHandler {
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
            fn handle_os(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
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
                fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
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
mod poll_once_tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    struct CountActor {
        data: usize,
        ctrl: usize,
    }

    impl Actor<i32, i32, i32> for CountActor {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            self.data += 1;
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            self.ctrl += 1;
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    #[test]
    fn poll_once_returns_false_when_still_running() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        let mut actor = CountActor { data: 0, ctrl: 0 };

        // No messages yet — doorbell empty, poll_once drains nothing, keeps running
        let result = rx.poll_once(&mut actor);
        assert!(!result);
        drop(tx);
    }

    #[test]
    fn poll_once_drains_messages_and_keeps_running() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(100, 100);
        let mut actor = CountActor { data: 0, ctrl: 0 };

        tx.send(Message::Control(1)).unwrap();
        tx.send(Message::Data(2)).unwrap();

        // Give messages time to arrive
        thread::sleep(Duration::from_millis(5));

        let result = rx.poll_once(&mut actor);
        assert!(!result); // still connected
        assert!(actor.ctrl >= 1, "control message should have been drained");

        drop(tx);
    }

    #[test]
    fn poll_once_returns_true_after_all_handles_dropped() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        let mut actor = CountActor { data: 0, ctrl: 0 };

        drop(tx);

        // Keep polling until done
        while !rx.poll_once(&mut actor) {}
    }

    #[test]
    #[should_panic(expected = "injected failure")]
    fn poll_once_panics_on_a_handler_error() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);

        struct FailOnData;
        impl Actor<i32, i32, i32> for FailOnData {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                Err(HandlerError::new("injected failure"))
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }

        tx.send(Message::Data(1)).unwrap();
        thread::sleep(Duration::from_millis(5));

        let mut actor = FailOnData;
        while !rx.poll_once(&mut actor) {}
    }

    #[test]
    fn poll_once_returns_true_on_shutdown_message() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        let mut actor = CountActor { data: 0, ctrl: 0 };

        tx.send(Message::Shutdown).unwrap();
        thread::sleep(Duration::from_millis(5));

        while !rx.poll_once(&mut actor) {}
    }
}

#[cfg(test)]
mod try_send_tests {
    use super::*;
    use crate::mealy::Flush;

    struct RecordActor {
        got: Option<i32>,
    }

    impl Actor<i32, i32, i32> for RecordActor {
        fn handle_data(&mut self, msg: i32) -> HandlerResult {
            self.got = Some(msg);
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    #[test]
    fn try_send_succeeds_and_is_received() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        let mut port = Some(Message::Data(7));
        assert_eq!(mealy::send_port(&mut port, &tx), Flush::Done);
        assert!(port.is_none(), "delivered ports are cleared");

        let mut actor = RecordActor { got: None };
        assert!(!rx.poll_once(&mut actor), "still connected");
        assert_eq!(actor.got, Some(7));
    }

    #[test]
    fn try_send_on_a_full_data_ring_returns_full_with_the_message_recoverable() {
        let (tx, _rx) = ActorScheduler::<i32, i32, i32>::new(10, 2);
        // Capacity rounds up to a power of 2 (minimum 2); fill it without a consumer draining.
        loop {
            let mut port = Some(Message::Data(1));
            if mealy::send_port(&mut port, &tx) != Flush::Done {
                break;
            }
        }

        let mut port = Some(Message::Data(99));
        assert_eq!(mealy::send_port(&mut port, &tx), Flush::Blocked);
        match port {
            Some(Message::Data(msg)) => assert_eq!(msg, 99),
            other => panic!("expected a blocked port to keep Data(99), got {other:?}"),
        }
    }

    #[test]
    fn try_send_after_receiver_drop_returns_disconnected() {
        let (tx, rx) = ActorScheduler::<i32, i32, i32>::new(10, 100);
        drop(rx);

        let mut port = Some(Message::Data(5));
        assert_eq!(mealy::send_port(&mut port, &tx), Flush::Disconnected);
        match port {
            Some(Message::Data(msg)) => assert_eq!(msg, 5),
            other => panic!("expected a disconnected target to retain Data(5), got {other:?}"),
        }
    }
}

// Tests targeting missed mutations in backoff_with_jitter and send_with_backoff.
// These functions are private so tests live in the same module.
#[cfg(test)]
mod backoff_unit_tests {
    use super::*;
    use std::time::Duration;

    fn params_with_bounds(min_us: u64, max_us: u64) -> SchedulerParams {
        SchedulerParams {
            min_backoff: Duration::from_micros(min_us),
            max_backoff: Duration::from_micros(max_us),
            jitter_min_pct: 50,
            jitter_range_pct: 49,
            ..SchedulerParams::DEFAULT
        }
    }

    // Kills: replace > with == (would only timeout at exact max, not above)
    // Kills: replace > with < (would timeout when still under max)
    #[test]
    fn backoff_returns_timeout_when_over_max() {
        // min=100us, max=1000us. attempt=4: backoff = 100 * 2^4 = 1600 > 1000 → Timeout
        let params = params_with_bounds(100, 1000);
        assert!(
            matches!(backoff_with_jitter(4, &params), Err(SendError::Timeout)),
            "Should return Timeout when backoff_micros > max_micros"
        );
    }

    // Kills: replace > with >= (>= fires at equality; > should NOT fire at equality)
    #[test]
    fn backoff_returns_ok_at_exact_max() {
        // min=max=100us. attempt=0: backoff = 100 * 1 = 100, max = 100.
        // `100 > 100` is false → should return Ok, not Timeout.
        let params = params_with_bounds(100, 100);
        assert!(
            backoff_with_jitter(0, &params).is_ok(),
            "backoff == max should NOT trigger timeout (> not >=)"
        );
    }

    #[test]
    fn backoff_returns_ok_when_under_max() {
        // min=100us, max=10000us. attempt=0: backoff=100 < 10000 → Ok
        let params = params_with_bounds(100, 10_000);
        assert!(
            backoff_with_jitter(0, &params).is_ok(),
            "Should return Ok when backoff_micros < max_micros"
        );
    }

    // Kills arithmetic mutations on lines 645-646:
    //   replace + with - (jitter_min + hash%range)
    //   replace % with / or + (hash % jitter_range_pct)
    //   replace * with + or / (backoff * jitter_pct)
    //   replace / with % or * (result / 100)
    //
    // Strategy: verify output duration is in [backoff*jitter_min/100, backoff*(jitter_min+range-1)/100]
    #[test]
    fn backoff_duration_within_jitter_bounds() {
        // backoff = 10000us, jitter 50-98% → duration in [5000, 9800] us
        let params = params_with_bounds(10_000, 1_000_000);
        let backoff_us = 10_000u64;
        let min_expected = Duration::from_micros(backoff_us * 50 / 100);
        let max_expected = Duration::from_micros(backoff_us * 98 / 100);

        // Run multiple times to exercise varying hash values
        for _ in 0..20 {
            let dur = backoff_with_jitter(0, &params).unwrap();
            assert!(
                dur >= min_expected,
                "Duration {}us below minimum {}us",
                dur.as_micros(),
                min_expected.as_micros()
            );
            assert!(
                dur <= max_expected,
                "Duration {}us above maximum {}us",
                dur.as_micros(),
                max_expected.as_micros()
            );
        }
    }

    // Kills: replace backoff_with_jitter with Ok(Default::default())
    #[test]
    fn backoff_duration_nonzero_for_nonzero_backoff() {
        let params = params_with_bounds(1000, 1_000_000);
        let dur = backoff_with_jitter(0, &params).unwrap();
        assert!(
            dur.as_micros() >= 500,
            "Duration should be at least 50% of 1000us"
        );
    }

    // Kills: send_with_backoff arithmetic/comparison mutations via observable behavior
    #[test]
    fn send_with_backoff_succeeds_on_empty_channel() {
        let (tx, _rx) = spsc::spsc_channel::<u32>(4);
        let params = SchedulerParams::DEFAULT;
        assert!(send_with_backoff(&tx, 42u32, &params).is_ok());
    }

    #[test]
    fn send_with_backoff_returns_disconnected_when_receiver_dropped() {
        let (tx, rx) = spsc::spsc_channel::<u32>(4);
        drop(rx);
        let params = SchedulerParams::DEFAULT;
        assert!(
            matches!(
                send_with_backoff(&tx, 42, &params),
                Err(SendError::Disconnected)
            ),
            "Should return Disconnected when receiver has dropped"
        );
    }

    // Kills: comparison mutations on attempt thresholds (< vs == vs <=)
    // With correct code: spin phase attempts 0..spin_attempts, then times out via backoff.
    // With wrong comparisons: phase transitions differ, but timeout still fires since
    // we use instant-timeout params (max_backoff barely above min_backoff).
    #[test]
    fn send_with_backoff_returns_timeout_on_permanently_full_channel() {
        // Channel of capacity 2, fill it. Use minimal backoff so timeout fires on attempt 1.
        let (tx, _rx) = spsc::spsc_channel::<u32>(2);
        tx.try_send(1u32).unwrap();
        tx.try_send(2u32).unwrap();

        let params = SchedulerParams {
            spin_attempts: 0,
            yield_attempts: 0,
            min_backoff: Duration::from_micros(1),
            max_backoff: Duration::from_micros(1),
            jitter_min_pct: 50,
            jitter_range_pct: 49,
            ..SchedulerParams::DEFAULT
        };
        // attempt=0 → sleep phase, sleep_attempt=0, backoff=1*1=1, max=1, 1>1 false → sleep
        // attempt=1 → sleep phase, sleep_attempt=1, backoff=1*2=2, max=1, 2>1 → Timeout
        assert!(
            matches!(send_with_backoff(&tx, 3, &params), Err(SendError::Timeout)),
            "Should return Timeout when channel permanently full"
        );
    }
}

// Tests targeting missed mutations in handle_wake, exercised entirely through the
// public ActorScheduler/ActorHandle surface (new_with_params, send, poll_once, run).
#[cfg(test)]
mod handle_wake_targeted_tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    // An actor that records the last `SystemStatus` handed to `handle_os`. Since `handle_os`
    // is called unconditionally at the end of every `handle_wake`, its argument is
    // an observable proxy for the private `more_work` computation.
    struct StatusRecorder {
        last_status: Option<SystemStatus>,
    }
    impl Actor<i32, i32, i32> for StatusRecorder {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            self.last_status = Some(status);
            Ok(ActorStatus::Idle)
        }
    }

    // Kills: replace || with && at each of the three operators joining
    // matches!(control1|mgmt|control2|data, DrainStatus::More) into `more_work` (lines 973-975).
    //
    // Setup: half_control = 2 (control_burst_limit=4). Send 3 control messages so
    // control1 drains 2 (1 remains -> More), mgmt is empty (not More), control2 then
    // drains the last 1 (0 remain -> not More), data is empty (not More). Only the
    // *first* term is More, so more_work is true iff the chain is a plain OR — any
    // of the three `||`s replaced by `&&` collapses the result to false, flipping
    // the SystemStatus handed to `handle_os` from Busy to Idle.
    #[test]
    fn more_work_is_true_when_only_first_control_pass_hits_burst_limit() {
        let params = SchedulerParams {
            control_mgmt_buffer_size: 4,
            control_burst_multiplier: 1,
            ..SchedulerParams::DEFAULT
        };
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new_with_params(100, 100, params);
        for _ in 0..3 {
            tx.send(Message::Control(0)).unwrap();
        }

        let mut actor = StatusRecorder { last_status: None };
        rx.poll_once(&mut actor);

        assert_eq!(
            actor.last_status,
            Some(SystemStatus::Busy),
            "control1 alone hitting the burst limit must make more_work true"
        );
    }

    // Kills: replace / with * and replace / with % in
    // `half_control = (control_burst_limit / 2).max(1)` (line 940).
    //
    // control_burst_limit=16 -> correct half_control=8. Sending 9 messages makes
    // control1 (limit 8) leave exactly 1 behind (More); a `*` mutant inflates
    // half_control to 32 and drains all 9 in one pass (not More).
    #[test]
    fn half_control_division_not_multiplication() {
        let params = SchedulerParams {
            control_mgmt_buffer_size: 16,
            control_burst_multiplier: 1,
            ..SchedulerParams::DEFAULT
        };
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new_with_params(100, 100, params);
        for _ in 0..9 {
            tx.send(Message::Control(0)).unwrap();
        }

        let mut actor = StatusRecorder { last_status: None };
        rx.poll_once(&mut actor);

        assert_eq!(
            actor.last_status,
            Some(SystemStatus::Busy),
            "9 messages against half_control=8 must leave one behind (More)"
        );
    }

    /// The `Message` docs' disclaimer — "no cross-lane ordering guarantees, only best-effort
    /// priority" — as an executable fact rather than prose.
    ///
    /// Priority reorders messages that are **simultaneously pending** when a drain pass
    /// begins. It is not a barrier: a `Control` message that arrives *while a Management
    /// drain is already in flight* is not observed until that burst ends, because
    /// `handle_wake` hands `rx_mgmt.drain` its whole budget before the `control2` pass runs.
    /// Nothing can retract work the drain loop has already been handed.
    ///
    /// This exists because the opposite belief keeps getting written as a test: send N ticks,
    /// send a shutdown, send N more ticks, assert only the first N were counted. Those pass
    /// by timing luck on an idle machine and fail on a loaded one. Here the actor enqueues
    /// the `Control` message *itself*, from inside the drain, so the interleaving is caused
    /// by the program rather than arranged by a sleep — the counterexample is deterministic.
    #[test]
    fn a_control_message_enqueued_mid_drain_is_not_seen_until_the_drain_ends() {
        struct MidDrainSender {
            control: ActorHandle<i32, i32, i32>,
            log: Vec<&'static str>,
            sent: bool,
        }
        impl Actor<i32, i32, i32> for MidDrainSender {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                self.log.push("C");
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                self.log.push("M");
                // Exactly one, on the first Management message: four more are already
                // queued behind it, so a barrier would have to stop them.
                if !self.sent {
                    self.sent = true;
                    self.control.send(Message::Control(0)).unwrap();
                }
                Ok(())
            }
            fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                Ok(ActorStatus::Idle)
            }
        }

        let mut builder = ActorBuilder::<i32, i32, i32>::new(100, None);
        let preload = builder.add_producer();
        let from_handler = builder.add_producer();
        let mut rx = builder.build_with_burst(100, ShutdownMode::default());

        for _ in 0..5 {
            preload.send(Message::Management(0)).unwrap();
        }

        let mut actor = MidDrainSender {
            control: from_handler,
            log: Vec::new(),
            sent: false,
        };
        while actor.log.len() < 6 {
            assert!(
                !rx.poll_once(&mut actor),
                "the scheduler exited before draining both lanes, log {:?}",
                actor.log
            );
        }

        assert_eq!(
            actor.log,
            ["M", "M", "M", "M", "M", "C"],
            "the in-flight Management burst runs to completion first; \
             Control is best-effort priority, not a barrier"
        );
    }

    // Same setup, but with 5 messages: correct half_control=8 drains all 5 (not
    // More). A `%` mutant collapses half_control to (16 % 2).max(1) == 1, so
    // control1 only drains 1 of 5 and reports More instead.
    #[test]
    fn half_control_division_not_modulo() {
        let params = SchedulerParams {
            control_mgmt_buffer_size: 16,
            control_burst_multiplier: 1,
            ..SchedulerParams::DEFAULT
        };
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new_with_params(100, 100, params);
        for _ in 0..5 {
            tx.send(Message::Control(0)).unwrap();
        }

        let mut actor = StatusRecorder { last_status: None };
        rx.poll_once(&mut actor);

        assert_eq!(
            actor.last_status,
            Some(SystemStatus::Idle),
            "5 messages against half_control=8 must drain fully (not More)"
        );
    }

    // Kills: replace || with && (line 985:35) and replace == with != (line 985:52) in
    // `let status = if more_work || returned_hint == ActorStatus::Busy { Working } else { Idle }`.
    //
    // `more_work` is false throughout (a single control message, nowhere near any
    // burst limit). `handle_os` returns Busy on its first call only. Correct code: the
    // Busy hint alone makes the first handle_wake report Working, so run_inner
    // retries without blocking and calls handle_wake (and therefore handle_os) a second
    // time before it finally blocks on the doorbell — handle_os called exactly twice.
    // Under either mutant, `more_work || <busy-check>` collapses to false on the
    // first call, handle_wake reports Idle immediately, and run_inner blocks
    // before ever calling handle_os a second time.
    #[test]
    fn busy_park_hint_forces_one_extra_wake_before_blocking() {
        let (tx, mut rx) = ActorScheduler::<i32, i32, i32>::new(100, 100);

        struct BusyOnceActor {
            park_calls: Arc<AtomicUsize>,
        }
        impl Actor<i32, i32, i32> for BusyOnceActor {
            fn handle_data(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_control(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_management(&mut self, _: i32) -> HandlerResult {
                Ok(())
            }
            fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
                let n = self.park_calls.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Ok(ActorStatus::Busy)
                } else {
                    Ok(ActorStatus::Idle)
                }
            }
        }

        tx.send(Message::Control(0)).unwrap();

        let park_calls = Arc::new(AtomicUsize::new(0));
        let pc = park_calls.clone();
        let handle = std::thread::spawn(move || {
            let mut actor = BusyOnceActor { park_calls: pc };
            rx.run(&mut actor);
        });

        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            park_calls.load(Ordering::SeqCst),
            2,
            "Busy handle_os hint alone should force exactly one extra non-blocking wake"
        );

        drop(tx);
        handle.join().unwrap();
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
        fn handle_os(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    impl ActorTypes for EngineActor<'_> {
        type Data = EngineData;
        type Control = EngineControl;
        type Management = EngineManagement;
    }

    impl<'a> TroupeActor<&'a Directory> for EngineActor<'a> {
        fn new(_dir: &'a Directory) -> Self {
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
        fn handle_os(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    impl ActorTypes for DisplayActor<'_> {
        type Data = DisplayData;
        type Control = DisplayControl;
        type Management = DisplayManagement;
    }

    impl<'a> TroupeActor<&'a Directory> for DisplayActor<'a> {
        fn new(_dir: &'a Directory) -> Self {
            panic!("use new_with_counter instead")
        }
    }

    // === Per-actor Directory (what troupe! generates with SPSC) ===
    // Each actor gets its OWN Directory with dedicated SPSC handles.

    pub struct Directory {
        pub engine: ActorHandle<EngineData, EngineControl, EngineManagement>,
        pub display: ActorHandle<DisplayData, DisplayControl, DisplayManagement>,
    }

    /// Test the SPSC-based directory pattern: each actor gets its own Directory
    /// with dedicated SPSC handles to every other actor.
    #[test]
    fn troupe_directory_pattern() {
        // Create builders for each actor
        let mut engine_builder =
            ActorBuilder::<EngineData, EngineControl, EngineManagement>::new(1024, None);
        let mut display_builder =
            ActorBuilder::<DisplayData, DisplayControl, DisplayManagement>::new(1024, None);

        // Each "producer" (actor + external caller) gets dedicated SPSC handles
        // Directory for the test caller:
        let test_dir = Directory {
            engine: engine_builder.add_producer(),
            display: display_builder.add_producer(),
        };

        // An additional producer handle (e.g. for exposed handles)
        let extra_engine_handle = engine_builder.add_producer();

        // Build schedulers (seals builders — no more producers after this)
        let _engine_s = engine_builder.build();
        let _display_s = display_builder.build();

        // Verify cross-actor messaging works via directory
        test_dir
            .display
            .send(Message::Control(DisplayControl::Render))
            .unwrap();
        test_dir
            .engine
            .send(Message::Control(EngineControl::Tick))
            .unwrap();

        // Multiple handles are independent (each is a separate SPSC channel)
        extra_engine_handle
            .send(Message::Control(EngineControl::Tick))
            .unwrap();
    }

    /// Adversarial test: Malicious control sender trying to starve data lane
    /// Uses CONTINUOUS flooding to ensure burst limiting works during active attack.
    /// With SPSC, each producer has its own channel — no send-side contention.
    #[test]
    fn adversarial_control_flood_vs_data() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let mut builder = ActorBuilder::<i32, (), ()>::new(100, None);
        let tx_flood = builder.add_producer();
        let tx_data = builder.add_producer();
        let mut rx = builder.build_with_burst(100, ShutdownMode::default());

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

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
                fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }
            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Malicious control sender: CONTINUOUS flood via dedicated SPSC
        let stop_flag = stop_flooding.clone();
        let control_sender = thread::spawn(move || {
            let mut sent = 0;
            while !stop_flag.load(Ordering::Relaxed) {
                if tx_flood.send(Message::Control(())).is_ok() {
                    sent += 1;
                }
            }
            sent
        });

        thread::sleep(Duration::from_millis(20));

        // Well-behaved data sender via its own dedicated SPSC
        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                tx_data.send(Message::Data(i)).ok();
            }
        });

        data_sender.join().unwrap();
        thread::sleep(Duration::from_millis(50));

        stop_flooding.store(true, Ordering::Relaxed);
        let control_sent = control_sender.join().unwrap();

        // All handles dropped → scheduler exits
        thread::sleep(Duration::from_millis(50));
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Control flood vs data - Control sent: {}, processed: {}, Data processed: {}/100",
            control_sent, control_count, data_count
        );

        assert!(
            data_count > 0,
            "Data lane was completely starved during continuous control flood"
        );
        assert!(
            data_count > 50,
            "Burst limiting too weak - only {}/100 data processed during flood",
            data_count
        );
    }

    /// Adversarial test: Multiple bad actors teaming up to flood control.
    /// With SPSC, each flooder has its own channel — the scheduler still needs
    /// burst limiting to prevent consumer-side starvation.
    #[test]
    fn adversarial_multiple_control_flooders() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let mut builder = ActorBuilder::<i32, (), ()>::new(100, None);
        // 5 flood producers + 1 data producer
        let flood_handles: Vec<_> = (0..5).map(|_| builder.add_producer()).collect();
        let tx_data = builder.add_producer();
        let mut rx = builder.build_with_burst(100, ShutdownMode::default());

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

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
                fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }
            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        // Each flooder has its own SPSC channel
        let mut control_threads = vec![];
        for tx_flood in flood_handles {
            let stop_flag = stop_flooding.clone();
            let handle = thread::spawn(move || {
                let mut sent = 0;
                while !stop_flag.load(Ordering::Relaxed) {
                    if tx_flood.send(Message::Control(())).is_ok() {
                        sent += 1;
                    }
                }
                sent
            });
            control_threads.push(handle);
        }

        thread::sleep(Duration::from_millis(20));

        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                tx_data.send(Message::Data(i)).ok();
            }
        });

        data_sender.join().unwrap();
        thread::sleep(Duration::from_millis(50));

        stop_flooding.store(true, Ordering::Relaxed);
        let mut total_control_sent = 0;
        for handle in control_threads {
            total_control_sent += handle.join().unwrap();
        }

        thread::sleep(Duration::from_millis(50));
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Multiple attackers - Control sent: {}, processed: {}, Data: {}/100",
            total_control_sent, control_count, data_count
        );

        assert!(
            data_count > 0,
            "Data lane completely starved by coordinated control attack"
        );
        assert!(
            data_count > 50,
            "Burst limiting too weak against coordinated attack - only {}/100 data processed",
            data_count
        );
    }

    /// Adversarial test: Continuous control flood with concurrent data
    #[test]
    fn adversarial_continuous_control_flood() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let mut builder = ActorBuilder::<i32, (), ()>::new(100, None);
        let tx_flood = builder.add_producer();
        let tx_data = builder.add_producer();
        let mut rx = builder.build_with_burst(100, ShutdownMode::default());

        let control_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));
        let stop_flooding = Arc::new(AtomicBool::new(false));

        let cp = control_processed.clone();
        let dp = data_processed.clone();

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
                fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    Ok(ActorStatus::Busy)
                }
            }
            let mut actor = TestActor {
                control_count: cp,
                data_count: dp,
            };
            rx.run(&mut actor);
        });

        let stop_flag = stop_flooding.clone();
        let control_flooder = thread::spawn(move || {
            let mut sent = 0;
            while !stop_flag.load(Ordering::Relaxed) {
                if tx_flood.send(Message::Control(())).is_ok() {
                    sent += 1;
                }
            }
            sent
        });

        thread::sleep(Duration::from_millis(50));

        let data_sender = thread::spawn(move || {
            for i in 0..100 {
                tx_data.send(Message::Data(i)).ok();
            }
        });

        data_sender.join().unwrap();
        thread::sleep(Duration::from_millis(100));

        stop_flooding.store(true, Ordering::Relaxed);
        let control_sent = control_flooder.join().unwrap();

        thread::sleep(Duration::from_millis(50));
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let data_count = data_processed.load(Ordering::Relaxed);

        println!(
            "Continuous flood - Control sent: {}, processed: {}, Data processed: {}/100",
            control_sent, control_count, data_count
        );

        assert!(
            data_count > 0,
            "Burst limiting FAILED - data was starved during continuous control flood"
        );
        assert!(
            data_count > 50,
            "Burst limiting is too weak - only {}/100 data messages processed",
            data_count
        );
    }

    /// Adversarial test: Slow receiver with multiple aggressive senders.
    /// Each sender has its own SPSC channels — backoff is per-producer.
    #[test]
    fn adversarial_slow_receiver_resilience() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let mut builder = ActorBuilder::<i32, i32, i32>::new(10, None);
        let senders: Vec<_> = (0..3).map(|_| builder.add_producer()).collect();
        let mut rx = builder.build_with_burst(10, ShutdownMode::default());

        let control_processed = Arc::new(AtomicUsize::new(0));
        let mgmt_processed = Arc::new(AtomicUsize::new(0));
        let data_processed = Arc::new(AtomicUsize::new(0));

        let cp = control_processed.clone();
        let mp = mgmt_processed.clone();
        let dp = data_processed.clone();

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
                fn handle_os(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
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

        // Each sender has its own SPSC channels
        let mut sender_handles = vec![];
        for (sender_id, tx) in senders.into_iter().enumerate() {
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let msg_val = (sender_id * 1000 + i) as i32;
                    tx.send(Message::Control(msg_val)).ok();
                    tx.send(Message::Management(msg_val)).ok();
                }
            });
            sender_handles.push(handle);
        }

        for handle in sender_handles {
            handle.join().unwrap();
        }

        thread::sleep(Duration::from_millis(1000));
        receiver_handle.join().unwrap();

        let control_count = control_processed.load(Ordering::Relaxed);
        let mgmt_count = mgmt_processed.load(Ordering::Relaxed);

        println!(
            "Slow receiver resilience - Control: {}, Mgmt: {}, Data: {}",
            control_count,
            mgmt_count,
            data_processed.load(Ordering::Relaxed)
        );

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

/// Test module for troupe nesting pattern (SPSC-based)
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
        fn handle_os(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    impl ActorTypes for WorkerActor<'_> {
        type Data = WorkerData;
        type Control = WorkerControl;
        type Management = WorkerManagement;
    }

    impl<'a> TroupeActor<&'a WorkerDirectory> for WorkerActor<'a> {
        fn new(_dir: &'a WorkerDirectory) -> Self {
            panic!("test only")
        }
    }

    // Manual directory for worker troupe (per-actor owned)
    pub struct WorkerDirectory {
        pub worker: ActorHandle<WorkerData, WorkerControl, WorkerManagement>,
    }

    // Manual ExposedHandles for worker troupe
    pub struct WorkerExposedHandles {
        pub worker: ActorHandle<WorkerData, WorkerControl, WorkerManagement>,
    }

    // Manual Troupe struct for worker - stores builder (not scheduler) until play()
    pub struct WorkerTroupe {
        // Builder stays alive until play() so exposed() can add producers
        worker_builder: ActorBuilder<WorkerData, WorkerControl, WorkerManagement>,
        // Pre-created directory for the worker actor itself
        pub worker_dir: WorkerDirectory,
    }

    impl WorkerTroupe {
        pub fn new() -> Self {
            let mut builder =
                ActorBuilder::<WorkerData, WorkerControl, WorkerManagement>::new(1024, None);

            // Worker's own handle to itself (self-loop)
            let worker_dir = WorkerDirectory {
                worker: builder.add_producer(),
            };

            Self {
                worker_builder: builder,
                worker_dir,
            }
        }

        /// Create exposed handles by adding new producers to the builder.
        /// Must be called before play() since play() consumes the builder.
        pub fn exposed(&mut self) -> WorkerExposedHandles {
            WorkerExposedHandles {
                worker: self.worker_builder.add_producer(),
            }
        }
    }

    /// Test the two-phase Troupe pattern: new() → exposed() → play()
    #[test]
    fn troupe_two_phase_pattern() {
        // Phase 1: Create child troupe (no threads yet)
        let mut child = WorkerTroupe::new();

        // Phase 2: Parent grabs exposed handles (each call creates new SPSC channels)
        let exposed = child.exposed();

        // Parent can now send to child even before child.play()
        exposed
            .worker
            .send(Message::Control(WorkerControl::Process))
            .unwrap();
        exposed
            .worker
            .send(Message::Data(WorkerData("hello".to_string())))
            .unwrap();

        // Multiple exposed() calls create independent handles
        let exposed2 = child.exposed();
        exposed2
            .worker
            .send(Message::Control(WorkerControl::Process))
            .unwrap();

        // Note: We don't call play() here since that would block.
        // The test verifies the two-phase construction pattern works.
    }

    /// Test that ExposedHandles can outlive the Troupe struct
    #[test]
    fn exposed_handles_outlive_troupe_struct() {
        let exposed = {
            let mut child = WorkerTroupe::new();
            child.exposed() // ExposedHandles escapes
        };
        // Troupe struct dropped, but handles still valid (SPSC channels still open
        // until both sides drop). Builder was not consumed by build(), so receiver
        // side is also dropped — handles become disconnected.

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

        fn handle_os(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            match status {
                SystemStatus::Idle => Ok(ActorStatus::Idle),
                SystemStatus::Busy => Ok(ActorStatus::Busy),
            }
        }
    }

    #[test]
    fn shutdown_immediate_exits_quickly_under_flood() {
        let (tx, mut rx) = ActorScheduler::new(100, 100);

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
            tx.send(Message::Data(i)).ok();
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
}
