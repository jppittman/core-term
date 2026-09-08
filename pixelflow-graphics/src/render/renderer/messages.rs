//! Message types for render actor communication.
//!
//! The render actor uses a **bootstrap pattern** for initialization:
//!
//! 1. Call `RendererActor::spawn_with_setup(num_threads)` to get a `SetupHandle`
//! 2. Send your response channel via `setup_handle.register(response_tx)`
//! 3. Receive the full `RendererHandle` back - now you can send render requests
//!
//! This pattern enforces at the **type level** that you cannot send render requests
//! without first registering where responses should go.
//!
//! ## Priority Lanes (after bootstrap)
//!
//! - **Data**: Frame rendering requests (backpressure when full)
//! - **Control**: Shutdown and high-priority commands
//! - **Management**: Configuration updates (thread count, etc.)

use crate::render::frame::Frame;
use crate::render::scene::Scene;
use crate::render::Pixel;
use actor_scheduler::ActorHandle;
use std::sync::mpsc::{self, Sender, SyncSender};
use std::time::Duration;

/// Frame rendering request (Data lane - high throughput, backpressure).
///
/// The Data lane is designed for high-volume work items and will block
/// senders when the buffer is full, providing natural backpressure.
pub struct RenderRequest<P: Pixel, Meta = ()> {
    /// The scene to render: four channel kernels compiled at the frame's
    /// lattice shape.
    pub scene: Scene,
    /// The frame buffer to render into.
    pub frame: Frame<P>,
    /// Caller-owned payload, returned untouched in the [`RenderResponse`].
    ///
    /// The renderer never reads this. It exists so a caller that has to split a larger value
    /// apart to send the frame here can send the *rest of it* along too, instead of stashing it
    /// somewhere and reassembling on completion — the state that gets stashed is not
    /// coordination state, it is half of a torn value. Defaults to `()` for callers with
    /// nothing to carry.
    pub meta: Meta,
}

/// Completed frame rendering response.
pub struct RenderResponse<P: Pixel, Meta = ()> {
    /// The rendered frame.
    pub frame: Frame<P>,
    /// Time taken to render, or `None` if the renderer was paused and did not render.
    ///
    /// The frame comes back either way — see [`RenderRequest::frame`]. Whether it was *rendered*
    /// is the optional part; whether the caller gets its buffer back never was.
    pub render_time: Option<Duration>,
    /// The [`RenderRequest::meta`] payload, returned exactly as it was given.
    pub meta: Meta,
}

/// Control messages (Control lane - highest priority, sleep-based fairness).
///
/// Control messages are processed before Management and Data messages.
/// The Control lane uses sleep-based backoff to ensure fairness and prevent
/// starvation of other message types.
///
/// To shut down the scheduler, use `Message::Shutdown` directly, not a control message.
#[derive(Debug, Clone, Copy)]
pub enum RenderControl {
    /// Pause rendering (stop processing Data messages).
    Pause,
    /// Resume rendering.
    Resume,
}

/// Management messages (Management lane - medium priority, configuration).
///
/// Management messages are processed after Control but before Data.
/// These are used for configuration changes that should be applied promptly
/// but don't need to interrupt ongoing work.
pub enum RenderManagement {
    /// Update the number of rendering threads.
    SetThreadCount(usize),
    /// Query current configuration (sends response via channel).
    GetConfig {
        response_tx: std::sync::mpsc::Sender<RenderConfig>,
    },
}

/// Current renderer configuration.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Number of threads used for work-stealing parallelism.
    pub num_threads: usize,
    /// Whether rendering is paused.
    pub paused: bool,
}

// ============================================================================
// Bootstrap Types - Type-level enforcement of initialization order
// ============================================================================

/// Setup message sent during bootstrap to register the response channel.
///
/// This message is sent through a dedicated setup channel, separate from
/// the actor's normal message lanes. The renderer blocks on this channel
/// before entering its main run loop.
pub struct RendererSetup<P: Pixel, Meta = ()> {
    /// Channel where completed frames will be sent.
    pub response_tx: Sender<RenderResponse<P, Meta>>,
    /// Channel to send back the full actor handle.
    pub(crate) reply_tx: SyncSender<RendererHandle<P, Meta>>,
}

/// Handle returned after successful bootstrap - now you can send render requests.
///
/// This is the full actor handle that allows sending Data, Control, and Management
/// messages. You can only obtain this by completing the bootstrap handshake.
pub type RendererHandle<P, Meta = ()> =
    ActorHandle<RenderRequest<P, Meta>, RenderControl, RenderManagement>;

/// Handle for initial setup - can ONLY register the response channel.
///
/// This is a capability-restricted handle. The only thing you can do with it
/// is call `register()` to complete the bootstrap handshake and receive
/// the full `RendererHandle`.
pub struct RendererSetupHandle<P: Pixel, Meta = ()> {
    setup_tx: SyncSender<RendererSetup<P, Meta>>,
}

impl<P: Pixel, Meta> RendererSetupHandle<P, Meta> {
    /// Create a new setup handle with the given channel.
    pub(crate) fn new(setup_tx: SyncSender<RendererSetup<P, Meta>>) -> Self {
        Self { setup_tx }
    }

    /// Complete the bootstrap handshake by registering the response channel.
    ///
    /// This method:
    /// 1. Sends your response channel to the renderer
    /// 2. Waits for the renderer to send back its full actor handle
    /// 3. Returns the handle, allowing you to send render requests
    ///
    /// # Panics
    ///
    /// Panics if the render thread has died before completing setup.
    #[must_use]
    pub fn register(self, response_tx: Sender<RenderResponse<P, Meta>>) -> RendererHandle<P, Meta> {
        // Create reply channel for this handshake
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);

        // Build the setup message
        let setup = RendererSetup {
            response_tx,
            reply_tx,
        };

        // Send setup - blocks if channel full (shouldn't happen with buffer=1)
        self.setup_tx
            .send(setup)
            .expect("Render thread died before setup");

        // Wait for the full handle
        reply_rx.recv().expect("Render thread died during setup")
    }
}

// Implement message traits for actor-scheduler integration
actor_scheduler::impl_control_message!(RenderControl);
