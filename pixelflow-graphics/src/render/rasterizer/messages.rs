//! Message types for rasterizer actor communication.
//!
//! The rasterizer actor uses three priority lanes:
//! - **Data**: Frame rendering requests (backpressure when full)
//! - **Control**: Shutdown and high-priority commands
//! - **Management**: Configuration updates (thread count, etc.)

use crate::render::frame::Frame;
use crate::render::Pixel;
use pixelflow_core::{Discrete, Manifold};
use std::sync::Arc;
use std::time::Duration;

/// Frame rendering request (Data lane - high throughput, backpressure).
///
/// The Data lane is designed for high-volume work items and will block
/// senders when the buffer is full, providing natural backpressure.
pub struct RenderRequest<P: Pixel> {
    /// The color manifold to render.
    pub manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync>,
    /// The frame buffer to render into.
    pub frame: Frame<P>,
    /// Channel to send the completed frame back.
    pub response_tx: std::sync::mpsc::Sender<RenderResponse<P>>,
}

/// Completed frame rendering response.
#[derive(Debug)]
pub struct RenderResponse<P: Pixel> {
    /// The rendered frame.
    pub frame: Frame<P>,
    /// Time taken to render the frame.
    pub render_time: Duration,
}

/// Control messages (Control lane - highest priority, sleep-based fairness).
///
/// Control messages are processed before Management and Data messages.
/// The Control lane uses sleep-based backoff to ensure fairness and prevent
/// starvation of other message types.
///
/// To shut down the scheduler, use `Message::Shutdown` directly, not a control message.
#[derive(Debug, Clone, Copy)]
pub enum RasterControl {
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
#[derive(Debug, Clone)]
pub enum RasterManagement {
    /// Update the number of rendering threads.
    SetThreadCount(usize),
    /// Query current configuration (sends response via channel).
    GetConfig {
        response_tx: std::sync::mpsc::Sender<RasterConfig>,
    },
}

/// Current rasterizer configuration.
#[derive(Debug, Clone)]
pub struct RasterConfig {
    /// Number of threads used for work-stealing parallelism.
    pub num_threads: usize,
    /// Whether rendering is paused.
    pub paused: bool,
}

// Implement message traits for actor-scheduler integration
actor_scheduler::impl_control_message!(RasterControl);
actor_scheduler::impl_management_message!(RasterManagement);
