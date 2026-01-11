//! Rasterizer actor for asynchronous frame rendering.
//!
//! The `RasterizerActor` provides a message-based interface for parallel frame
//! rendering using the actor-scheduler's three-lane priority system:
//!
//! - **Data Lane**: Frame rendering requests with natural backpressure
//! - **Management Lane**: Thread count updates and configuration queries
//! - **Control Lane**: Shutdown and pause/resume commands
//!
//! # Example
//!
//! ```ignore
//! use actor_scheduler::{ActorScheduler, Message};
//! use pixelflow_graphics::render::rasterizer::{RasterizerActor, RasterManagement};
//!
//! let (handle, mut scheduler) = ActorScheduler::new(10, 64);
//! let mut actor = RasterizerActor::new(4); // 4 rendering threads
//!
//! // Spawn actor thread
//! std::thread::spawn(move || {
//!     scheduler.run(&mut actor);
//! });
//!
//! // Update thread count
//! handle.send(Message::Management(RasterManagement::SetThreadCount(8))).unwrap();
//! ```

use super::messages::{RasterConfig, RasterControl, RasterManagement, RenderRequest, RenderResponse};
use super::{rasterize, TensorShape};
use crate::render::Pixel;
use actor_scheduler::{Actor, ParkHint};
use std::time::Instant;

/// Rasterizer actor for parallel frame rendering.
///
/// This actor manages a pool of worker threads for rendering frames via
/// work-stealing parallelism. It processes rendering requests asynchronously
/// while allowing dynamic reconfiguration of thread count.
pub struct RasterizerActor<P: Pixel> {
    /// Number of threads for work-stealing parallelism.
    num_threads: usize,
    /// Whether rendering is currently paused.
    paused: bool,
    /// Phantom data to tie the pixel type.
    _phantom: std::marker::PhantomData<P>,
}

impl<P: Pixel> RasterizerActor<P> {
    /// Create a new rasterizer actor with the specified number of threads.
    ///
    /// # Arguments
    ///
    /// * `num_threads` - Number of worker threads for parallel rendering.
    ///   Use 1 for single-threaded, or `std::thread::available_parallelism()`
    ///   for utilizing all CPU cores.
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
            paused: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get current configuration.
    fn config(&self) -> RasterConfig {
        RasterConfig {
            num_threads: self.num_threads,
            paused: self.paused,
        }
    }
}

impl<P: Pixel + Send> Actor<RenderRequest<P>, RasterControl, RasterManagement>
    for RasterizerActor<P>
{
    fn handle_data(&mut self, request: RenderRequest<P>) {
        // Skip rendering if paused
        if self.paused {
            log::debug!("Rasterizer paused, dropping render request");
            return;
        }

        let RenderRequest {
            manifold,
            mut frame,
            width,
            height,
            response_tx,
        } = request;

        // Render the frame
        let start = Instant::now();
        let shape = TensorShape::new(width, height);
        rasterize(&manifold, frame.as_slice_mut(), shape, self.num_threads);
        let render_time = start.elapsed();

        log::trace!(
            "Rendered {}x{} frame in {:?} ({} threads)",
            width,
            height,
            render_time,
            self.num_threads
        );

        // Send response back (ignore errors if receiver dropped)
        let response = RenderResponse { frame, render_time };
        let _ = response_tx.send(response);
    }

    fn handle_control(&mut self, ctrl: RasterControl) {
        match ctrl {
            RasterControl::Shutdown => {
                log::info!("Rasterizer actor shutting down");
                // Shutdown is handled by the scheduler, nothing to do here
            }
            RasterControl::Pause => {
                log::info!("Rasterizer paused");
                self.paused = true;
            }
            RasterControl::Resume => {
                log::info!("Rasterizer resumed");
                self.paused = false;
            }
        }
    }

    fn handle_management(&mut self, mgmt: RasterManagement) {
        match mgmt {
            RasterManagement::SetThreadCount(count) => {
                let new_count = count.max(1);
                log::info!(
                    "Rasterizer thread count updated: {} -> {}",
                    self.num_threads,
                    new_count
                );
                self.num_threads = new_count;
            }
            RasterManagement::GetConfig { response_tx } => {
                let _ = response_tx.send(self.config());
            }
        }
    }

    fn park(&mut self, hint: ParkHint) -> ParkHint {
        // No external work to do during park, just wait for messages
        hint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::color::Rgba8;
    use crate::render::frame::Frame;
    use crate::render::Color;
    use actor_scheduler::{ActorScheduler, Message};
    use std::sync::{mpsc, Arc};

    #[test]
    fn test_rasterizer_actor_basic() {
        let (handle, mut scheduler) = ActorScheduler::new(10, 64);
        let mut actor = RasterizerActor::<Rgba8>::new(1);

        // Spawn actor thread
        let actor_thread = std::thread::spawn(move || {
            scheduler.run(&mut actor);
        });

        // Create a render request
        let (response_tx, response_rx) = mpsc::channel();
        let frame = Frame::new(64, 64);
        let red = Color::Rgb(255, 0, 0);

        let request = RenderRequest {
            manifold: Arc::new(red),
            frame,
            width: 64,
            height: 64,
            response_tx,
        };

        // Send render request
        handle
            .send(Message::Data(request))
            .expect("Failed to send render request");

        // Wait for response
        let response = response_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("Failed to receive response");

        // Verify frame was rendered
        assert_eq!(response.frame.width, 64);
        assert_eq!(response.frame.height, 64);
        assert!(response.render_time.as_nanos() > 0);

        // Shutdown
        handle
            .send(Message::Control(RasterControl::Shutdown))
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }

    #[test]
    fn test_rasterizer_actor_thread_count_update() {
        let (handle, mut scheduler) = ActorScheduler::new(10, 64);
        let mut actor = RasterizerActor::<Rgba8>::new(2);

        // Spawn actor thread
        let actor_thread = std::thread::spawn(move || {
            scheduler.run(&mut actor);
        });

        // Update thread count
        handle
            .send(Message::Management(RasterManagement::SetThreadCount(4)))
            .expect("Failed to send SetThreadCount");

        // Query config
        let (config_tx, config_rx) = mpsc::channel();
        handle
            .send(Message::Management(RasterManagement::GetConfig {
                response_tx: config_tx,
            }))
            .expect("Failed to send GetConfig");

        let config = config_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("Failed to receive config");

        assert_eq!(config.num_threads, 4);
        assert!(!config.paused);

        // Shutdown
        handle
            .send(Message::Control(RasterControl::Shutdown))
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }

    #[test]
    fn test_rasterizer_actor_pause_resume() {
        let (handle, mut scheduler) = ActorScheduler::new(10, 64);
        let mut actor = RasterizerActor::<Rgba8>::new(1);

        // Spawn actor thread
        let actor_thread = std::thread::spawn(move || {
            scheduler.run(&mut actor);
        });

        // Pause rendering
        handle
            .send(Message::Control(RasterControl::Pause))
            .expect("Failed to send Pause");

        // Send a render request (should be dropped)
        let (response_tx, response_rx) = mpsc::channel();
        let frame = Frame::new(32, 32);
        let blue = Color::Rgb(0, 0, 255);

        let request = RenderRequest {
            manifold: Arc::new(blue),
            frame,
            width: 32,
            height: 32,
            response_tx,
        };

        handle
            .send(Message::Data(request))
            .expect("Failed to send render request");

        // Should timeout because rendering is paused
        assert!(response_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err());

        // Resume
        handle
            .send(Message::Control(RasterControl::Resume))
            .expect("Failed to send Resume");

        // Shutdown
        handle
            .send(Message::Control(RasterControl::Shutdown))
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }
}
