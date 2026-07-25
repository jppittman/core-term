//! Rasterizer actor for asynchronous frame rendering.
//!
//! The `RasterizerActor` provides a message-based interface for parallel frame
//! rendering using the actor-scheduler's three-lane priority system:
//!
//! - **Data Lane**: Frame rendering requests with natural backpressure
//! - **Management Lane**: Thread count updates and configuration queries
//! - **Control Lane**: Shutdown and pause/resume commands
//!
//! # Bootstrap Pattern
//!
//! The rasterizer uses a **bootstrap handshake** to ensure you can't send
//! render requests before registering where responses should go:
//!
//! ```ignore
//! use pixelflow_graphics::render::rasterizer::RasterizerActor;
//! use std::sync::mpsc;
//!
//! // Step 1: Spawn with setup handle
//! let (setup_handle, join_handle) = RasterizerActor::spawn_with_setup(4);
//!
//! // Step 2: Create your response channel
//! let (response_tx, response_rx) = mpsc::channel();
//!
//! // Step 3: Register and get full handle - NOW you can send render requests
//! let rasterizer = setup_handle.register(response_tx);
//!
//! // Step 4: Send render requests
//! rasterizer.send(Message::Data(my_render_request)).unwrap();
//!
//! // Step 5: Receive responses
//! let response = response_rx.recv().unwrap();
//! ```

use super::messages::{
    RasterConfig, RasterControl, RasterManagement, RasterSetup, RasterizerSetupHandle,
    RenderRequest, RenderResponse,
};
use super::rasterize;
use crate::render::Pixel;
use actor_scheduler::mealy::Transducer;
use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, ActorTypes, HandlerError, HandlerResult, SystemStatus,
};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────────────
// RasterCore: the pure decision logic
// ────────────────────────────────────────────────────────────────────────────

/// What a render request decides to emit. `Default` (no response) covers the paused case —
/// mirrors `VsyncCoreOut` in `vsync_actor.rs`: at most one output value per step.
pub(crate) struct RasterCoreOut<P: Pixel, Meta = ()> {
    pub(crate) response: Option<RenderResponse<P, Meta>>,
}

// Hand-written rather than `#[derive(Default)]`: the derive macro would add a spurious
// `P: Default` bound even though `Option<RenderResponse<P>>` needs no such bound on `P`.
impl<P: Pixel, Meta> Default for RasterCoreOut<P, Meta> {
    fn default() -> Self {
        Self { response: None }
    }
}

/// Pure rasterizer decision core: pause/resume, thread-count, and the actual `rasterize` call,
/// with no channel or bootstrap-handshake machinery in it — a `step_*` call takes a message and
/// returns what to emit, table-testable with no actor scheduler in the loop. Rollout step 3 of
/// `docs/designs/pixelflow-runtime-engine-mesh-migration.md` §5, following the same
/// core/adapter split step 2 established for `vsync_actor.rs`'s `VsyncCore`.
pub(crate) struct RasterCore<P: Pixel, Meta = ()> {
    num_threads: usize,
    paused: bool,
    _pixel: std::marker::PhantomData<(P, Meta)>,
}

impl<P: Pixel, Meta> RasterCore<P, Meta> {
    fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
            paused: false,
            _pixel: std::marker::PhantomData,
        }
    }

    fn config(&self) -> RasterConfig {
        RasterConfig {
            num_threads: self.num_threads,
            paused: self.paused,
        }
    }
}

impl<P: Pixel, Meta> Transducer for RasterCore<P, Meta> {
    type Control = RasterControl;
    type Management = RasterManagement;
    type Data = RenderRequest<P, Meta>;
    type Out = RasterCoreOut<P, Meta>;

    fn step_data(
        &mut self,
        request: RenderRequest<P, Meta>,
    ) -> Result<RasterCoreOut<P, Meta>, HandlerError> {
        if self.paused {
            log::debug!("Rasterizer paused, dropping render request");
            return Ok(RasterCoreOut::default());
        }

        let RenderRequest {
            manifold,
            mut frame,
            meta,
        } = request;

        let start = Instant::now();
        rasterize(&manifold, &mut frame, self.num_threads);
        let render_time = start.elapsed();

        log::trace!(
            "Rendered {}x{} frame in {:?} ({} threads)",
            frame.width,
            frame.height,
            render_time,
            self.num_threads
        );

        Ok(RasterCoreOut {
            // `meta` is handed straight back, never inspected — see `RenderRequest::meta`.
            response: Some(RenderResponse {
                frame,
                render_time,
                meta,
            }),
        })
    }

    fn step_control(
        &mut self,
        ctrl: RasterControl,
    ) -> Result<RasterCoreOut<P, Meta>, HandlerError> {
        match ctrl {
            RasterControl::Pause => {
                log::info!("Rasterizer paused");
                self.paused = true;
            }
            RasterControl::Resume => {
                log::info!("Rasterizer resumed");
                self.paused = false;
            }
        }
        Ok(RasterCoreOut::default())
    }

    fn step_management(
        &mut self,
        mgmt: RasterManagement,
    ) -> Result<RasterCoreOut<P, Meta>, HandlerError> {
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
                // Receiver may be dropped if requester cancelled, that's fine.
                response_tx.send(self.config()).ok();
            }
        }
        Ok(RasterCoreOut::default())
    }
}

#[cfg(test)]
mod core_tests {
    //! `RasterCore` in isolation — no threads, no bootstrap handshake, no scheduler. Mirrors
    //! `vsync_actor.rs`'s `mod tests` for `VsyncCore`.

    use super::*;
    use crate::render::color::Rgba8;
    use crate::render::frame::Frame;
    use crate::render::Color;
    use std::sync::Arc;
    use std::sync::mpsc;

    fn request(size: u32) -> RenderRequest<Rgba8> {
        RenderRequest {
            manifold: Arc::new(Color::Rgb(255, 0, 0)),
            frame: Frame::new(size, size),
            meta: (),
        }
    }

    /// The property `meta` exists for: a caller that must split a value apart to send the frame
    /// can send the rest of it along and get it back, instead of stashing it and reassembling
    /// on completion. `pixelflow-runtime`'s `pending_render` is exactly that stash.
    #[test]
    fn meta_round_trips_untouched_through_a_render() {
        #[derive(Debug, PartialEq)]
        struct WindowMeta {
            id: u64,
            width_px: u32,
            scale: f64,
        }

        let mut core = RasterCore::<Rgba8, WindowMeta>::new(1);
        let out = core
            .step_data(RenderRequest {
                manifold: Arc::new(Color::Rgb(0, 255, 0)),
                frame: Frame::new(8, 8),
                meta: WindowMeta {
                    id: 42,
                    width_px: 1920,
                    scale: 2.0,
                },
            })
            .unwrap();

        let response = out.response.expect("must render when not paused");
        assert_eq!(
            response.meta,
            WindowMeta {
                id: 42,
                width_px: 1920,
                scale: 2.0
            },
            "meta must come back exactly as it went in"
        );
        assert_eq!(response.frame.width, 8, "and the frame is still rendered");
    }

    #[test]
    fn an_unpaused_core_renders_and_emits_a_response() {
        let mut core = RasterCore::<Rgba8>::new(1);
        let out = core.step_data(request(8)).unwrap();
        let response = out.response.expect("must render when not paused");
        assert_eq!(response.frame.width, 8);
        assert_eq!(response.frame.height, 8);
    }

    #[test]
    fn a_paused_core_drops_the_request_without_rendering() {
        let mut core = RasterCore::<Rgba8>::new(1);
        core.step_control(RasterControl::Pause).unwrap();
        let out = core.step_data(request(8)).unwrap();
        assert!(
            out.response.is_none(),
            "a paused core must not render or emit a response"
        );
    }

    #[test]
    fn resume_after_pause_allows_rendering_again() {
        let mut core = RasterCore::<Rgba8>::new(1);
        core.step_control(RasterControl::Pause).unwrap();
        core.step_control(RasterControl::Resume).unwrap();
        let out = core.step_data(request(8)).unwrap();
        assert!(out.response.is_some(), "resume must unblock rendering");
    }

    #[test]
    fn set_thread_count_is_reflected_in_config() {
        let mut core = RasterCore::<Rgba8>::new(1);
        core.step_management(RasterManagement::SetThreadCount(4))
            .unwrap();

        let (response_tx, response_rx) = mpsc::channel();
        core.step_management(RasterManagement::GetConfig { response_tx })
            .unwrap();

        let config = response_rx.recv().unwrap();
        assert_eq!(config.num_threads, 4);
        assert!(!config.paused);
    }

    #[test]
    fn zero_thread_count_is_clamped_to_one() {
        let mut core = RasterCore::<Rgba8>::new(1);
        core.step_management(RasterManagement::SetThreadCount(0))
            .unwrap();

        let (response_tx, response_rx) = mpsc::channel();
        core.step_management(RasterManagement::GetConfig { response_tx })
            .unwrap();

        assert_eq!(
            response_rx.recv().unwrap().num_threads,
            1,
            "a thread count of 0 would never render — clamp to at least 1"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RasterizerActor: the thin adapter — bootstrap handshake, real sends
// ────────────────────────────────────────────────────────────────────────────

/// Rasterizer actor for parallel frame rendering.
///
/// Owns exactly what [`RasterCore`] cannot: the response channel and the bootstrap handshake
/// that sets it up. Every decision (pause/resume, thread count, whether to actually render) is
/// [`RasterCore`]'s job; this type only turns its `Out` into the real `response_tx.send(...)`.
///
/// Use [`spawn_with_setup`](Self::spawn_with_setup) to create and start the actor.
pub struct RasterizerActor<P: Pixel, Meta = ()> {
    /// Channel to send completed frames back. Set during bootstrap.
    response_tx: Sender<RenderResponse<P, Meta>>,
    core: RasterCore<P, Meta>,
}

impl<P: Pixel + Send + 'static, Meta: Send + 'static> ActorTypes for RasterizerActor<P, Meta> {
    type Data = RenderRequest<P, Meta>;
    type Control = RasterControl;
    type Management = RasterManagement;
}

impl<P: Pixel + Send + 'static, Meta: Send + 'static> RasterizerActor<P, Meta> {
    /// Spawn the rasterizer actor with a bootstrap handshake.
    ///
    /// This is the **primary way** to create a rasterizer. It spawns the actor
    /// thread and returns a `SetupHandle` that you must use to register your
    /// response channel before sending any render requests.
    ///
    /// # Arguments
    ///
    /// * `num_threads` - Number of worker threads for parallel rendering.
    ///   Use 1 for single-threaded, or `std::thread::available_parallelism()`
    ///   for utilizing all CPU cores.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `RasterizerSetupHandle` - Use this to register your response channel
    /// - `JoinHandle` - The thread handle for the rasterizer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (setup_handle, _thread) = RasterizerActor::spawn_with_setup(4);
    /// let (response_tx, response_rx) = std::sync::mpsc::channel();
    /// let rasterizer = setup_handle.register(response_tx);
    /// // Now you can send render requests via `rasterizer`
    /// ```
    #[must_use]
    pub fn spawn_with_setup(
        num_threads: usize,
    ) -> (RasterizerSetupHandle<P, Meta>, JoinHandle<()>) {
        // Create the setup channel (buffer=1, only one setup message ever)
        let (setup_tx, setup_rx) = mpsc::sync_channel::<RasterSetup<P, Meta>>(1);

        // Spawn the actor thread
        let join_handle = thread::spawn(move || {
            // PHASE 1: Wait for setup message (blocks until register() is called)
            let setup = setup_rx
                .recv()
                .expect("Setup handle dropped without calling register()");

            // Extract response channel from setup
            let response_tx = setup.response_tx;
            let reply_tx = setup.reply_tx;

            // PHASE 2: Create the actor scheduler
            let (handle, mut scheduler) =
                ActorScheduler::<RenderRequest<P, Meta>, RasterControl, RasterManagement>::new(
                    64, 16,
                );

            // Send the full handle back to the caller
            reply_tx
                .send(handle)
                .expect("Setup caller dropped reply channel");

            // PHASE 3: Create actor and run
            let mut actor = RasterizerActor {
                response_tx,
                core: RasterCore::new(num_threads),
            };

            log::info!(
                "RasterizerActor started with {} threads",
                actor.core.num_threads
            );

            scheduler.run(&mut actor);
        });

        // Return the setup handle
        let setup_handle = RasterizerSetupHandle::new(setup_tx);
        (setup_handle, join_handle)
    }
}

impl<P: Pixel + Send, Meta> Actor<RenderRequest<P, Meta>, RasterControl, RasterManagement>
    for RasterizerActor<P, Meta>
{
    fn handle_data(&mut self, request: RenderRequest<P, Meta>) -> HandlerResult {
        let out = self.core.step_data(request)?;
        if let Some(response) = out.response {
            // Receiver may be dropped if display was shutdown - that's expected.
            if self.response_tx.send(response).is_err() {
                log::debug!("Render response receiver dropped");
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, ctrl: RasterControl) -> HandlerResult {
        self.core.step_control(ctrl)?;
        Ok(())
    }

    fn handle_management(&mut self, mgmt: RasterManagement) -> HandlerResult {
        self.core.step_management(mgmt)?;
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // No external work to do during park, just wait for messages
        Ok(ActorStatus::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::color::Rgba8;
    use crate::render::frame::Frame;
    use crate::render::Color;
    use actor_scheduler::Message;
    use std::sync::{mpsc, Arc};

    #[test]
    fn rasterizer_actor_basic() {
        // Step 1: Spawn with setup handle
        let (setup_handle, actor_thread) = RasterizerActor::<Rgba8>::spawn_with_setup(1);

        // Step 2: Create response channel and register
        let (response_tx, response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

        // Create a render request (no response_tx field anymore!)
        let frame = Frame::new(64, 64);
        let red = Color::Rgb(255, 0, 0);

        let request = RenderRequest {
            manifold: Arc::new(red),
            frame,
            meta: (),
        };

        // Send render request
        handle
            .send(Message::Data(request))
            .expect("Failed to send render request");

        // Wait for response (comes through our registered channel)
        let response = response_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("Failed to receive response");

        // Verify frame was rendered
        assert_eq!(response.frame.width, 64);
        assert_eq!(response.frame.height, 64);
        assert!(response.render_time.as_nanos() > 0);

        // Shutdown
        handle
            .send(Message::Shutdown)
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }

    #[test]
    fn rasterizer_actor_thread_count_update() {
        // Spawn with setup
        let (setup_handle, actor_thread) = RasterizerActor::<Rgba8>::spawn_with_setup(2);

        // Register response channel
        let (response_tx, _response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

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
            .send(Message::Shutdown)
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }

    #[test]
    fn rasterizer_actor_pause_resume() {
        // Spawn with setup
        let (setup_handle, actor_thread) = RasterizerActor::<Rgba8>::spawn_with_setup(1);

        // Register response channel
        let (response_tx, response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

        // Pause rendering
        handle
            .send(Message::Control(RasterControl::Pause))
            .expect("Failed to send Pause");

        // Send a render request (should be dropped because paused)
        let frame = Frame::new(32, 32);
        let blue = Color::Rgb(0, 0, 255);

        let request = RenderRequest {
            manifold: Arc::new(blue),
            frame,
            meta: (),
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
            .send(Message::Shutdown)
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }
}
