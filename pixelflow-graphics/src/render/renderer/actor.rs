//! Render actor for asynchronous frame rendering.
//!
//! The `RendererActor` provides a message-based interface for parallel frame
//! rendering using the actor-scheduler's three-lane priority system:
//!
//! - **Data Lane**: Frame rendering requests with natural backpressure
//! - **Management Lane**: Thread count updates and configuration queries
//! - **Control Lane**: Shutdown and pause/resume commands
//!
//! # Bootstrap Pattern
//!
//! The actor uses a **bootstrap handshake** to ensure you can't send
//! render requests before registering where responses should go:
//!
//! ```ignore
//! use pixelflow_graphics::render::renderer::RendererActor;
//! use std::sync::mpsc;
//!
//! // Step 1: Spawn with setup handle
//! let (setup_handle, join_handle) = RendererActor::spawn_with_setup(4);
//!
//! // Step 2: Create your response channel
//! let (response_tx, response_rx) = mpsc::channel();
//!
//! // Step 3: Register and get full handle - NOW you can send render requests
//! let renderer = setup_handle.register(response_tx);
//!
//! // Step 4: Send render requests
//! renderer.send(Message::Data(my_render_request)).unwrap();
//!
//! // Step 5: Receive responses
//! let response = response_rx.recv().unwrap();
//! ```

use super::messages::{
    RenderConfig, RenderControl, RenderManagement, RenderRequest, RenderResponse, RendererSetup,
    RendererSetupHandle,
};
use crate::render::Pixel;
use actor_scheduler::mealy::Transducer;
use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, ActorTypes, HandlerError, HandlerResult, SystemStatus,
};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────────────
// RenderCore: the pure decision logic
// ────────────────────────────────────────────────────────────────────────────

/// What a step decides to emit. `Default` (no response) is for the control and management steps,
/// which legitimately emit nothing — mirrors `VsyncCoreOut` in `vsync_actor.rs`.
///
/// **`step_data` never uses the `None` case.** A render always returns its frame, paused or not;
/// see [`RenderCore::render`], where the return type makes that a compile-time obligation rather
/// than a convention.
pub(crate) struct RenderCoreOut<P: Pixel, Meta = ()> {
    pub(crate) response: Option<RenderResponse<P, Meta>>,
}

// Hand-written rather than `#[derive(Default)]`: the derive macro would add a spurious
// `P: Default` bound even though `Option<RenderResponse<P>>` needs no such bound on `P`.
impl<P: Pixel, Meta> Default for RenderCoreOut<P, Meta> {
    fn default() -> Self {
        Self { response: None }
    }
}

/// Pure render decision core: pause/resume, thread-count, and the actual `Scene::render` call,
/// with no channel or bootstrap-handshake machinery in it — a `step_*` call takes a message and
/// returns what to emit, table-testable with no actor scheduler in the loop. Rollout step 3 of
/// `docs/designs/pixelflow-runtime-engine-mesh-migration.md` §5, following the same
/// core/adapter split step 2 established for `vsync_actor.rs`'s `VsyncCore`.
pub(crate) struct RenderCore<P: Pixel, Meta = ()> {
    num_threads: usize,
    paused: bool,
    _pixel: std::marker::PhantomData<(P, Meta)>,
}

impl<P: Pixel, Meta> RenderCore<P, Meta> {
    fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
            paused: false,
            _pixel: std::marker::PhantomData,
        }
    }

    /// Render a request, **always returning its frame**.
    ///
    /// The obligation is in the signature rather than in a comment: this returns
    /// `RenderResponse`, which requires a `Frame`, and the only frame in scope is the one that
    /// arrived in `request`. A paused early-return that dropped the request would leave nothing
    /// to build the return value from, so it does not compile. That matters because the frame is
    /// the caller's sole render buffer — losing it is not a dropped frame, it is a lost
    /// allocation the caller can never recover.
    ///
    /// Infallible on purpose. Rendering writes pixels into a buffer and has no failure mode,
    /// and a `Result` here would reintroduce exactly the hole this signature closes: an `Err`
    /// path carrying no frame.
    fn render(&mut self, request: RenderRequest<P, Meta>) -> RenderResponse<P, Meta> {
        let RenderRequest {
            scene,
            mut frame,
            meta,
        } = request;

        if self.paused {
            log::debug!("Renderer paused; returning the frame unrendered");
            return RenderResponse {
                frame,
                render_time: None,
                meta,
            };
        }

        let start = Instant::now();
        scene.render(&mut frame, self.num_threads);
        let render_time = start.elapsed();

        log::trace!(
            "Rendered {}x{} frame in {:?} ({} threads)",
            frame.width,
            frame.height,
            render_time,
            self.num_threads
        );

        // `meta` is handed straight back, never inspected — see `RenderRequest::meta`.
        RenderResponse {
            frame,
            render_time: Some(render_time),
            meta,
        }
    }

    fn config(&self) -> RenderConfig {
        RenderConfig {
            num_threads: self.num_threads,
            paused: self.paused,
        }
    }
}

impl<P: Pixel, Meta> Transducer for RenderCore<P, Meta> {
    type Control = RenderControl;
    type Management = RenderManagement;
    type Data = RenderRequest<P, Meta>;
    type Out = RenderCoreOut<P, Meta>;

    /// Delegates to [`RenderCore::render`], which is where the frame-return obligation is
    /// enforced. Deliberately a one-liner: there is no branch here that could lose a frame.
    fn step_data(
        &mut self,
        request: RenderRequest<P, Meta>,
    ) -> Result<RenderCoreOut<P, Meta>, HandlerError> {
        Ok(RenderCoreOut {
            response: Some(self.render(request)),
        })
    }

    fn step_control(
        &mut self,
        ctrl: RenderControl,
    ) -> Result<RenderCoreOut<P, Meta>, HandlerError> {
        match ctrl {
            RenderControl::Pause => {
                log::info!("Renderer paused");
                self.paused = true;
            }
            RenderControl::Resume => {
                log::info!("Renderer resumed");
                self.paused = false;
            }
        }
        Ok(RenderCoreOut::default())
    }

    fn step_management(
        &mut self,
        mgmt: RenderManagement,
    ) -> Result<RenderCoreOut<P, Meta>, HandlerError> {
        match mgmt {
            RenderManagement::SetThreadCount(count) => {
                let new_count = count.max(1);
                log::info!(
                    "Renderer thread count updated: {} -> {}",
                    self.num_threads,
                    new_count
                );
                self.num_threads = new_count;
            }
            RenderManagement::GetConfig { response_tx } => {
                // Receiver may be dropped if requester cancelled, that's fine.
                response_tx.send(self.config()).ok();
            }
        }
        Ok(RenderCoreOut::default())
    }
}

#[cfg(test)]
mod core_tests {
    //! `RenderCore` in isolation — no threads, no bootstrap handshake, no scheduler. Mirrors
    //! `vsync_actor.rs`'s `mod tests` for `VsyncCore`.

    use super::*;
    use crate::render::color::Rgba8;
    use crate::render::frame::Frame;
    use crate::render::scene::constant_scene_for;
    use std::sync::mpsc;

    /// Opaque red — a scene has to be *some* colour; which one is not the point.
    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

    fn request(size: u32) -> RenderRequest<Rgba8> {
        RenderRequest {
            scene: constant_scene_for::<Rgba8>(RED, [size, size]),
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

        let mut core = RenderCore::<Rgba8, WindowMeta>::new(1);
        let out = core
            .step_data(RenderRequest {
                scene: constant_scene_for::<Rgba8>(GREEN, [8, 8]),
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
        let mut core = RenderCore::<Rgba8>::new(1);
        let out = core.step_data(request(8)).unwrap();
        let response = out.response.expect("must render when not paused");
        assert_eq!(response.frame.width, 8);
        assert_eq!(response.frame.height, 8);
    }

    #[test]
    fn a_paused_core_returns_the_frame_unrendered() {
        // Previously this asserted the response was `None` — i.e. that pausing *dropped* the
        // request. That destroyed the caller's sole render buffer, which is a lost allocation
        // rather than a skipped frame. The frame now comes back either way; only `render_time`
        // reports whether it was drawn into.
        let mut core = RenderCore::<Rgba8>::new(1);
        core.step_control(RenderControl::Pause).unwrap();

        let out = core.step_data(request(8)).unwrap();
        let response = out
            .response
            .expect("a paused core must still return the caller's frame");
        assert!(
            response.render_time.is_none(),
            "and must report that it did not render"
        );
        assert_eq!(response.frame.width, 8, "the frame comes back intact");
    }

    #[test]
    fn resume_after_pause_allows_rendering_again() {
        let mut core = RenderCore::<Rgba8>::new(1);
        core.step_control(RenderControl::Pause).unwrap();
        core.step_control(RenderControl::Resume).unwrap();
        let out = core.step_data(request(8)).unwrap();
        assert!(out.response.is_some(), "resume must unblock rendering");
    }

    #[test]
    fn set_thread_count_is_reflected_in_config() {
        let mut core = RenderCore::<Rgba8>::new(1);
        core.step_management(RenderManagement::SetThreadCount(4))
            .unwrap();

        let (response_tx, response_rx) = mpsc::channel();
        core.step_management(RenderManagement::GetConfig { response_tx })
            .unwrap();

        let config = response_rx.recv().unwrap();
        assert_eq!(config.num_threads, 4);
        assert!(!config.paused);
    }

    #[test]
    fn zero_thread_count_is_clamped_to_one() {
        let mut core = RenderCore::<Rgba8>::new(1);
        core.step_management(RenderManagement::SetThreadCount(0))
            .unwrap();

        let (response_tx, response_rx) = mpsc::channel();
        core.step_management(RenderManagement::GetConfig { response_tx })
            .unwrap();

        assert_eq!(
            response_rx.recv().unwrap().num_threads,
            1,
            "a thread count of 0 would never render — clamp to at least 1"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RendererActor: the thin adapter — bootstrap handshake, real sends
// ────────────────────────────────────────────────────────────────────────────

/// Render actor for parallel frame rendering.
///
/// Owns exactly what [`RenderCore`] cannot: the response channel and the bootstrap handshake
/// that sets it up. Every decision (pause/resume, thread count, whether to actually render) is
/// [`RenderCore`]'s job; this type only turns its `Out` into the real `response_tx.send(...)`.
///
/// Use [`spawn_with_setup`](Self::spawn_with_setup) to create and start the actor.
pub struct RendererActor<P: Pixel, Meta = ()> {
    /// Channel to send completed frames back. Set during bootstrap.
    response_tx: Sender<RenderResponse<P, Meta>>,
    core: RenderCore<P, Meta>,
}

impl<P: Pixel + Send + 'static, Meta: Send + 'static> ActorTypes for RendererActor<P, Meta> {
    type Data = RenderRequest<P, Meta>;
    type Control = RenderControl;
    type Management = RenderManagement;
}

impl<P: Pixel + Send + 'static, Meta: Send + 'static> RendererActor<P, Meta> {
    /// Spawn the render actor with a bootstrap handshake.
    ///
    /// This is the **primary way** to create a renderer. It spawns the actor
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
    /// - `RendererSetupHandle` - Use this to register your response channel
    /// - `JoinHandle` - The thread handle for the renderer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (setup_handle, _thread) = RendererActor::spawn_with_setup(4);
    /// let (response_tx, response_rx) = std::sync::mpsc::channel();
    /// let renderer = setup_handle.register(response_tx);
    /// // Now you can send render requests via `renderer`
    /// ```
    #[must_use]
    pub fn spawn_with_setup(num_threads: usize) -> (RendererSetupHandle<P, Meta>, JoinHandle<()>) {
        // Create the setup channel (buffer=1, only one setup message ever)
        let (setup_tx, setup_rx) = mpsc::sync_channel::<RendererSetup<P, Meta>>(1);

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
                ActorScheduler::<RenderRequest<P, Meta>, RenderControl, RenderManagement>::new(
                    64, 16,
                );

            // Send the full handle back to the caller
            reply_tx
                .send(handle)
                .expect("Setup caller dropped reply channel");

            // PHASE 3: Create actor and run
            let mut actor = RendererActor {
                response_tx,
                core: RenderCore::new(num_threads),
            };

            log::info!(
                "RendererActor started with {} threads",
                actor.core.num_threads
            );

            scheduler.run(&mut actor);
        });

        // Return the setup handle
        let setup_handle = RendererSetupHandle::new(setup_tx);
        (setup_handle, join_handle)
    }
}

impl<P: Pixel + Send, Meta> Actor<RenderRequest<P, Meta>, RenderControl, RenderManagement>
    for RendererActor<P, Meta>
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

    fn handle_control(&mut self, ctrl: RenderControl) -> HandlerResult {
        self.core.step_control(ctrl)?;
        Ok(())
    }

    fn handle_management(&mut self, mgmt: RenderManagement) -> HandlerResult {
        self.core.step_management(mgmt)?;
        Ok(())
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // No external work to do during handle_os, just wait for messages
        Ok(ActorStatus::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::color::Rgba8;
    use crate::render::frame::Frame;
    use crate::render::scene::constant_scene_for;
    use actor_scheduler::Message;
    use std::sync::mpsc;

    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

    #[test]
    fn render_actor_basic() {
        // Step 1: Spawn with setup handle
        let (setup_handle, actor_thread) = RendererActor::<Rgba8>::spawn_with_setup(1);

        // Step 2: Create response channel and register
        let (response_tx, response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

        // Create a render request (no response_tx field anymore!)
        let frame = Frame::new(64, 64);

        let request = RenderRequest {
            scene: constant_scene_for::<Rgba8>(RED, [64, 64]),
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
        assert!(
            response.render_time.is_some_and(|d| d.as_nanos() > 0),
            "an unpaused renderer reports the time it spent"
        );

        // Shutdown
        handle
            .send(Message::Shutdown)
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }

    #[test]
    fn render_actor_thread_count_update() {
        // Spawn with setup
        let (setup_handle, actor_thread) = RendererActor::<Rgba8>::spawn_with_setup(2);

        // Register response channel
        let (response_tx, _response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

        // Update thread count
        handle
            .send(Message::Management(RenderManagement::SetThreadCount(4)))
            .expect("Failed to send SetThreadCount");

        // Query config
        let (config_tx, config_rx) = mpsc::channel();
        handle
            .send(Message::Management(RenderManagement::GetConfig {
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
    fn render_actor_pause_resume() {
        // Spawn with setup
        let (setup_handle, actor_thread) = RendererActor::<Rgba8>::spawn_with_setup(1);

        // Register response channel
        let (response_tx, response_rx) = mpsc::channel();
        let handle = setup_handle.register(response_tx);

        // Pause rendering
        handle
            .send(Message::Control(RenderControl::Pause))
            .expect("Failed to send Pause");

        // Send a render request (should be dropped because paused)
        let frame = Frame::new(32, 32);

        let request = RenderRequest {
            scene: constant_scene_for::<Rgba8>(BLUE, [32, 32]),
            frame,
            meta: (),
        };

        handle
            .send(Message::Data(request))
            .expect("Failed to send render request");

        // The frame comes back even while paused — losing it would cost the caller its only
        // render buffer. `render_time: None` is how "not drawn into" is reported.
        let response = response_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("a paused renderer must still return the frame");
        assert!(
            response.render_time.is_none(),
            "paused means not rendered, not buffer withheld"
        );

        // Resume
        handle
            .send(Message::Control(RenderControl::Resume))
            .expect("Failed to send Resume");

        // Shutdown
        handle
            .send(Message::Shutdown)
            .expect("Failed to send shutdown");

        actor_thread.join().expect("Actor thread panicked");
    }
}
