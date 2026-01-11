//! Engine Troupe - Render pipeline actor coordination using troupe! macro.

use crate::api::private::{EngineControl, EngineData};
use crate::api::public::{
    AppData, AppManagement, Application, EngineEvent, EngineEventControl, EngineEventData,
    EngineEventManagement, WindowId,
};
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt};
use crate::display::platform::PlatformActor;
use crate::error::RuntimeError;
use crate::input::MouseButton;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{
    RenderedResponse, VsyncActor, VsyncCommand, VsyncConfig, VsyncManagement,
};
use actor_scheduler::{Actor, ActorHandle, ActorTypes, Message, ParkHint, TroupeActor};
use pixelflow_core::{Discrete, Manifold};
use pixelflow_graphics::render::rasterizer::{render_work_stealing, RenderOptions, TensorShape};
use pixelflow_graphics::render::Frame;
use std::sync::Arc;
use std::time::Instant;

/// Engine handler - coordinates app, rendering, display.
pub struct EngineHandler {
    /// Handle to the display driver actor.
    driver: ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>,
    /// Handle to the vsync actor (for feedback loop).
    vsync: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
    /// Handle to the application (for event forwarding).
    app_handle: Option<Arc<dyn Application + Send + Sync>>,
    /// Frame counter for VSync feedback.
    frame_number: u64,
    /// Reusable frame buffer (returned by driver after presentation).
    frame_buffer: Option<Frame<PlatformPixel>>,
    /// Number of render threads for work-stealing parallelism.
    render_threads: usize,
}

// ActorTypes impls - required for troupe! macro
impl ActorTypes for EngineHandler {
    type Data = EngineData;
    type Control = EngineControl;
    type Management = AppManagement;
}

impl ActorTypes for DriverActor<ActivePlatform> {
    type Data = DisplayData<PlatformPixel>;
    type Control = DisplayControl;
    type Management = DisplayMgmt;
}

// Generate troupe structures using macro
actor_scheduler::troupe! {
    driver: DriverActor<ActivePlatform> [main],
    engine: EngineHandler [expose],
    vsync: VsyncActor,
}

// Implement Actor for EngineHandler
impl Actor<EngineData, EngineControl, AppManagement>
    for EngineHandler
{
    fn handle_data(&mut self, data: EngineData) {
        match data {
            EngineData::FromApp(app_data) => self.handle_app_data(app_data),
            EngineData::FromDriver(event) => self.handle_driver_event(event),
        }
    }

    fn handle_control(&mut self, ctrl: EngineControl) {
        match ctrl {
            EngineControl::VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            } => {
                // VSync tick - request frame from application
                // App will respond with AppData::RenderSurface which triggers immediate render
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Data(EngineEventData::RequestFrame {
                        timestamp,
                        target_timestamp,
                        refresh_interval,
                    }))
                    .expect("failed to send to app. it probably crashed");
                } else {
                    self.return_vsync_token();
                }
            }
            EngineControl::PresentComplete(frame) => {
                // Driver returned the frame - store it for next render
                self.frame_buffer = Some(frame);

                // Frame is now available - notify VSync that we can render again
                self.vsync
                    .send(Message::Data(RenderedResponse {
                        frame_number: self.frame_number,
                        rendered_at: Instant::now(),
                    }))
                    .expect("Failed to notify VSync of completed frame");
            }
            EngineControl::Quit => {
                self.driver
                    .send(Message::Control(DisplayControl::Shutdown))
                    .expect("Failed to send Shutdown to driver on Quit");
            }
            EngineControl::UpdateRefreshRate(rr) => {
                self.vsync
                    .send(VsyncCommand::UpdateRefreshRate(rr))
                    .expect("failed to update refresh rate");
            }
            EngineControl::VsyncActorReady(handle) => {
                self.vsync = handle;
            }
            EngineControl::DriverAck => {
                unimplemented!("DriverAck not yet implemented");
            }
        }
    }

    fn handle_management(&mut self, mgmt: AppManagement) {
        match mgmt {
            AppManagement::Configure(config) => {
                self.render_threads = config.performance.render_threads;
                log::info!("Engine configured: {} render threads", self.render_threads);
            }
            AppManagement::SetTitle(title) => {
                self.driver
                    .send(Message::Control(DisplayControl::SetTitle {
                        id: WindowId::PRIMARY,
                        title,
                    }))
                    .expect("Failed to relay SetTitle to driver");
            }
            AppManagement::ResizeRequest(width, height) => {
                self.driver
                    .send(Message::Control(DisplayControl::SetSize {
                        id: WindowId::PRIMARY,
                        width,
                        height,
                    }))
                    .expect("Failed to send SetSize to driver");
            }
            AppManagement::CopyToClipboard(text) => {
                self.driver
                    .send(Message::Control(DisplayControl::Copy { text }))
                    .expect("Failed to send Copy to driver");
            }
            AppManagement::RequestPaste => {
                self.driver
                    .send(Message::Control(DisplayControl::RequestPaste))
                    .expect("Failed to send RequestPaste to driver");
            }
            AppManagement::SetCursorIcon(icon) => {
                self.driver
                    .send(Message::Control(DisplayControl::SetCursor {
                        id: WindowId::PRIMARY,
                        cursor: icon,
                    }))
                    .expect("Failed to send SetCursor to driver");
            }
            AppManagement::RegisterApp(app) => {
                log::info!("Application handle registered");
                self.app_handle = Some(app);
            }
            AppManagement::CreateWindow(descriptor) => {
                // Engine assigns the window ID (for now, just use PRIMARY for single window)
                let id = WindowId::PRIMARY;
                log::info!(
                    "Relaying CreateWindow request: assigning id={}, {}x{} \"{}\"",
                    id.0,
                    descriptor.width,
                    descriptor.height,
                    descriptor.title
                );
                self.driver
                    .send(Message::Management(DisplayMgmt::Create {
                        id,
                        settings: descriptor,
                    }))
                    .expect("Failed to relay CreateWindow to driver");
            }
            AppManagement::Quit => {
                self.driver
                    .send(Message::Control(DisplayControl::Shutdown))
                    .expect("Failed to send Shutdown to driver on AppManagement::Quit");
            }
        }
    }

    fn park(&mut self, _hint: ParkHint) -> ParkHint {
        // engine has no external channels which might be busy
        ParkHint::Wait
    }
}

impl EngineHandler {
    /// Handle app data messages (render surfaces, etc.)
    fn handle_app_data(&mut self, app_data: AppData) {
        match app_data {
            AppData::RenderSurface(manifold) | AppData::RenderSurfaceU32(manifold) => {
                // Render if we have a frame buffer available
                if self.frame_buffer.is_some() {
                    self.render_and_present(manifold);
                } else {
                    // No frame buffer unavailable - either window not created yet or frame in-flight
                    // Always return token to avoid leaks
                    log::trace!("Frame buffer unavailable, returning token without rendering");
                    self.return_vsync_token();
                }
            }
            AppData::Skipped => {
                // App says nothing to render - return token
                self.return_vsync_token();
            }
        }
    }

    /// Return a VSync token without rendering (for skipped frames)
    fn return_vsync_token(&mut self) {
        self.vsync
            .send(Message::Data(RenderedResponse {
                frame_number: self.frame_number,
                rendered_at: Instant::now(),
            }))
            .expect("Failed to return VSync token");
    }

    /// Rasterize manifold and present to driver, then notify VSync.
    fn render_and_present(&mut self, manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync>) {
        // Take the frame buffer (driver will return it via PresentComplete)
        let Some(mut frame) = self.frame_buffer.take() else {
            // No frame available - skip this render (waiting for driver to return one)
            log::trace!("Frame buffer unavailable, returning token without rendering");
            self.return_vsync_token();
            return;
        };

        // Rasterize the manifold into the frame (work-stealing parallelism)
        let t0 = Instant::now();
        let shape = TensorShape::new(frame.width, frame.height);
        let options = RenderOptions {
            num_threads: self.render_threads,
        };
        render_work_stealing(&manifold, frame.as_slice_mut(), shape, options);
        let render_time = t0.elapsed();

        // Send to driver for presentation (transfers ownership)
        let t1 = Instant::now();
        self.driver
            .send(Message::Data(DisplayData::Present {
                id: WindowId::PRIMARY,
                frame,
            }))
            .expect("Failed to send frame to driver for presentation");
        let send_time = t1.elapsed();

        self.frame_number += 1;
        if self.frame_number % 60 == 0 {
            log::info!(
                "Frame {}: render={:?}, send={:?}",
                self.frame_number,
                render_time,
                send_time
            );
        }
        // Note: RenderedResponse is sent in PresentComplete handler, not here
        // This ensures VSync only gets a token back when the frame buffer is available
    }

    /// Handle events from the display driver
    fn handle_driver_event(&mut self, event: DisplayEvent) {
        match event {
            DisplayEvent::WindowCreated {
                id,
                width_px,
                height_px,
                scale,
            } => {
                log::info!(
                    "Relaying WindowCreated: id={}, {}x{}, scale={}",
                    id.0,
                    width_px,
                    height_px,
                    scale
                );

                // Allocate initial frame buffer for this window
                self.frame_buffer = Some(Frame::new(width_px, height_px));

                // Relay WindowCreated event to app
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::WindowCreated {
                        id,
                        width_px,
                        height_px,
                        scale,
                    }))
                    .expect("Failed to relay WindowCreated event to app");
                }
            }
            DisplayEvent::Resized {
                id,
                width_px,
                height_px,
            } => {
                log::info!("Relaying Resized: id={}, {}x{}", id.0, width_px, height_px);

                // Reallocate frame buffer for new size
                self.frame_buffer = Some(Frame::new(width_px, height_px));

                // Relay resize event to app
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::Resized {
                        id,
                        width_px,
                        height_px,
                    }))
                    .expect("Failed to relay Resized event to app");
                }
            }
            DisplayEvent::Key {
                symbol,
                modifiers,
                text,
                ..
            } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::KeyDown {
                        key: symbol,
                        mods: modifiers,
                        text,
                    }))
                    .expect("Failed to send KeyDown event to app");
                }
            }
            DisplayEvent::MouseButtonPress { button, x, y, .. } => {
                if let Some(app) = &self.app_handle {
                    let button = convert_mouse_button(button);
                    app.send(EngineEvent::Management(EngineEventManagement::MouseClick {
                        x: x as u32,
                        y: y as u32,
                        button,
                    }))
                    .expect("Failed to send MouseClick event to app");
                }
            }
            DisplayEvent::MouseButtonRelease { button, x, y, .. } => {
                if let Some(app) = &self.app_handle {
                    let button = convert_mouse_button(button);
                    app.send(EngineEvent::Management(
                        EngineEventManagement::MouseRelease {
                            x: x as u32,
                            y: y as u32,
                            button,
                        },
                    ))
                    .expect("Failed to send MouseRelease event to app");
                }
            }
            DisplayEvent::MouseMove {
                x, y, modifiers, ..
            } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::MouseMove {
                        x: x as u32,
                        y: y as u32,
                        mods: modifiers,
                    }))
                    .expect("Failed to send MouseMove event to app");
                }
            }
            DisplayEvent::MouseScroll {
                dx,
                dy,
                x,
                y,
                modifiers,
                ..
            } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(
                        EngineEventManagement::MouseScroll {
                            x: x as u32,
                            y: y as u32,
                            dx,
                            dy,
                            mods: modifiers,
                        },
                    ))
                    .expect("Failed to send MouseScroll event to app");
                }
            }
            DisplayEvent::CloseRequested { .. } => {
                log::info!("Close requested");
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::CloseRequested))
                        .expect("Failed to send CloseRequested event to app");
                }
                self.driver
                    .send(Message::Control(DisplayControl::Shutdown))
                    .expect("Failed to send Shutdown to driver on CloseRequested");
            }
            DisplayEvent::FocusGained { .. } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::FocusGained))
                        .expect("Failed to send FocusGained event to app");
                }
            }
            DisplayEvent::FocusLost { .. } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::FocusLost))
                        .expect("Failed to send FocusLost event to app");
                }
            }
            DisplayEvent::PasteData { text } => {
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Management(EngineEventManagement::Paste(text)))
                        .expect("Failed to send Paste event to app");
                }
            }
            DisplayEvent::ScaleChanged { id, scale } => {
                log::info!("Relaying ScaleChanged: id={}, scale={}", id.0, scale);
                if let Some(app) = &self.app_handle {
                    app.send(EngineEvent::Control(EngineEventControl::ScaleChanged {
                        id,
                        scale,
                    }))
                    .expect("Failed to relay ScaleChanged event to app");
                }
            }
            DisplayEvent::ClipboardDataRequested => {
                // Ignore - driver is asking for clipboard data
            }
            DisplayEvent::WindowDestroyed { .. } => {
                // Window was destroyed
            }
        }
    }
}

/// Convert raw mouse button code to MouseButton enum
fn convert_mouse_button(button: u8) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        _ => MouseButton::Other(button),
    }
}

// Implement TroupeActor for EngineHandler
impl<'a> TroupeActor<'a, Directory> for EngineHandler {
    fn new(dir: &'a Directory) -> Self {
        Self {
            driver: dir.driver.clone(),
            vsync: dir.vsync.clone(),
            app_handle: None,
            frame_number: 0,
            frame_buffer: None,
            render_threads: 1, // Default, will be set by Configure message
        }
    }
}

// Implement TroupeActor for DriverActor
impl<'a> TroupeActor<'a, Directory> for DriverActor<ActivePlatform> {
    fn new(dir: &'a Directory) -> Self {
        #[cfg(target_os = "macos")]
        {
            use crate::platform::MetalOps;
            let ops = MetalOps::new(dir.engine.clone()).expect("Failed to create Metal ops");
            let platform = PlatformActor::new(ops);
            DriverActor::new(platform)
        }
        #[cfg(target_os = "linux")]
        {
            use crate::platform::linux::LinuxOps;
            let ops = LinuxOps::new(dir.engine.clone()).expect("Failed to create Linux ops");
            let platform = PlatformActor::new(ops);
            DriverActor::new(platform)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            panic!("Unsupported platform");
        }
    }
}

impl Troupe {
    /// Create troupe and configure vsync actor.
    pub fn with_config(config: EngineConfig) -> Result<Self, RuntimeError> {
        // Create troupe with platform-specific waker for the main (driver) actor
        #[cfg(target_os = "macos")]
        let troupe = {
            use crate::platform::waker::CocoaWaker;
            Self::new_with_waker(Some(std::sync::Arc::new(CocoaWaker::new())))
        };
        #[cfg(not(target_os = "macos"))]
        let troupe = Self::new();
        let dir = troupe.directory();

        // Configure the engine with window settings
        dir.engine
            .send(Message::Management(AppManagement::Configure(
                config.clone(),
            )))
            .map_err(|e| RuntimeError::InitError(format!("Failed to configure engine: {}", e)))?;

        // Configure vsync with target FPS
        dir.vsync
            .send(Message::Management(VsyncManagement::SetConfig {
                config: VsyncConfig {
                    refresh_rate: config.performance.target_fps as f64,
                },
                engine_handle: dir.engine.clone(),
                self_handle: dir.vsync.clone(),
            }))
            .map_err(|e| RuntimeError::InitError(format!("Failed to configure vsync: {}", e)))?;

        Ok(troupe)
    }

    /// Get an unregistered engine handle.
    ///
    /// You must call `register()` on this handle before you can use the engine.
    /// This ensures proper initialization (app registration + window creation).
    pub fn engine_handle(&self) -> crate::api::public::UnregisteredEngineHandle {
        crate::api::public::UnregisteredEngineHandle::new(self.directory().engine.clone())
    }
}

// TODO: Tests need updating for new EngineHandler struct
// The upstream tests used EngineState which we replaced with direct fields + vsync feedback
// Tests should be re-enabled once the struct is stabilized
