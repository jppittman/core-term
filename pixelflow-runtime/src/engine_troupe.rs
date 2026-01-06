//! Engine Troupe - Render pipeline actor coordination using troupe! macro.

use crate::api::private::{EngineControl, EngineData, WindowId};
use crate::api::public::{
    AppData, AppManagement, Application, EngineEvent, EngineEventControl, EngineEventManagement,
    WindowDescriptor,
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
    app_handle: Option<Box<dyn Application>>,
    /// Whether we've created a window yet.
    window_created: bool,
    /// Current window width in pixels.
    width: u32,
    /// Current window height in pixels.
    height: u32,
    /// Window title.
    title: String,
    /// Current manifold to render (if any).
    current_manifold: Option<Arc<dyn Manifold<Output = Discrete> + Send + Sync>>,
    /// Frame counter for VSync feedback.
    frame_number: u64,
    /// Reusable frame buffer (returned by driver after presentation).
    frame_buffer: Option<Frame<PlatformPixel>>,
    /// Number of render threads for work-stealing parallelism.
    render_threads: usize,
}

// ActorTypes impls - required for troupe! macro
impl ActorTypes for EngineHandler {
    type Data = EngineData<PlatformPixel>;
    type Control = EngineControl<PlatformPixel>;
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
impl Actor<EngineData<PlatformPixel>, EngineControl<PlatformPixel>, AppManagement>
    for EngineHandler
{
    fn handle_data(&mut self, data: EngineData<PlatformPixel>) {
        match data {
            EngineData::FromApp(app_data) => self.handle_app_data(app_data),
            EngineData::FromDriver(event) => self.handle_driver_event(event),
        }
    }

    fn handle_control(&mut self, ctrl: EngineControl<PlatformPixel>) {
        match ctrl {
            EngineControl::VSync { .. } => {
                // VSync tick - re-render if we have a manifold and a frame buffer
                if let Some(ref manifold) = self.current_manifold {
                    if self.window_created && self.frame_buffer.is_some() {
                        self.render_and_present(manifold.clone());
                    }
                }
            }
            EngineControl::PresentComplete(frame) => {
                // Driver returned the frame - check if size still matches
                if frame.width == self.width as usize && frame.height == self.height as usize {
                    self.frame_buffer = Some(frame);
                } else {
                    // Window was resized while frame was in-flight, reallocate
                    log::debug!(
                        "Frame size mismatch: {}x{} vs {}x{}, reallocating",
                        frame.width,
                        frame.height,
                        self.width,
                        self.height
                    );
                    self.frame_buffer = Some(Frame::new(self.width, self.height));
                }
                // Frame is now available - notify VSync that we can render again
                // This is the proper place to add a token, not after render_and_present
                let _ = self.vsync.send(Message::Data(RenderedResponse {
                    frame_number: self.frame_number,
                    rendered_at: Instant::now(),
                }));
            }
            EngineControl::Quit => {
                let _ = self.driver.send(Message::Control(DisplayControl::Shutdown));
            }
            EngineControl::UpdateRefreshRate(_) => {
                unimplemented!("UpdateRefreshRate not yet implemented");
            }
            EngineControl::VsyncActorReady(_) => {
                unimplemented!("VsyncActorReady not yet implemented");
            }
            EngineControl::DriverAck => {
                unimplemented!("DriverAck not yet implemented");
            }
        }
    }

    fn handle_management(&mut self, mgmt: AppManagement) {
        match mgmt {
            AppManagement::Configure(config) => {
                self.width = config.window.width;
                self.height = config.window.height;
                self.title = config.window.title.clone();
                self.render_threads = config.performance.render_threads;
                log::info!(
                    "Engine configured: {}x{} \"{}\" ({} render threads)",
                    self.width,
                    self.height,
                    self.title,
                    self.render_threads
                );
            }
            AppManagement::SetTitle(title) => {
                self.title = title.clone();
                let _ = self.driver.send(Message::Control(DisplayControl::SetTitle {
                    id: WindowId::PRIMARY,
                    title,
                }));
            }
            AppManagement::ResizeRequest(width, height) => {
                let _ = self.driver.send(Message::Control(DisplayControl::SetSize {
                    id: WindowId::PRIMARY,
                    width,
                    height,
                }));
            }
            AppManagement::CopyToClipboard(text) => {
                let _ = self
                    .driver
                    .send(Message::Control(DisplayControl::Copy { text }));
            }
            AppManagement::RequestPaste => {
                let _ = self
                    .driver
                    .send(Message::Control(DisplayControl::RequestPaste));
            }
            AppManagement::SetCursorIcon(icon) => {
                let _ = self
                    .driver
                    .send(Message::Control(DisplayControl::SetCursor {
                        id: WindowId::PRIMARY,
                        cursor: icon,
                    }));
            }
            AppManagement::Quit => {
                let _ = self.driver.send(Message::Control(DisplayControl::Shutdown));
            }
        }
    }

    fn park(&mut self, hint: ParkHint) -> ParkHint {
        hint
    }
}

impl EngineHandler {
    /// Handle app data messages (render surfaces, etc.)
    fn handle_app_data(&mut self, app_data: AppData<PlatformPixel>) {
        match app_data {
            AppData::RenderSurface(manifold) | AppData::RenderSurfaceU32(manifold) => {
                // Store the manifold for VSync-driven re-rendering
                self.current_manifold = Some(manifold);

                // Create window and start VSync if this is the first frame
                if !self.window_created {
                    log::info!(
                        "Creating window: {}x{} \"{}\"",
                        self.width,
                        self.height,
                        self.title
                    );
                    let _ = self.driver.send(Message::Management(DisplayMgmt::Create {
                        id: WindowId::PRIMARY,
                        settings: WindowDescriptor {
                            width: self.width,
                            height: self.height,
                            title: self.title.clone(),
                            resizable: true,
                        },
                    }));
                    self.window_created = true;

                    // Allocate the initial frame buffer
                    self.frame_buffer = Some(Frame::new(self.width, self.height));

                    // Start VSync now that we have content to render
                    let _ = self.vsync.send(Message::Control(VsyncCommand::Start));
                    log::info!("VSync started - first frame will render on next tick");
                }
                // Don't render immediately - let VSync tick trigger the render
            }
            AppData::Skipped => {
                // No rendering needed
            }
            AppData::_Phantom(_) => {}
        }
    }

    /// Rasterize manifold and present to driver, then notify VSync.
    fn render_and_present(&mut self, manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync>) {
        // Take the frame buffer (driver will return it via PresentComplete)
        let Some(mut frame) = self.frame_buffer.take() else {
            // No frame available - skip this render (waiting for driver to return one)
            return;
        };

        // Rasterize the manifold into the frame (work-stealing parallelism)
        let t0 = Instant::now();
        let shape = TensorShape::new(self.width as usize, self.height as usize);
        let options = RenderOptions {
            num_threads: self.render_threads,
        };
        render_work_stealing(&manifold, frame.as_slice_mut(), shape, options);
        let render_time = t0.elapsed();

        // Send to driver for presentation (transfers ownership)
        let t1 = Instant::now();
        let _ = self.driver.send(Message::Data(DisplayData::Present {
            id: WindowId::PRIMARY,
            frame,
        }));
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
                width_px,
                height_px,
                ..
            } => {
                log::info!("Window created: {}x{}", width_px, height_px);
                self.width = width_px;
                self.height = height_px;
            }
            DisplayEvent::Resized {
                width_px,
                height_px,
                ..
            } => {
                log::info!("Window resized: {}x{}", width_px, height_px);
                self.width = width_px;
                self.height = height_px;
                // Reallocate frame buffer for new size
                if self.frame_buffer.is_some() {
                    self.frame_buffer = Some(Frame::new(width_px, height_px));
                }
                // Forward resize event to app
                if let Some(app) = &self.app_handle {
                    let _ = app.send(EngineEvent::Control(EngineEventControl::Resize(
                        width_px, height_px,
                    )));
                }
            }
            DisplayEvent::Key {
                symbol,
                modifiers,
                text,
                ..
            } => {
                if let Some(app) = &self.app_handle {
                    let _ = app.send(EngineEvent::Management(EngineEventManagement::KeyDown {
                        key: symbol,
                        mods: modifiers,
                        text,
                    }));
                }
            }
            DisplayEvent::MouseButtonPress { button, x, y, .. } => {
                if let Some(app) = &self.app_handle {
                    let button = convert_mouse_button(button);
                    let _ = app.send(EngineEvent::Management(EngineEventManagement::MouseClick {
                        x: x as u32,
                        y: y as u32,
                        button,
                    }));
                }
            }
            DisplayEvent::MouseButtonRelease { button, x, y, .. } => {
                if let Some(app) = &self.app_handle {
                    let button = convert_mouse_button(button);
                    let _ = app.send(EngineEvent::Management(
                        EngineEventManagement::MouseRelease {
                            x: x as u32,
                            y: y as u32,
                            button,
                        },
                    ));
                }
            }
            DisplayEvent::MouseMove {
                x, y, modifiers, ..
            } => {
                if let Some(app) = &self.app_handle {
                    let _ = app.send(EngineEvent::Management(EngineEventManagement::MouseMove {
                        x: x as u32,
                        y: y as u32,
                        mods: modifiers,
                    }));
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
                    let _ = app.send(EngineEvent::Management(
                        EngineEventManagement::MouseScroll {
                            x: x as u32,
                            y: y as u32,
                            dx,
                            dy,
                            mods: modifiers,
                        },
                    ));
                }
            }
            DisplayEvent::CloseRequested { .. } => {
                log::info!("Close requested");
                if let Some(app) = &self.app_handle {
                    let _ = app.send(EngineEvent::Control(EngineEventControl::CloseRequested));
                }
                let _ = self.driver.send(Message::Control(DisplayControl::Shutdown));
            }
            DisplayEvent::FocusGained { .. } => {
                if let Some(app) = &self.app_handle {
                    let _ = app.send(EngineEvent::Management(EngineEventManagement::FocusGained));
                }
            }
            DisplayEvent::FocusLost { .. } => {
                if let Some(app) = &self.app_handle {
                    let _ = app.send(EngineEvent::Management(EngineEventManagement::FocusLost));
                }
            }
            DisplayEvent::PasteData { text } => {
                if let Some(app) = &self.app_handle {
                    let _ = app.send(EngineEvent::Management(EngineEventManagement::Paste(text)));
                }
            }
            DisplayEvent::ScaleChanged { scale, .. } => {
                if let Some(app) = &self.app_handle {
                    let _ = app.send(EngineEvent::Control(EngineEventControl::ScaleChanged(
                        scale,
                    )));
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
            window_created: false,
            width: 800, // Default, will be set by Configure message
            height: 600,
            title: "PixelFlow".to_string(),
            current_manifold: None,
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

    /// Get a handle to the engine actor for external components to communicate with.
    pub fn engine_handle(
        &self,
    ) -> actor_scheduler::ActorHandle<
        EngineData<PlatformPixel>,
        EngineControl<PlatformPixel>,
        AppManagement,
    > {
        self.directory().engine.clone()
    }
}

// TODO: Tests need updating for new EngineHandler struct
// The upstream tests used EngineState which we replaced with direct fields + vsync feedback
// Tests should be re-enabled once the struct is stabilized
