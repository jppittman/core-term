//! Engine Troupe - Render pipeline actor coordination using troupe! macro.

use crate::api::private::{EngineControl, EngineData, WindowId};
use crate::api::public::AppManagement;
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt};
use crate::display::platform::PlatformActor;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{VsyncActor, VsyncConfig, VsyncManagement};
use crate::error::RuntimeError;
use actor_scheduler::{Actor, ActorHandle, ActorTypes, Message, ParkHint, TroupeActor};
use pixelflow_graphics::render::{Frame, TensorShape, execute};
use std::time::Instant;

/// Engine handler state - tracks window and rendering state
#[derive(Debug, Clone)]
struct EngineState {
    window_id: WindowId,
    window_width: u32,
    window_height: u32,
    initialized: bool,
    waiting_for_frame: bool,
    last_vsync: Option<Instant>,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            window_id: WindowId::PRIMARY,
            window_width: 800,
            window_height: 600,
            initialized: false,
            waiting_for_frame: false,
            last_vsync: None,
        }
    }
}

/// Engine handler - coordinates app, rendering, display.
pub struct EngineHandler {
    state: EngineState,
    driver_handle: ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>,
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
            EngineData::FromDriver(evt) => {
                self.handle_driver_event(evt);
            }
            EngineData::FromApp(app_data) => {
                self.handle_app_data(app_data);
            }
        }
    }

    fn handle_control(&mut self, ctrl: EngineControl<PlatformPixel>) {
        match ctrl {
            EngineControl::VSync { timestamp, target_timestamp, refresh_interval } => {
                self.handle_vsync(timestamp, target_timestamp, refresh_interval);
            }
            EngineControl::PresentComplete(frame) => {
                self.handle_present_complete(frame);
            }
            EngineControl::UpdateRefreshRate(_) => {
                // TODO: Handle refresh rate updates
            }
            EngineControl::VsyncActorReady(_) => {
                // TODO: Handle vsync ready
            }
            EngineControl::Quit => {
                self.handle_quit();
            }
            EngineControl::DriverAck => {
                // Acknowledge driver message
            }
        }
    }

    fn handle_management(&mut self, mgmt: AppManagement) {
        match mgmt {
            AppManagement::SetTitle(title) => {
                let _ = self.driver_handle.send(Message::Control(DisplayControl::SetTitle {
                    id: self.state.window_id,
                    title,
                }));
            }
            AppManagement::ResizeRequest(width, height) => {
                let _ = self.driver_handle.send(Message::Control(DisplayControl::SetSize {
                    id: self.state.window_id,
                    width,
                    height,
                }));
            }
            AppManagement::CopyToClipboard(text) => {
                let _ = self.driver_handle.send(Message::Control(DisplayControl::Copy { text }));
            }
            AppManagement::RequestPaste => {
                let _ = self.driver_handle.send(Message::Control(DisplayControl::RequestPaste));
            }
            AppManagement::SetCursorIcon(icon) => {
                let _ = self.driver_handle.send(Message::Control(DisplayControl::SetCursor {
                    id: self.state.window_id,
                    cursor: icon,
                }));
            }
            AppManagement::Quit => {
                self.handle_quit();
            }
        }
    }

    fn park(&mut self, hint: ParkHint) -> ParkHint {
        hint
    }
}

impl EngineHandler {
    /// Handle VSync control message - initialize window and signal frame request
    fn handle_vsync(&mut self, _timestamp: Instant, _target_timestamp: Instant, _refresh_interval: std::time::Duration) {
        // Initialize window on first VSync
        if !self.state.initialized {
            let descriptor = crate::api::public::WindowDescriptor {
                width: self.state.window_width,
                height: self.state.window_height,
                title: "Terminal".to_string(),
                resizable: true,
            };
            let _ = self.driver_handle.send(Message::Management(DisplayMgmt::Create {
                id: self.state.window_id,
                settings: descriptor,
            }));
            self.state.initialized = true;
        }

        // TODO: Request frame from app when app handle is available
        // For now, just mark that we're ready for the next frame
        self.state.waiting_for_frame = false;

        self.state.last_vsync = Some(Instant::now());
    }

    /// Handle app data - render manifold to frame and present to driver
    fn handle_app_data(&mut self, app_data: crate::api::public::AppData<PlatformPixel>) {
        self.state.waiting_for_frame = false;

        match app_data {
            crate::api::public::AppData::RenderSurface(manifold) => {
                // Rasterize manifold to frame
                let mut frame = Frame::new(self.state.window_width, self.state.window_height);
                let shape = TensorShape::new(self.state.window_width as usize, self.state.window_height as usize);
                execute(&*manifold, frame.as_slice_mut(), shape);

                // Send to driver
                let _ = self.driver_handle.send(Message::Data(DisplayData::Present {
                    id: self.state.window_id,
                    frame,
                }));
            }
            crate::api::public::AppData::RenderSurfaceU32(manifold) => {
                // Same as RenderSurface but with u32 coordinates
                let mut frame = Frame::new(self.state.window_width, self.state.window_height);
                let shape = TensorShape::new(self.state.window_width as usize, self.state.window_height as usize);
                execute(&*manifold, frame.as_slice_mut(), shape);

                let _ = self.driver_handle.send(Message::Data(DisplayData::Present {
                    id: self.state.window_id,
                    frame,
                }));
            }
            crate::api::public::AppData::Skipped => {
                // No rendering needed
            }
            crate::api::public::AppData::_Phantom(_) => {
                // Phantom data
            }
        }
    }

    /// Handle driver events - process window/input events
    fn handle_driver_event(&mut self, evt: DisplayEvent) {
        match evt {
            DisplayEvent::WindowCreated { width_px, height_px, .. } => {
                self.state.window_width = width_px;
                self.state.window_height = height_px;
            }
            DisplayEvent::Resized { width_px, height_px, .. } => {
                self.state.window_width = width_px;
                self.state.window_height = height_px;
            }
            DisplayEvent::Key { .. } => {
                // Forward to app (TODO: implement when app handle is available)
            }
            DisplayEvent::MouseButtonPress { .. } => {
                // Forward to app (TODO: implement when app handle is available)
            }
            DisplayEvent::MouseButtonRelease { .. } => {
                // Forward to app (TODO: implement when app handle is available)
            }
            DisplayEvent::MouseMove { .. } => {
                // Forward to app (TODO: implement when app handle is available)
            }
            DisplayEvent::MouseScroll { .. } => {
                // Forward to app (TODO: implement when app handle is available)
            }
            DisplayEvent::CloseRequested { .. } => {
                // Notify app of close request (TODO: implement when app handle is available)
            }
            DisplayEvent::FocusGained { .. } => {
                // Notify app (TODO: implement when app handle is available)
            }
            DisplayEvent::FocusLost { .. } => {
                // Notify app (TODO: implement when app handle is available)
            }
            DisplayEvent::PasteData { .. } => {
                // Forward paste data to app (TODO: implement when app handle is available)
            }
            DisplayEvent::ScaleChanged { .. } => {
                // Notify app of scale change (TODO: implement when app handle is available)
            }
            DisplayEvent::ClipboardDataRequested => {
                // Ignore - driver is asking for clipboard data
            }
            DisplayEvent::WindowDestroyed { .. } => {
                // Window was destroyed
            }
        }
    }

    /// Handle frame recycling - frame buffer is ready for reuse
    fn handle_present_complete(&mut self, _frame: Frame<PlatformPixel>) {
        // In a full implementation, we'd return the frame to a pool for reuse
        // For now, we just let it drop
    }

    /// Handle shutdown
    fn handle_quit(&mut self) {
        let _ = self.driver_handle.send(Message::Control(DisplayControl::Shutdown));
    }
}

// Implement TroupeActor for EngineHandler
impl<'a> TroupeActor<'a, Directory> for EngineHandler {
    fn new(dir: &'a Directory) -> Self {
        Self {
            state: EngineState::default(),
            driver_handle: dir.driver.clone(),
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
        let troupe = Self::new();
        let dir = troupe.directory();

        // Configure vsync with target FPS
        dir.vsync
            .send(Message::Management(VsyncManagement::SetConfig {
                config: VsyncConfig {
                    refresh_rate: config.performance.target_fps as f64,
                },
                engine_handle: dir.engine.clone(),
                self_handle: dir.vsync.clone(),
            })).map_err(|e| RuntimeError::InitError(format!("Failed to configure vsync: {}", e)))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use pixelflow_graphics::render::Frame;

    /// Message received by mock driver.
    #[derive(Debug, Clone)]
    enum DriverMessage {
        Create { id: WindowId, descriptor: crate::api::public::WindowDescriptor },
        Present { id: WindowId },
        SetTitle { id: WindowId, title: String },
        SetSize { id: WindowId, width: u32, height: u32 },
        SetCursor { id: WindowId },
        SetVisible { id: WindowId, visible: bool },
        Shutdown,
        Copy { text: String },
        RequestPaste,
        Bell,
    }

    /// Mock driver that captures messages
    struct MockDriver {
        messages: Arc<Mutex<Vec<DriverMessage>>>,
    }

    impl MockDriver {
        fn new() -> (Self, ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>) {
            let messages = Arc::new(Mutex::new(Vec::new()));
            let messages_clone = messages.clone();

            // Create actor scheduler for mock driver
            let (handle, mut scheduler) = actor_scheduler::create_actor::<
                DisplayData<PlatformPixel>,
                DisplayControl,
                DisplayMgmt,
            >(256, None);

            // Spawn the mock driver in a thread
            std::thread::spawn(move || {
                struct InternalMockDriver {
                    messages: Arc<Mutex<Vec<DriverMessage>>>,
                }

                impl Actor<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt> for InternalMockDriver {
                    fn handle_data(&mut self, msg: DisplayData<PlatformPixel>) {
                        match msg {
                            DisplayData::Present { id, frame: _ } => {
                                self.messages.lock().unwrap().push(DriverMessage::Present { id });
                            }
                        }
                    }

                    fn handle_control(&mut self, msg: DisplayControl) {
                        match msg {
                            DisplayControl::SetTitle { id, title } => {
                                self.messages.lock().unwrap().push(DriverMessage::SetTitle { id, title });
                            }
                            DisplayControl::SetSize { id, width, height } => {
                                self.messages.lock().unwrap().push(DriverMessage::SetSize { id, width, height });
                            }
                            DisplayControl::SetCursor { id, cursor: _ } => {
                                self.messages.lock().unwrap().push(DriverMessage::SetCursor { id });
                            }
                            DisplayControl::SetVisible { id, visible } => {
                                self.messages.lock().unwrap().push(DriverMessage::SetVisible { id, visible });
                            }
                            DisplayControl::RequestRedraw { id: _ } => {}
                            DisplayControl::Bell => {
                                self.messages.lock().unwrap().push(DriverMessage::Bell);
                            }
                            DisplayControl::Copy { text } => {
                                self.messages.lock().unwrap().push(DriverMessage::Copy { text });
                            }
                            DisplayControl::RequestPaste => {
                                self.messages.lock().unwrap().push(DriverMessage::RequestPaste);
                            }
                            DisplayControl::Shutdown => {
                                self.messages.lock().unwrap().push(DriverMessage::Shutdown);
                            }
                        }
                    }

                    fn handle_management(&mut self, msg: DisplayMgmt) {
                        match msg {
                            DisplayMgmt::Create { id, settings } => {
                                self.messages.lock().unwrap().push(DriverMessage::Create {
                                    id,
                                    descriptor: settings,
                                });
                            }
                            DisplayMgmt::Destroy { id: _ } => {}
                        }
                    }

                    fn park(&mut self, hint: ParkHint) -> ParkHint {
                        hint
                    }
                }

                let mut driver = InternalMockDriver { messages: messages_clone };
                scheduler.run(&mut driver);
            });

            (MockDriver { messages }, handle)
        }

        fn captured_messages(&self) -> std::sync::MutexGuard<'_, Vec<DriverMessage>> {
            self.messages.lock().unwrap()
        }
    }

    #[test]
    fn test_initialization_on_first_vsync() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        // Trigger first VSync
        engine.handle_vsync(Instant::now(), Instant::now(), Duration::from_millis(16));

        // Give scheduler time to process
        std::thread::sleep(Duration::from_millis(50));

        // Verify window was created
        let messages = mock_driver.captured_messages();
        assert!(messages.iter().any(|m| matches!(m, DriverMessage::Create { .. })));
        assert_eq!(messages.iter().filter(|m| matches!(m, DriverMessage::Create { .. })).count(), 1);
    }

    #[test]
    fn test_initialization_only_once() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        // Trigger multiple VSyncs
        engine.handle_vsync(Instant::now(), Instant::now(), Duration::from_millis(16));
        std::thread::sleep(Duration::from_millis(50));
        engine.handle_vsync(Instant::now(), Instant::now(), Duration::from_millis(16));
        std::thread::sleep(Duration::from_millis(50));
        engine.handle_vsync(Instant::now(), Instant::now(), Duration::from_millis(16));
        std::thread::sleep(Duration::from_millis(50));

        // Verify window was created only once
        let messages = mock_driver.captured_messages();
        assert_eq!(messages.iter().filter(|m| matches!(m, DriverMessage::Create { .. })).count(), 1);
    }

    #[test]
    fn test_vsync_tracks_timestamp() {
        let (_mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        assert!(engine.state.last_vsync.is_none());

        let now = Instant::now();
        engine.handle_vsync(now, now, Duration::from_millis(16));

        assert!(engine.state.last_vsync.is_some());
    }

    #[test]
    fn test_window_resize_updates_state() {
        let (_mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        // Initial size
        assert_eq!(engine.state.window_width, 800);
        assert_eq!(engine.state.window_height, 600);

        // Handle resize event
        engine.handle_driver_event(DisplayEvent::Resized {
            id: WindowId::PRIMARY,
            width_px: 1024,
            height_px: 768,
        });

        // State should be updated
        assert_eq!(engine.state.window_width, 1024);
        assert_eq!(engine.state.window_height, 768);
    }

    #[test]
    fn test_window_created_event_updates_state() {
        let (_mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        engine.handle_driver_event(DisplayEvent::WindowCreated {
            id: WindowId::PRIMARY,
            width_px: 1920,
            height_px: 1080,
            scale: 1.5,
        });

        assert_eq!(engine.state.window_width, 1920);
        assert_eq!(engine.state.window_height, 1080);
    }

    #[test]
    fn test_quit_sends_shutdown_to_driver() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        engine.handle_quit();
        std::thread::sleep(Duration::from_millis(50));

        let messages = mock_driver.captured_messages();
        assert!(messages.iter().any(|m| matches!(m, DriverMessage::Shutdown)));
    }

    #[test]
    fn test_set_title_forwarding() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        engine.handle_management(AppManagement::SetTitle("New Title".to_string()));
        std::thread::sleep(Duration::from_millis(50));

        let messages = mock_driver.captured_messages();
        assert!(messages.iter().any(|m| {
            matches!(m, DriverMessage::SetTitle { title, .. } if title == "New Title")
        }));
    }

    #[test]
    fn test_copy_to_clipboard() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        engine.handle_management(AppManagement::CopyToClipboard("copied text".to_string()));
        std::thread::sleep(Duration::from_millis(50));

        let messages = mock_driver.captured_messages();
        assert!(messages.iter().any(|m| {
            matches!(m, DriverMessage::Copy { text } if text == "copied text")
        }));
    }

    #[test]
    fn test_engine_handler_state_default() {
        let state = EngineState::default();
        assert_eq!(state.window_width, 800);
        assert_eq!(state.window_height, 600);
        assert!(!state.initialized);
        assert!(!state.waiting_for_frame);
        assert_eq!(state.window_id, WindowId::PRIMARY);
    }

    #[test]
    fn test_render_surface_creates_frame() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        // Create a simple manifold that always returns black
        use pixelflow_core::{Discrete, Field, Manifold};
        use std::sync::Arc;

        struct SimpleBlackManifold;
        impl Manifold for SimpleBlackManifold {
            type Output = Discrete;

            fn eval_raw(
                &self,
                _x: Field,
                _y: Field,
                _z: Field,
                _w: Field,
            ) -> Discrete {
                // Black pixel: R=0, G=0, B=0, A=1
                Discrete::pack(Field::from(0.0), Field::from(0.0), Field::from(0.0), Field::from(1.0))
            }
        }

        let manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync> =
            Arc::new(SimpleBlackManifold);

        // Send render surface
        engine.handle_app_data(crate::api::public::AppData::RenderSurface(manifold));
        std::thread::sleep(Duration::from_millis(50));

        // Verify Present was sent
        let messages = mock_driver.captured_messages();
        assert!(messages.iter().any(|m| matches!(m, DriverMessage::Present { .. })));
    }

    #[test]
    fn test_render_surface_u32_creates_frame() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        use pixelflow_core::{Discrete, Field, Manifold};
        use std::sync::Arc;

        struct SimpleBlackManifold;
        impl Manifold for SimpleBlackManifold {
            type Output = Discrete;

            fn eval_raw(
                &self,
                _x: Field,
                _y: Field,
                _z: Field,
                _w: Field,
            ) -> Discrete {
                // Black pixel: R=0, G=0, B=0, A=1
                Discrete::pack(Field::from(0.0), Field::from(0.0), Field::from(0.0), Field::from(1.0))
            }
        }

        let manifold: Arc<dyn Manifold<Output = Discrete> + Send + Sync> =
            Arc::new(SimpleBlackManifold);

        // Send render surface (u32 variant)
        engine.handle_app_data(crate::api::public::AppData::RenderSurfaceU32(manifold));
        std::thread::sleep(Duration::from_millis(50));

        // Verify Present was sent
        let messages = mock_driver.captured_messages();
        assert!(messages.iter().any(|m| matches!(m, DriverMessage::Present { .. })));
    }

    #[test]
    fn test_skipped_frame_no_render() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        // Send skipped frame
        engine.handle_app_data(crate::api::public::AppData::Skipped);
        std::thread::sleep(Duration::from_millis(50));

        // Verify no Present was sent
        let messages = mock_driver.captured_messages();
        assert!(!messages.iter().any(|m| matches!(m, DriverMessage::Present { .. })));
    }

    #[test]
    fn test_resize_request() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        engine.handle_management(AppManagement::ResizeRequest(1024, 768));
        std::thread::sleep(Duration::from_millis(50));

        let messages = mock_driver.captured_messages();
        assert!(messages.iter().any(|m| {
            matches!(m, DriverMessage::SetSize { width: 1024, height: 768, .. })
        }));
    }

    #[test]
    fn test_request_paste() {
        let (mock_driver, driver_handle) = MockDriver::new();

        let mut engine = EngineHandler {
            state: EngineState::default(),
            driver_handle,
        };

        engine.handle_management(AppManagement::RequestPaste);
        std::thread::sleep(Duration::from_millis(50));

        let messages = mock_driver.captured_messages();
        assert!(messages.iter().any(|m| matches!(m, DriverMessage::RequestPaste)));
    }
}
