//! Engine handler tests - comprehensive testing of frame lifecycle coordination

use super::*;
use crate::api::private::{DisplayEvent, DriverCommand, EngineControl, EngineData};
use crate::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use crate::vsync_actor::RenderedResponse;
use actor_scheduler::Actor;
use pixelflow_render::{Frame, Rgba};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

// ============================================================================
// Mock Infrastructure
// ============================================================================

/// Mock app that records all events it receives
#[derive(Clone)]
struct MockApp {
    events_tx: Sender<EngineEvent>,
}

impl MockApp {
    fn new() -> (Self, Receiver<EngineEvent>) {
        let (tx, rx) = channel();
        (Self { events_tx: tx }, rx)
    }
}

impl crate::api::public::Application for MockApp {
    fn send(&self, event: EngineEvent) -> Result<(), actor_scheduler::SendError> {
        self.events_tx
            .send(event)
            .map_err(|_| actor_scheduler::SendError)
    }
}

/// Mock driver that records all commands
#[derive(Clone)]
struct MockDriver {
    commands_tx: Sender<DriverCommand<Rgba>>,
}

impl MockDriver {
    fn new() -> (Self, Receiver<DriverCommand<Rgba>>) {
        let (tx, rx) = channel();
        (Self { commands_tx: tx }, rx)
    }

    fn send(&self, cmd: DriverCommand<Rgba>) -> anyhow::Result<()> {
        self.commands_tx
            .send(cmd)
            .map_err(|e| anyhow::anyhow!("Mock driver send failed: {:?}", e))
    }
}

impl crate::display::driver::DisplayDriver for MockDriver {
    type Pixel = Rgba;

    fn new(_handle: crate::api::private::EngineActorHandle<Rgba>) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self::new().0)
    }

    fn send(&self, cmd: DriverCommand<Rgba>) -> anyhow::Result<()> {
        self.send(cmd)
    }

    fn run(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Mock VSync actor that records RenderedResponse messages
#[derive(Clone)]
struct MockVSync {
    responses_tx: Sender<RenderedResponse>,
}

impl MockVSync {
    fn new() -> (Self, Receiver<RenderedResponse>) {
        let (tx, rx) = channel();
        (Self { responses_tx: tx }, rx)
    }

    fn send(&self, response: RenderedResponse) -> Result<(), std::sync::mpsc::SendError<RenderedResponse>> {
        self.responses_tx.send(response)
    }
}

impl crate::vsync_actor::VsyncHandle for MockVSync {
    fn send(&self, response: RenderedResponse) -> Result<(), std::sync::mpsc::SendError<RenderedResponse>> {
        self.send(response)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simple test surface for rendering
fn create_test_surface() -> Rgba {
    Rgba(0xFF0000FF) // Red (implements Surface trait)
}

/// Wait for an event with timeout
fn wait_for_event(rx: &Receiver<EngineEvent>, timeout: Duration) -> Option<EngineEvent> {
    rx.recv_timeout(timeout).ok()
}

/// Wait for a driver command with timeout
fn wait_for_command(rx: &Receiver<DriverCommand<Rgba>>, timeout: Duration) -> Option<DriverCommand<Rgba>> {
    rx.recv_timeout(timeout).ok()
}

/// Wait for a VSync response with timeout
fn wait_for_response(rx: &Receiver<RenderedResponse>, timeout: Duration) -> Option<RenderedResponse> {
    rx.recv_timeout(timeout).ok()
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_vsync_forwards_request_frame_to_app() {
    // Setup mocks
    let (mock_app, app_rx) = MockApp::new();
    let (mock_driver, _driver_rx) = MockDriver::new();
    let (engine_handle, mut scheduler) = crate::api::private::create_engine_actor::<Rgba>(None);

    // Create handler
    let mut handler: EngineHandler<_, _, MockVSync> = EngineHandler {
        app: mock_app,
        engine_handle: engine_handle.clone(),
        driver: mock_driver,
        framebuffer: None,
        physical_width: 800,
        physical_height: 600,
        scale_factor: 1.0,
        vsync_actor: None,
        render_threads: 1,
    };

    // Send VSync control message
    let timestamp = Instant::now();
    let target = timestamp + Duration::from_millis(16);
    handler.handle_control(EngineControl::VSync {
        timestamp,
        target_timestamp: target,
        refresh_interval: Duration::from_millis(16),
    });

    // App should receive RequestFrame event
    let event = wait_for_event(&app_rx, Duration::from_millis(100))
        .expect("App should receive RequestFrame");

    match event {
        EngineEvent::Data(EngineEventData::RequestFrame {
            timestamp: ts,
            target_timestamp: target_ts,
            refresh_interval: interval,
        }) => {
            assert_eq!(ts, timestamp);
            assert_eq!(target_ts, target);
            assert_eq!(interval, Duration::from_millis(16));
        }
        _ => panic!("Expected RequestFrame event, got {:?}", event),
    }
}

#[test]
fn test_render_surface_sends_present_to_driver() {
    // Setup mocks
    let (mock_app, _app_rx) = MockApp::new();
    let (mock_driver, driver_rx) = MockDriver::new();
    let (engine_handle, mut scheduler) = crate::api::private::create_engine_actor::<Rgba>(None);

    // Create handler
    let mut handler: EngineHandler<_, _, MockVSync> = EngineHandler {
        app: mock_app,
        engine_handle: engine_handle.clone(),
        driver: mock_driver,
        framebuffer: None,
        physical_width: 800,
        physical_height: 600,
        scale_factor: 1.0,
        vsync_actor: None,
        render_threads: 1,
    };

    // Send RenderSurface from app
    let surface = create_test_surface();
    handler.handle_data(EngineData::FromApp(AppData::RenderSurface(Box::new(surface))));

    // Driver should receive Present command
    let command = wait_for_command(&driver_rx, Duration::from_millis(100))
        .expect("Driver should receive Present command");

    match command {
        DriverCommand::Present { id, frame } => {
            assert_eq!(id, WindowId::PRIMARY);
            assert_eq!(frame.width, 800);
            assert_eq!(frame.height, 600);
            // Frame should be rendered with red pixels
            assert_eq!(frame.as_slice()[0], Rgba(0xFF0000FF));
        }
        _ => panic!("Expected Present command, got {:?}", command),
    }
}

#[test]
fn test_present_complete_sends_rendered_response() {
    // Setup mocks
    let (mock_app, _app_rx) = MockApp::new();
    let (mock_driver, _driver_rx) = MockDriver::new();
    let (mock_vsync, vsync_rx) = MockVSync::new();
    let (engine_handle, mut scheduler) = crate::api::private::create_engine_actor::<Rgba>(None);

    // Create handler with VSync
    let mut handler = EngineHandler {
        app: mock_app,
        engine_handle: engine_handle.clone(),
        driver: mock_driver,
        framebuffer: None,
        physical_width: 800,
        physical_height: 600,
        scale_factor: 1.0,
        vsync_actor: Some(mock_vsync),
        render_threads: 1,
    };

    // Create a frame
    let frame = Frame::new(800, 600);

    // Send PresentComplete
    handler.handle_control(EngineControl::PresentComplete(frame));

    // VSync should receive RenderedResponse
    let response = wait_for_response(&vsync_rx, Duration::from_millis(100))
        .expect("VSync should receive RenderedResponse");

    // Response is a unit struct, just verify we got it
    assert_eq!(format!("{:?}", response), "RenderedResponse");
}

#[test]
fn test_full_frame_lifecycle() {
    // Setup mocks
    let (mock_app, app_rx) = MockApp::new();
    let (mock_driver, driver_rx) = MockDriver::new();
    let (mock_vsync, vsync_rx) = MockVSync::new();
    let (engine_handle, mut scheduler) = crate::api::private::create_engine_actor::<Rgba>(None);

    // Create handler
    let mut handler = EngineHandler {
        app: mock_app,
        engine_handle: engine_handle.clone(),
        driver: mock_driver,
        framebuffer: None,
        physical_width: 800,
        physical_height: 600,
        scale_factor: 1.0,
        vsync_actor: Some(mock_vsync),
        render_threads: 1,
    };

    // STEP 1: VSync sends tick
    let timestamp = Instant::now();
    handler.handle_control(EngineControl::VSync {
        timestamp,
        target_timestamp: timestamp + Duration::from_millis(16),
        refresh_interval: Duration::from_millis(16),
    });

    // STEP 2: App should receive RequestFrame
    let event = wait_for_event(&app_rx, Duration::from_millis(100))
        .expect("App should receive RequestFrame");
    assert!(matches!(
        event,
        EngineEvent::Data(EngineEventData::RequestFrame { .. })
    ));

    // STEP 3: App sends RenderSurface
    let surface = create_test_surface();
    handler.handle_data(EngineData::FromApp(AppData::RenderSurface(Box::new(surface))));

    // STEP 4: Driver should receive Present
    let command = wait_for_command(&driver_rx, Duration::from_millis(100))
        .expect("Driver should receive Present");
    let frame = match command {
        DriverCommand::Present { frame, .. } => frame,
        _ => panic!("Expected Present command"),
    };

    // STEP 5: Driver sends PresentComplete
    handler.handle_control(EngineControl::PresentComplete(frame));

    // STEP 6: VSync should receive RenderedResponse
    wait_for_response(&vsync_rx, Duration::from_millis(100))
        .expect("VSync should receive RenderedResponse");
}

#[test]
fn test_window_created_forwards_resize_to_app() {
    // Setup mocks
    let (mock_app, app_rx) = MockApp::new();
    let (mock_driver, _driver_rx) = MockDriver::new();
    let (engine_handle, mut scheduler) = crate::api::private::create_engine_actor::<Rgba>(None);

    // Create handler
    let mut handler: EngineHandler<_, _, MockVSync> = EngineHandler {
        app: mock_app,
        engine_handle: engine_handle.clone(),
        driver: mock_driver,
        framebuffer: None,
        physical_width: 0,
        physical_height: 0,
        scale_factor: 1.0,
        vsync_actor: None,
        render_threads: 1,
    };

    // Send WindowCreated event from driver
    handler.handle_data(EngineData::FromDriver(DisplayEvent::WindowCreated {
        id: WindowId::PRIMARY,
        width_px: 1600,
        height_px: 1200,
        scale: 2.0,
    }));

    // App should receive Resize event with logical pixels
    let event = wait_for_event(&app_rx, Duration::from_millis(100))
        .expect("App should receive Resize event");

    match event {
        EngineEvent::Control(EngineEventControl::Resize(w, h)) => {
            assert_eq!(w, 800); // 1600 / 2.0
            assert_eq!(h, 600); // 1200 / 2.0
        }
        _ => panic!("Expected Resize event, got {:?}", event),
    }

    // Handler should track physical dimensions
    assert_eq!(handler.physical_width, 1600);
    assert_eq!(handler.physical_height, 1200);
    assert_eq!(handler.scale_factor, 2.0);
}

#[test]
fn test_framebuffer_reuse() {
    // Setup mocks
    let (mock_app, _app_rx) = MockApp::new();
    let (mock_driver, driver_rx) = MockDriver::new();
    let (engine_handle, mut scheduler) = crate::api::private::create_engine_actor::<Rgba>(None);

    // Create handler
    let mut handler: EngineHandler<_, _, MockVSync> = EngineHandler {
        app: mock_app,
        engine_handle: engine_handle.clone(),
        driver: mock_driver,
        framebuffer: None,
        physical_width: 800,
        physical_height: 600,
        scale_factor: 1.0,
        vsync_actor: None,
        render_threads: 1,
    };

    // First render
    let surface1 = create_test_surface();
    handler.handle_data(EngineData::FromApp(AppData::RenderSurface(Box::new(surface1))));
    let frame1 = match wait_for_command(&driver_rx, Duration::from_millis(100)).unwrap() {
        DriverCommand::Present { frame, .. } => frame,
        _ => panic!("Expected Present"),
    };

    // PresentComplete returns frame
    handler.handle_control(EngineControl::PresentComplete(frame1));

    // Framebuffer should be stored
    assert!(handler.framebuffer.is_some());

    // Second render should reuse framebuffer
    let surface2 = create_test_surface();
    handler.handle_data(EngineData::FromApp(AppData::RenderSurface(Box::new(surface2))));

    // Frame should be reused (same allocation)
    let frame2 = match wait_for_command(&driver_rx, Duration::from_millis(100)).unwrap() {
        DriverCommand::Present { frame, .. } => frame,
        _ => panic!("Expected Present"),
    };

    assert_eq!(frame2.width, 800);
    assert_eq!(frame2.height, 600);
}
