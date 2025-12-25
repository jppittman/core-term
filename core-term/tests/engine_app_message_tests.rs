//! Engine ↔ App Message CUJ Tests (Phase 2)
//!
//! These tests verify the message flows between the Engine and TerminalApp.
//! Each test covers a specific CUJ identified in MESSAGE_CUJ_COVERAGE.md.

use actor_scheduler::{Actor, ActorScheduler, Message, ParkHint};
use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::Discrete;
use pixelflow_runtime::api::public::{
    AppData, EngineEventControl, EngineEventData, EngineEventManagement,
};
use pixelflow_runtime::input::{KeySymbol, Modifiers, MouseButton};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// Test Fixtures
// =============================================================================

/// Mock terminal app that records engine events and can respond with render surfaces
struct MockTerminalApp {
    control_events: Arc<std::sync::Mutex<Vec<EngineEventControl>>>,
    management_events: Arc<std::sync::Mutex<Vec<EngineEventManagement>>>,
    data_events: Arc<std::sync::Mutex<Vec<EngineEventData>>>,
    pty_writes: Arc<std::sync::Mutex<Vec<Vec<u8>>>>,
    resize_count: Arc<AtomicUsize>,
    close_requested: Arc<AtomicBool>,
    current_width: Arc<AtomicU32>,
    current_height: Arc<AtomicU32>,
    current_scale: Arc<std::sync::Mutex<f64>>,
}

impl MockTerminalApp {
    fn new() -> (Self, MockTerminalAppState) {
        let control_events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let management_events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let data_events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let pty_writes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let resize_count = Arc::new(AtomicUsize::new(0));
        let close_requested = Arc::new(AtomicBool::new(false));
        let current_width = Arc::new(AtomicU32::new(800));
        let current_height = Arc::new(AtomicU32::new(600));
        let current_scale = Arc::new(std::sync::Mutex::new(1.0));

        let state = MockTerminalAppState {
            control_events: control_events.clone(),
            management_events: management_events.clone(),
            data_events: data_events.clone(),
            pty_writes: pty_writes.clone(),
            resize_count: resize_count.clone(),
            close_requested: close_requested.clone(),
            current_width: current_width.clone(),
            current_height: current_height.clone(),
            current_scale: current_scale.clone(),
        };

        (
            Self {
                control_events,
                management_events,
                data_events,
                pty_writes,
                resize_count,
                close_requested,
                current_width,
                current_height,
                current_scale,
            },
            state,
        )
    }
}

/// External state handle for test assertions
#[derive(Clone)]
struct MockTerminalAppState {
    control_events: Arc<std::sync::Mutex<Vec<EngineEventControl>>>,
    management_events: Arc<std::sync::Mutex<Vec<EngineEventManagement>>>,
    data_events: Arc<std::sync::Mutex<Vec<EngineEventData>>>,
    pty_writes: Arc<std::sync::Mutex<Vec<Vec<u8>>>>,
    resize_count: Arc<AtomicUsize>,
    close_requested: Arc<AtomicBool>,
    current_width: Arc<AtomicU32>,
    current_height: Arc<AtomicU32>,
    current_scale: Arc<std::sync::Mutex<f64>>,
}

impl Actor<EngineEventData, EngineEventControl, EngineEventManagement> for MockTerminalApp {
    fn handle_data(&mut self, data: EngineEventData) {
        self.data_events.lock().unwrap().push(data);
    }

    fn handle_control(&mut self, ctrl: EngineEventControl) {
        match &ctrl {
            EngineEventControl::Resize(w, h) => {
                self.current_width.store(*w, Ordering::SeqCst);
                self.current_height.store(*h, Ordering::SeqCst);
                self.resize_count.fetch_add(1, Ordering::SeqCst);
            }
            EngineEventControl::CloseRequested => {
                self.close_requested.store(true, Ordering::SeqCst);
            }
            EngineEventControl::ScaleChanged(scale) => {
                *self.current_scale.lock().unwrap() = *scale;
            }
        }
        self.control_events.lock().unwrap().push(ctrl);
    }

    fn handle_management(&mut self, mgmt: EngineEventManagement) {
        match &mgmt {
            EngineEventManagement::KeyDown { text, .. } => {
                if let Some(t) = text {
                    self.pty_writes.lock().unwrap().push(t.as_bytes().to_vec());
                }
            }
            EngineEventManagement::Paste(content) => {
                self.pty_writes
                    .lock()
                    .unwrap()
                    .push(content.as_bytes().to_vec());
            }
            _ => {}
        }
        self.management_events.lock().unwrap().push(mgmt);
    }

    fn park(&mut self, _: ParkHint) -> ParkHint {
        ParkHint::Wait
    }
}

/// Helper to spawn a mock app and return handles
fn spawn_mock_app() -> (
    actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>,
    MockTerminalAppState,
    thread::JoinHandle<()>,
) {
    let (app, state) = MockTerminalApp::new();
    let (tx, mut rx) =
        ActorScheduler::<EngineEventData, EngineEventControl, EngineEventManagement>::new(64, 32);

    let handle = thread::spawn(move || {
        let mut app = app;
        rx.run(&mut app);
    });

    (tx, state, handle)
}

// =============================================================================
// ENG-01: Resize Message Tests
// =============================================================================

#[test]
fn cuj_eng01_resize_updates_terminal_dimensions() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: A resize event is sent
    tx.send(Message::Control(EngineEventControl::Resize(1920, 1080)))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Terminal dimensions should be updated
    assert_eq!(state.current_width.load(Ordering::SeqCst), 1920);
    assert_eq!(state.current_height.load(Ordering::SeqCst), 1080);
    assert_eq!(state.resize_count.load(Ordering::SeqCst), 1);
}

#[test]
fn cuj_eng01_multiple_resizes_processed_in_order() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: Multiple resize events are sent
    tx.send(Message::Control(EngineEventControl::Resize(800, 600)))
        .unwrap();
    tx.send(Message::Control(EngineEventControl::Resize(1024, 768)))
        .unwrap();
    tx.send(Message::Control(EngineEventControl::Resize(1920, 1080)))
        .unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Then: Final dimensions should reflect last resize
    assert_eq!(state.current_width.load(Ordering::SeqCst), 1920);
    assert_eq!(state.current_height.load(Ordering::SeqCst), 1080);
    assert_eq!(state.resize_count.load(Ordering::SeqCst), 3);

    // And: All resizes should be recorded in order
    let events = state.control_events.lock().unwrap();
    assert_eq!(events.len(), 3);
    assert!(matches!(events[0], EngineEventControl::Resize(800, 600)));
    assert!(matches!(events[1], EngineEventControl::Resize(1024, 768)));
    assert!(matches!(events[2], EngineEventControl::Resize(1920, 1080)));
}

// =============================================================================
// ENG-02: CloseRequested Tests
// =============================================================================

#[test]
fn cuj_eng02_close_requested_triggers_shutdown_flag() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: CloseRequested is sent
    tx.send(Message::Control(EngineEventControl::CloseRequested))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Close flag should be set
    assert!(state.close_requested.load(Ordering::SeqCst));

    let events = state.control_events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], EngineEventControl::CloseRequested));
}

// =============================================================================
// ENG-03: ScaleChanged Tests
// =============================================================================

#[test]
fn cuj_eng03_scale_changed_updates_scale_factor() {
    // Given: A terminal app actor with default scale
    let (tx, state, handle) = spawn_mock_app();

    // When: ScaleChanged event is sent
    tx.send(Message::Control(EngineEventControl::ScaleChanged(2.0)))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Scale factor should be updated
    let scale = *state.current_scale.lock().unwrap();
    assert!((scale - 2.0).abs() < 0.001);

    let events = state.control_events.lock().unwrap();
    assert_eq!(events.len(), 1);
    if let EngineEventControl::ScaleChanged(s) = events[0] {
        assert!((s - 2.0).abs() < 0.001);
    } else {
        panic!("Expected ScaleChanged event");
    }
}

// =============================================================================
// ENG-04: KeyDown Tests
// =============================================================================

#[test]
fn cuj_eng04_keydown_simple_character() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: A simple key event is sent
    tx.send(Message::Management(EngineEventManagement::KeyDown {
        key: KeySymbol::Char('a'),
        mods: Modifiers::empty(),
        text: Some("a".to_string()),
    }))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Key event should be received
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 1);

    if let EngineEventManagement::KeyDown { key, mods, text } = &events[0] {
        assert_eq!(*key, KeySymbol::Char('a'));
        assert!(mods.is_empty());
        assert_eq!(*text, Some("a".to_string()));
    } else {
        panic!("Expected KeyDown event");
    }

    // And: Text should be written to PTY
    let writes = state.pty_writes.lock().unwrap();
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0], b"a");
}

#[test]
fn cuj_eng04_keydown_with_modifiers() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: Key event with modifiers is sent
    tx.send(Message::Management(EngineEventManagement::KeyDown {
        key: KeySymbol::Char('c'),
        mods: Modifiers::CONTROL,
        text: None,
    }))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Key event with modifiers should be received
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 1);

    if let EngineEventManagement::KeyDown { key, mods, text } = &events[0] {
        assert_eq!(*key, KeySymbol::Char('c'));
        assert!(mods.contains(Modifiers::CONTROL));
        assert!(text.is_none());
    } else {
        panic!("Expected KeyDown event");
    }
}

#[test]
fn cuj_eng04_keydown_sequence_order_preserved() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: Multiple key events are sent rapidly
    for c in ['H', 'e', 'l', 'l', 'o'] {
        tx.send(Message::Management(EngineEventManagement::KeyDown {
            key: KeySymbol::Char(c),
            mods: Modifiers::empty(),
            text: Some(c.to_string()),
        }))
        .unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Then: All keys should be received in order
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 5);

    let chars: Vec<char> = events
        .iter()
        .filter_map(|e| {
            if let EngineEventManagement::KeyDown { key, .. } = e {
                if let KeySymbol::Char(c) = key {
                    Some(*c)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o']);
}

// =============================================================================
// ENG-05: MouseClick Tests
// =============================================================================

#[test]
fn cuj_eng05_mouse_click_left_button() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: A left click event is sent
    tx.send(Message::Management(EngineEventManagement::MouseClick {
        x: 100,
        y: 200,
        button: MouseButton::Left,
    }))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Click event should be received
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 1);

    if let EngineEventManagement::MouseClick { x, y, button } = &events[0] {
        assert_eq!(*x, 100);
        assert_eq!(*y, 200);
        assert_eq!(*button, MouseButton::Left);
    } else {
        panic!("Expected MouseClick event");
    }
}

#[test]
fn cuj_eng05_mouse_click_right_button() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: A right click event is sent
    tx.send(Message::Management(EngineEventManagement::MouseClick {
        x: 300,
        y: 400,
        button: MouseButton::Right,
    }))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Click event should be received with correct button
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 1);

    if let EngineEventManagement::MouseClick { button, .. } = &events[0] {
        assert_eq!(*button, MouseButton::Right);
    } else {
        panic!("Expected MouseClick event");
    }
}

// =============================================================================
// ENG-08: Paste Tests
// =============================================================================

#[test]
fn cuj_eng08_paste_simple_text() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: Paste event with simple text is sent
    tx.send(Message::Management(EngineEventManagement::Paste(
        "Hello, World!".to_string(),
    )))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Paste event should be received
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 1);

    if let EngineEventManagement::Paste(content) = &events[0] {
        assert_eq!(content, "Hello, World!");
    } else {
        panic!("Expected Paste event");
    }

    // And: Content should be written to PTY
    let writes = state.pty_writes.lock().unwrap();
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0], b"Hello, World!");
}

#[test]
fn cuj_eng08_paste_multiline_text() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: Paste event with multiline text is sent
    let multiline = "Line 1\nLine 2\nLine 3".to_string();
    tx.send(Message::Management(EngineEventManagement::Paste(
        multiline.clone(),
    )))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Full multiline content should be received
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 1);

    if let EngineEventManagement::Paste(content) = &events[0] {
        assert_eq!(content, &multiline);
        assert!(content.contains('\n'));
    } else {
        panic!("Expected Paste event");
    }
}

// =============================================================================
// ENG-09: RequestFrame Tests
// =============================================================================

#[test]
fn cuj_eng09_request_frame_with_timing_info() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: A frame request is sent with timing info
    let now = Instant::now();
    let target = now + Duration::from_millis(16);
    let refresh = Duration::from_millis(16);

    tx.send(Message::Data(EngineEventData::RequestFrame {
        timestamp: now,
        target_timestamp: target,
        refresh_interval: refresh,
    }))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Frame request should be received
    let events = state.data_events.lock().unwrap();
    assert_eq!(events.len(), 1);

    let EngineEventData::RequestFrame {
        refresh_interval, ..
    } = &events[0];
    assert_eq!(*refresh_interval, Duration::from_millis(16));
}

#[test]
fn cuj_eng09_request_frame_at_varying_rates() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: Frame requests are sent at different refresh rates
    let now = Instant::now();

    // 60 Hz request
    tx.send(Message::Data(EngineEventData::RequestFrame {
        timestamp: now,
        target_timestamp: now + Duration::from_millis(16),
        refresh_interval: Duration::from_millis(16),
    }))
    .unwrap();

    // 120 Hz request
    tx.send(Message::Data(EngineEventData::RequestFrame {
        timestamp: now,
        target_timestamp: now + Duration::from_millis(8),
        refresh_interval: Duration::from_millis(8),
    }))
    .unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Then: Both frame requests should be received
    let events = state.data_events.lock().unwrap();
    assert_eq!(events.len(), 2);

    let EngineEventData::RequestFrame {
        refresh_interval: interval_0,
        ..
    } = &events[0];
    assert_eq!(*interval_0, Duration::from_millis(16));

    let EngineEventData::RequestFrame {
        refresh_interval: interval_1,
        ..
    } = &events[1];
    assert_eq!(*interval_1, Duration::from_millis(8));
}

// =============================================================================
// ENG-10: RenderSurface Delivery Tests
// =============================================================================

/// A simple test manifold that returns a constant color
struct ConstantColorManifold {
    r: f32,
    g: f32,
    b: f32,
}

impl Manifold for ConstantColorManifold {
    type Output = Discrete;

    fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Self::Output {
        Discrete::pack(
            Field::from(self.r),
            Field::from(self.g),
            Field::from(self.b),
            Field::from(1.0),
        )
    }
}

/// Mock engine that receives AppData responses
struct MockEngineReceiver {
    render_surfaces: Arc<std::sync::Mutex<Vec<AppDataVariant>>>,
}

#[derive(Debug, Clone)]
enum AppDataVariant {
    RenderSurface,
    RenderSurfaceU32,
    Skipped,
}

impl Actor<AppData<pixelflow_graphics::Rgba8>, (), ()> for MockEngineReceiver {
    fn handle_data(&mut self, data: AppData<pixelflow_graphics::Rgba8>) {
        let variant = match data {
            AppData::RenderSurface(_) => AppDataVariant::RenderSurface,
            AppData::RenderSurfaceU32(_) => AppDataVariant::RenderSurfaceU32,
            AppData::Skipped => AppDataVariant::Skipped,
            AppData::_Phantom(_) => return,
        };
        self.render_surfaces.lock().unwrap().push(variant);
    }

    fn handle_control(&mut self, _: ()) {}
    fn handle_management(&mut self, _: ()) {}

    fn park(&mut self, _: ParkHint) -> ParkHint {
        ParkHint::Wait
    }
}

#[test]
fn cuj_eng10_render_surface_delivery() {
    // Given: A mock engine receiver
    let render_surfaces = Arc::new(std::sync::Mutex::new(Vec::new()));
    let surfaces_clone = render_surfaces.clone();

    let (tx, mut rx) =
        ActorScheduler::<AppData<pixelflow_graphics::Rgba8>, (), ()>::new(10, 64);

    let handle = thread::spawn(move || {
        let mut receiver = MockEngineReceiver {
            render_surfaces: surfaces_clone,
        };
        rx.run(&mut receiver);
    });

    // When: A render surface is sent to the engine
    let manifold = Arc::new(ConstantColorManifold {
        r: 1.0,
        g: 0.0,
        b: 0.0,
    });
    tx.send(Message::Data(AppData::RenderSurface(manifold)))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Engine should receive the render surface
    let surfaces = render_surfaces.lock().unwrap();
    assert_eq!(surfaces.len(), 1);
    assert!(matches!(surfaces[0], AppDataVariant::RenderSurface));
}

#[test]
fn cuj_eng10_render_surface_u32_delivery() {
    // Given: A mock engine receiver
    let render_surfaces = Arc::new(std::sync::Mutex::new(Vec::new()));
    let surfaces_clone = render_surfaces.clone();

    let (tx, mut rx) =
        ActorScheduler::<AppData<pixelflow_graphics::Rgba8>, (), ()>::new(10, 64);

    let handle = thread::spawn(move || {
        let mut receiver = MockEngineReceiver {
            render_surfaces: surfaces_clone,
        };
        rx.run(&mut receiver);
    });

    // When: A U32 render surface is sent
    let manifold = Arc::new(ConstantColorManifold {
        r: 0.0,
        g: 1.0,
        b: 0.0,
    });
    tx.send(Message::Data(AppData::RenderSurfaceU32(manifold)))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Engine should receive the U32 render surface
    let surfaces = render_surfaces.lock().unwrap();
    assert_eq!(surfaces.len(), 1);
    assert!(matches!(surfaces[0], AppDataVariant::RenderSurfaceU32));
}

// =============================================================================
// ENG-11: Frame Skip Handling Tests
// =============================================================================

#[test]
fn cuj_eng11_frame_skip_delivery() {
    // Given: A mock engine receiver
    let render_surfaces = Arc::new(std::sync::Mutex::new(Vec::new()));
    let surfaces_clone = render_surfaces.clone();

    let (tx, mut rx) =
        ActorScheduler::<AppData<pixelflow_graphics::Rgba8>, (), ()>::new(10, 64);

    let handle = thread::spawn(move || {
        let mut receiver = MockEngineReceiver {
            render_surfaces: surfaces_clone,
        };
        rx.run(&mut receiver);
    });

    // When: A frame skip is sent
    tx.send(Message::Data(AppData::Skipped)).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Engine should receive the skip notification
    let surfaces = render_surfaces.lock().unwrap();
    assert_eq!(surfaces.len(), 1);
    assert!(matches!(surfaces[0], AppDataVariant::Skipped));
}

// =============================================================================
// Priority Ordering Tests (Control > Management > Data)
// =============================================================================

#[test]
fn cuj_priority_control_events_processed_before_data() {
    // Given: An actor that records event order
    let event_order = Arc::new(std::sync::Mutex::new(Vec::new()));

    struct OrderRecordingActor {
        order: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl Actor<EngineEventData, EngineEventControl, EngineEventManagement> for OrderRecordingActor {
        fn handle_data(&mut self, _: EngineEventData) {
            self.order.lock().unwrap().push("Data".to_string());
        }
        fn handle_control(&mut self, ctrl: EngineEventControl) {
            let name = match ctrl {
                EngineEventControl::Resize(_, _) => "Resize",
                EngineEventControl::CloseRequested => "CloseRequested",
                EngineEventControl::ScaleChanged(_) => "ScaleChanged",
            };
            self.order.lock().unwrap().push(format!("Control:{}", name));
        }
        fn handle_management(&mut self, _: EngineEventManagement) {
            self.order.lock().unwrap().push("Management".to_string());
        }
        fn park(&mut self, _: ParkHint) -> ParkHint {
            ParkHint::Wait
        }
    }

    let (tx, mut rx) =
        ActorScheduler::<EngineEventData, EngineEventControl, EngineEventManagement>::new(64, 32);
    let order_clone = event_order.clone();

    let handle = thread::spawn(move || {
        let mut actor = OrderRecordingActor { order: order_clone };
        rx.run(&mut actor);
    });

    // When: Events are sent in Data, Management, Control order
    let now = Instant::now();
    tx.send(Message::Data(EngineEventData::RequestFrame {
        timestamp: now,
        target_timestamp: now,
        refresh_interval: Duration::from_millis(16),
    }))
    .unwrap();
    tx.send(Message::Management(EngineEventManagement::Paste(
        "test".to_string(),
    )))
    .unwrap();
    tx.send(Message::Control(EngineEventControl::CloseRequested))
        .unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Then: Control should be processed before earlier Data
    let order = event_order.lock().unwrap();
    let control_idx = order.iter().position(|s| s.starts_with("Control:"));
    let data_idx = order.iter().position(|s| s == "Data");

    assert!(
        control_idx.is_some() && data_idx.is_some(),
        "Both control and data should be processed"
    );
    assert!(
        control_idx.unwrap() < data_idx.unwrap(),
        "Control should be processed before data. Order: {:?}",
        *order
    );
}

// =============================================================================
// Focus Events Tests
// =============================================================================

#[test]
fn cuj_focus_gained_and_lost_events() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: Focus events are sent
    tx.send(Message::Management(EngineEventManagement::FocusGained))
        .unwrap();
    tx.send(Message::Management(EngineEventManagement::FocusLost))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Both focus events should be received in order
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], EngineEventManagement::FocusGained));
    assert!(matches!(events[1], EngineEventManagement::FocusLost));
}

// =============================================================================
// Mouse Scroll Tests
// =============================================================================

#[test]
fn cuj_eng07_mouse_scroll_event() {
    // Given: A terminal app actor
    let (tx, state, handle) = spawn_mock_app();

    // When: A scroll event is sent
    tx.send(Message::Management(EngineEventManagement::MouseScroll {
        x: 400,
        y: 300,
        dx: 0.0,
        dy: -3.0,
        mods: Modifiers::empty(),
    }))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: Scroll event should be received
    let events = state.management_events.lock().unwrap();
    assert_eq!(events.len(), 1);

    if let EngineEventManagement::MouseScroll { x, y, dy, .. } = &events[0] {
        assert_eq!(*x, 400);
        assert_eq!(*y, 300);
        assert!((*dy - (-3.0)).abs() < 0.001);
    } else {
        panic!("Expected MouseScroll event");
    }
}

// =============================================================================
// Concurrent Event Delivery Tests
// =============================================================================

#[test]
fn cuj_concurrent_events_all_delivered() {
    const NUM_SENDERS: usize = 4;
    const EVENTS_PER_SENDER: usize = 25;

    // Given: A terminal app actor
    let event_count = Arc::new(AtomicUsize::new(0));

    struct CountingActor {
        count: Arc<AtomicUsize>,
    }

    impl Actor<EngineEventData, EngineEventControl, EngineEventManagement> for CountingActor {
        fn handle_data(&mut self, _: EngineEventData) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        fn handle_control(&mut self, _: EngineEventControl) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        fn handle_management(&mut self, _: EngineEventManagement) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        fn park(&mut self, _: ParkHint) -> ParkHint {
            ParkHint::Wait
        }
    }

    let (tx, mut rx) =
        ActorScheduler::<EngineEventData, EngineEventControl, EngineEventManagement>::new(256, 64);
    let count_clone = event_count.clone();

    let receiver_handle = thread::spawn(move || {
        let mut actor = CountingActor { count: count_clone };
        rx.run(&mut actor);
    });

    // When: Multiple senders send events concurrently
    let mut sender_handles = Vec::new();
    for _ in 0..NUM_SENDERS {
        let tx_clone = tx.clone();
        let handle = thread::spawn(move || {
            for _ in 0..EVENTS_PER_SENDER {
                tx_clone
                    .send(Message::Management(EngineEventManagement::Paste(
                        "test".to_string(),
                    )))
                    .unwrap();
            }
        });
        sender_handles.push(handle);
    }

    for h in sender_handles {
        h.join().unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    receiver_handle.join().unwrap();

    // Then: All events should be delivered
    assert_eq!(
        event_count.load(Ordering::SeqCst),
        NUM_SENDERS * EVENTS_PER_SENDER,
        "All {} events should be delivered",
        NUM_SENDERS * EVENTS_PER_SENDER
    );
}
