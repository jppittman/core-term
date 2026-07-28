use actor_scheduler::{Actor, ActorStatus, HandlerError, HandlerResult, SystemStatus};
use pixelflow_graphics::render::Frame;
use pixelflow_runtime::api::private::create_engine_actor;
use pixelflow_runtime::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use pixelflow_runtime::display::ops::{DriverOut, PlatformOps};
use pixelflow_runtime::display::platform::PlatformActor;
use std::sync::{Arc, Mutex};

// Mock Platform Operations
#[derive(Clone)]
struct MockOps {
    // Shared state to verify calls
    pub log: Arc<Mutex<Vec<String>>>,
}

impl MockOps {
    fn new() -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn push_log(&self, msg: &str) {
        self.log.lock().unwrap().push(msg.to_string());
    }
}

unsafe impl Send for MockOps {}

impl PlatformOps for MockOps {
    fn handle_data(&mut self, msg: DisplayData, _out: &mut DriverOut) -> HandlerResult {
        match msg {
            DisplayData::Present { window } => {
                self.push_log(&format!("Present {:?}", window.id));
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, msg: DisplayControl, _out: &mut DriverOut) -> HandlerResult {
        match msg {
            DisplayControl::SetTitle { id, title } => {
                self.push_log(&format!("SetTitle {:?} {}", id, title));
            }
            _ => self.push_log(&format!("Control {:?}", msg)),
        }
        Ok(())
    }

    fn handle_management(&mut self, msg: DisplayMgmt, _out: &mut DriverOut) -> HandlerResult {
        match msg {
            DisplayMgmt::Create { .. } => {
                self.push_log("Create");
            }
            _ => self.push_log(&format!("Management {:?}", msg)),
        }
        Ok(())
    }

    fn handle_os(
        &mut self,
        _status: SystemStatus,
        _out: &mut DriverOut,
    ) -> Result<ActorStatus, HandlerError> {
        self.push_log("HandleOs");
        Ok(ActorStatus::Idle)
    }
}

#[test]
fn platform_actor_delegation() {
    // 1. Create MockOps
    let ops = MockOps::new();
    let log_ref = ops.log.clone();

    // 2. Create PlatformActor
    // The adapter is what performs sends, so it needs a handle even though MockOps never
    // emits. `_sched` stays bound to keep the receiving end alive for the actor's lifetime.
    let (engine_handle, _sched) = create_engine_actor(None);
    let mut actor = PlatformActor::new(ops, engine_handle);

    // 3. Send messages manually to verify delegation
    // Note: We test `Actor` trait implementation directly, skipping Scheduler for unit test simplicity

    // Test Management (Create)
    actor
        .handle_management(DisplayMgmt::Create {
            settings: Default::default(),
        })
        .expect("handle_management should succeed");

    // Test Control (SetTitle)
    actor
        .handle_control(DisplayControl::SetTitle {
            id: pixelflow_runtime::api::private::WindowId(1),
            title: "Test Window".to_string(),
        })
        .expect("handle_control should succeed");

    // Test Data (Present)
    actor
        .handle_data(DisplayData::Present {
            window: pixelflow_runtime::display::messages::Window {
                id: pixelflow_runtime::api::private::WindowId(1),
                frame: Frame::new(100, 100),
                width_px: 100,
                height_px: 100,
                scale: 1.0,
                generation: Default::default(),
            },
        })
        .expect("handle_data should succeed");

    // Test HandleOs
    actor
        .handle_os(SystemStatus::Busy)
        .expect("handle_os should succeed");

    // 4. Verify Log
    let log = log_ref.lock().unwrap();
    assert_eq!(log.len(), 4);
    assert_eq!(log[0], "Create");
    assert!(log[1].contains("SetTitle WindowId(1) Test Window"));
    assert!(log[2].contains("Present WindowId(1)"));
    assert_eq!(log[3], "HandleOs");
}

/// The point of moving sends out of `PlatformOps`: an implementation can now be driven and
/// observed with no engine, no adapter, no scheduler and no channels anywhere in the picture.
/// Before this, `PlatformOps` held a live `EngineActorHandle` and its outbound events could
/// only be seen by standing up an engine to receive them.
#[test]
fn platform_ops_emit_without_an_engine() {
    use pixelflow_runtime::api::private::WindowId;
    use pixelflow_runtime::display::messages::{DisplayEvent, Window};
    use pixelflow_runtime::display::ops::DriverEmit;
    use pixelflow_runtime::platform::PlatformPixel;

    /// Emits on every lane, so the test can show `DriverOut` accumulating rather than sending.
    struct EmittingOps;

    impl PlatformOps for EmittingOps {
        fn handle_data(&mut self, msg: DisplayData, out: &mut DriverOut) -> HandlerResult {
            let DisplayData::Present { window } = msg;
            // The buffer, after its pixels reached the screen: handed back rather than sent.
            out.blitted(window);
            Ok(())
        }
        fn handle_control(&mut self, _: DisplayControl, _out: &mut DriverOut) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: DisplayMgmt, _out: &mut DriverOut) -> HandlerResult {
            Ok(())
        }
        fn handle_os(
            &mut self,
            _status: SystemStatus,
            out: &mut DriverOut,
        ) -> Result<ActorStatus, HandlerError> {
            // `handle_os` is the reason `DriverOut` carries a Vec rather than an Option: draining
            // the OS queue yields N events from one step.
            out.event(DisplayEvent::FocusGained { id: WindowId(1) });
            out.event(DisplayEvent::FocusLost { id: WindowId(1) });
            Ok(ActorStatus::Idle)
        }
    }

    let mut ops = EmittingOps;
    let mut out = DriverOut::default();

    // An empty step emits nothing, and costs no allocation.
    ops.handle_control(DisplayControl::Bell, &mut out).unwrap();
    assert!(
        out.emits.is_empty(),
        "a step that does nothing must emit nothing"
    );

    // One step, N events.
    ops.handle_os(SystemStatus::Idle, &mut out).unwrap();
    assert_eq!(
        out.emits.len(),
        2,
        "handle_os drains N events in a single step"
    );
    assert!(matches!(
        out.emits[0],
        DriverEmit::Event(DisplayEvent::FocusGained { .. })
    ));
    assert!(matches!(
        out.emits[1],
        DriverEmit::Event(DisplayEvent::FocusLost { .. })
    ));

    // The buffer travels back as a value, so the test can inspect it directly rather than
    // needing an engine to receive it.
    out.emits.clear();
    let window = Window {
        id: WindowId(7),
        frame: Frame::<PlatformPixel>::new(4, 4),
        width_px: 4,
        height_px: 4,
        scale: 1.0,
        generation: Default::default(),
    };
    ops.handle_data(DisplayData::Present { window }, &mut out)
        .unwrap();
    match &out.emits[..] {
        [DriverEmit::Blitted(returned)] => assert_eq!(returned.id, WindowId(7)),
        other => panic!("expected exactly one buffer hand-back, got {:?}", other),
    }
}
