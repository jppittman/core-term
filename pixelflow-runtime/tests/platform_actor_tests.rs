use actor_scheduler::{Actor, ActorStatus, SystemStatus};
use pixelflow_graphics::render::Frame;
use pixelflow_runtime::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use pixelflow_runtime::display::ops::PlatformOps;
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
    fn handle_data(&mut self, msg: DisplayData) {
        match msg {
            DisplayData::Present { id, .. } => {
                self.push_log(&format!("Present {:?}", id));
            }
        }
    }

    fn handle_control(&mut self, msg: DisplayControl) {
        match msg {
            DisplayControl::SetTitle { id, title } => {
                self.push_log(&format!("SetTitle {:?} {}", id, title));
            }
            _ => self.push_log(&format!("Control {:?}", msg)),
        }
    }

    fn handle_management(&mut self, msg: DisplayMgmt) {
        match msg {
            DisplayMgmt::Create { .. } => {
                self.push_log("Create");
            }
            _ => self.push_log(&format!("Management {:?}", msg)),
        }
    }

    fn park(&mut self, _status: SystemStatus) -> ActorStatus {
        self.push_log("Park");
        ActorStatus::Idle
    }
}

#[test]
fn test_platform_actor_delegation() {
    // 1. Create MockOps
    let ops = MockOps::new();
    let log_ref = ops.log.clone();

    // 2. Create PlatformActor
    let mut actor = PlatformActor::new(ops);

    // 3. Send messages manually to verify delegation
    // Note: We test `Actor` trait implementation directly, skipping Scheduler for unit test simplicity

    // Test Management (Create)
    actor.handle_management(DisplayMgmt::Create {
        settings: Default::default(),
    });

    // Test Control (SetTitle)
    actor.handle_control(DisplayControl::SetTitle {
        id: pixelflow_runtime::api::private::WindowId(1),
        title: "Test Window".to_string(),
    });

    // Test Data (Present)
    actor.handle_data(DisplayData::Present {
        id: pixelflow_runtime::api::private::WindowId(1),
        frame: Frame::new(100, 100),
    });

    // Test Park
    actor.park(SystemStatus::Busy);

    // 4. Verify Log
    let log = log_ref.lock().unwrap();
    assert_eq!(log.len(), 4);
    assert_eq!(log[0], "Create");
    assert!(log[1].contains("SetTitle WindowId(1) Test Window"));
    assert!(log[2].contains("Present WindowId(1)"));
    assert_eq!(log[3], "Park");
}
