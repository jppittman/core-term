#[cfg(target_os = "macos")]
mod tests {
    use actor_scheduler::{Actor, ParkHint};
    use pixelflow_runtime::api::private::{
        create_engine_actor, EngineControl, EngineData, WindowId,
    };
    use pixelflow_runtime::api::public::{AppManagement, WindowDescriptor};
    use pixelflow_runtime::display::messages::{DisplayControl, DisplayMgmt};
    use pixelflow_runtime::display::ops::PlatformOps;
    use pixelflow_runtime::platform::macos::MetalOps;
    use pixelflow_runtime::platform::PlatformPixel;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    // A mock Engine actor to capture events
    struct MockEngine {
        pub captured_events: Arc<Mutex<Vec<pixelflow_runtime::display::messages::DisplayEvent>>>,
    }

    impl MockEngine {
        fn new(
            captured_events: Arc<Mutex<Vec<pixelflow_runtime::display::messages::DisplayEvent>>>,
        ) -> Self {
            Self { captured_events }
        }
    }

    // Use generics as required by the Actor trait definition
    impl Actor<EngineData<PlatformPixel>, EngineControl<PlatformPixel>, AppManagement> for MockEngine {
        fn handle_data(&mut self, msg: EngineData<PlatformPixel>) {
            if let EngineData::FromDriver(evt) = msg {
                self.captured_events.lock().unwrap().push(evt);
            }
        }

        fn handle_control(&mut self, _msg: EngineControl<PlatformPixel>) {}
        fn handle_management(&mut self, _msg: AppManagement) {}
        fn park(&mut self, _hint: ParkHint) {
            // No-op for mock, maybe sleep to prevent tight loop if we were polling in run
            // But scheduler usually handles waiting on channel
        }
    }

    #[test]
    #[ignore = "Requires UI interaction or window server"]
    fn test_metal_ops_lifecycle() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        // 1. Create Engine Actor (Scheduler + Handle)
        let (handle, mut scheduler) = create_engine_actor::<PlatformPixel>(None);

        // 2. Spawn Scheduler in background
        thread::spawn(move || {
            let mut mock_engine = MockEngine::new(events_clone);
            scheduler.run(&mut mock_engine);
        });

        // 3. Instantiate MetalOps
        let mut ops = MetalOps::new(handle).expect("Failed to create MetalOps");

        // 4. Create a Window
        let win_id = WindowId(1);
        let settings = WindowDescriptor {
            title: "Integration Test Window".to_string(),
            width: 800,
            height: 600,
            ..Default::default()
        };

        ops.handle_management(DisplayMgmt::Create {
            id: win_id,
            settings,
        });

        // 5. Emulate run loop step (Platform)
        // This should trigger window creation and send event to Engine
        ops.park(ParkHint::Poll);

        // Give some time for message passing
        thread::sleep(Duration::from_millis(100));

        // 6. Verify Window Creation Event within MockEngine
        let captured = events.lock().unwrap();
        let found = captured.iter().any(|e| matches!(
            e,
            pixelflow_runtime::display::messages::DisplayEvent::WindowCreated { id, .. } if *id == win_id
        ));
        assert!(
            found,
            "Expected WindowCreated event, found: {:?}",
            *captured
        );

        // 7. Update Window Title
        ops.handle_control(DisplayControl::SetTitle {
            id: win_id,
            title: "Updated Title".to_string(),
        });

        // 8. Explicitly drop ops to close the handle, allowing scheduler to exit (though thread detach is fine for test)
        drop(ops);
    }
}
