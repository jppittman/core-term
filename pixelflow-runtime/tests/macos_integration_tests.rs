#[cfg(target_os = "macos")]
mod tests {
    //! `MetalOps` against a real window server.
    //!
    //! This file used to stand up an entire engine actor, a `MockEngine` to capture events, a
    //! background scheduler thread, and a 100ms sleep — all of it existing only to observe the
    //! `WindowCreated` event that `MetalOps` sent. Since rollout step 4
    //! (`docs/designs/pixelflow-runtime-engine-mesh-migration.md` §5.1) `PlatformOps` *returns*
    //! its outbound events in a `DriverOut` instead of sending them, so all of that apparatus is
    //! gone: the events are read directly out of the value the call fills in.
    //!
    //! That also removes the `thread::sleep`-then-check, which was the one genuinely
    //! flake-prone thing here — it asserted an event had arrived by an arbitrary wall-clock
    //! deadline while racing a background thread. There is no longer anything asynchronous to
    //! race.

    use actor_scheduler::SystemStatus;
    use pixelflow_runtime::api::public::WindowDescriptor;
    use pixelflow_runtime::display::messages::{DisplayControl, DisplayEvent, DisplayMgmt};
    use pixelflow_runtime::display::ops::{DriverEmit, DriverOut, PlatformOps};
    use pixelflow_runtime::platform::macos::MetalOps;

    #[test]
    #[ignore = "Requires UI interaction or window server"]
    fn metal_ops_lifecycle() {
        // No engine, no scheduler, no background thread, no sleep.
        let mut ops = MetalOps::new().expect("Failed to create MetalOps");
        let mut out = DriverOut::default();

        let settings = WindowDescriptor {
            title: "Integration Test Window".to_string(),
            width: 800,
            height: 600,
            ..Default::default()
        };

        ops.handle_management(DisplayMgmt::Create { settings }, &mut out)
            .expect("Create failed");

        // Emulate a run-loop step, which drains the Cocoa event queue.
        let _status = ops
            .park(SystemStatus::Busy, &mut out)
            .expect("park failed");

        let found_window_id = out.emits.iter().find_map(|emit| match emit {
            DriverEmit::Event(DisplayEvent::WindowCreated { window }) => Some(window.id),
            _ => None,
        });
        assert!(
            found_window_id.is_some(),
            "Expected WindowCreated event, found: {:?}",
            out.emits
        );

        let win_id = found_window_id.expect("checked immediately above");
        ops.handle_control(
            DisplayControl::SetTitle {
                id: win_id,
                title: "Updated Title".to_string(),
            },
            &mut out,
        )
        .expect("SetTitle failed");
    }
}
