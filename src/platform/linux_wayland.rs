use anyhow::Result;

use crate::platform::actions::PlatformAction;
use crate::platform::backends::wayland::driver::WaylandDriver;
use crate::platform::backends::{Driver, PlatformState}; // Added Driver
use crate::platform::os::pty::{NixPty, PtyConfig}; // Added PtyConfig
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;

pub struct LinuxWaylandPlatform {
    driver: WaylandDriver,
    pty: NixPty,
}

impl Platform for LinuxWaylandPlatform {
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)> {
        log::info!(
            "LinuxWaylandPlatform::new() with PTY: {}x{}, cmd: {}, args: {:?}",
            initial_pty_cols,
            initial_pty_rows,
            shell_command,
            shell_args.clone() // Clone for logging if needed elsewhere, config takes refs
        );
        let driver = WaylandDriver::new()?;

        // Convert Vec<String> to Vec<&str> for PtyConfig
        let shell_args_str: Vec<&str> = shell_args.iter().map(|s| s.as_str()).collect();

        let pty_config = PtyConfig {
            command_executable: &shell_command,
            args: &shell_args_str,
            initial_cols: initial_pty_cols,
            initial_rows: initial_pty_rows,
        };
        let pty = NixPty::spawn_with_config(&pty_config)?;

        let initial_platform_state = driver.get_platform_state();
        Ok((
            Self { driver, pty },
            initial_platform_state,
        ))
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        log::info!("LinuxWaylandPlatform::poll_events()");
        Ok(vec![])
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        log::info!("LinuxWaylandPlatform::dispatch_actions() received {} actions", actions.len());
        // for action in actions {
        //     match action {
        //         PlatformAction::Pty(pty_action) => {
        //             log::info!("PtyAction: {:?}", pty_action);
        //             // self.pty.handle_action(pty_action)?; // Placeholder for actual handling
        //         }
        //         PlatformAction::Ui(ui_action) => {
        //             log::info!("UiAction: {:?}", ui_action);
        //             // self.driver.handle_action(ui_action)?; // Placeholder for actual handling
        //         }
        //     }
        // }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        log::info!("LinuxWaylandPlatform::get_current_platform_state()");
        self.driver.get_platform_state()
    }

    fn cleanup(&mut self) -> Result<()> {
        log::info!("LinuxWaylandPlatform::cleanup()");
        // self.driver.cleanup()?; // Assuming Driver trait will have a cleanup method
        // self.pty.cleanup()?; // Assuming NixPty will have a cleanup method
        Ok(())
    }
}
