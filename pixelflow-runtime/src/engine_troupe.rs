//! Engine Troupe - Render pipeline actor coordination using troupe! macro.

use crate::api::private::{EngineControl, EngineData};
use crate::api::public::AppManagement;
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use crate::display::platform::PlatformActor;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{
    RenderedResponse, VsyncActor, VsyncCommand, VsyncConfig, VsyncManagement,
};
use actor_scheduler::{Actor, Message, TroupeActor};
use anyhow::{Context, Result};

/// Engine handler - coordinates app, rendering, display.
pub struct EngineHandler<'a> {
    dir: &'a Directory,
    app: Option<Box<dyn crate::api::public::Application + Send>>,
    framebuffer: Option<crate::render_pool::Frame<PlatformPixel>>,
    frame_count: u64,
}

#[actor_scheduler::actor_impl]
impl EngineHandler<'_> {
    type Data = EngineData<PlatformPixel>;
    type Control = EngineControl<PlatformPixel>;
    type Management = AppManagement;

    fn new(dir: &Directory) -> Self {
        Self {
            dir,
            app: None,
            framebuffer: None,
            frame_count: 0,
        }
    }

    fn handle_data(&mut self, _data: Self::Data) {
        // TODO: Implement rendering logic from platform/mod.rs EngineHandler
    }

    fn handle_control(&mut self, _ctrl: Self::Control) {
        // TODO: Implement control logic
    }

    fn handle_management(&mut self, mgmt: Self::Management) {
        match mgmt {
            AppManagement::SetTitle(_) => {}
            AppManagement::Quit => {}
            _ => {}
        }
    }

    fn park(&mut self, hint: actor_scheduler::ParkHint) -> actor_scheduler::ParkHint {
        hint
    }
}

// Generate troupe structures using macro
actor_scheduler::troupe! {
    driver: DriverActor<ActivePlatform> [main],
    engine: EngineHandler [expose],
    vsync: VsyncActor,
}

impl Troupe {
    /// Create troupe with platform-specific configuration.
    #[cfg(target_os = "macos")]
    pub fn with_config(config: EngineConfig) -> Result<Self> {
        use crate::platform::{waker, MetalOps};

        let troupe = Self::new();
        let dir = troupe.directory();

        // Create Metal platform
        let ops = MetalOps::new(dir.engine.clone()).context("Failed to create Metal ops")?;
        let platform = PlatformActor::new(ops);

        // Configure vsync
        dir.vsync
            .send(Message::Management(VsyncManagement::SetConfig {
                config: VsyncConfig {
                    refresh_rate: config.performance.target_fps as f64,
                },
                engine_handle: dir.engine.clone(),
                self_handle: dir.vsync.clone(),
            }))?;

        // Send platform to driver - TODO: Add SetPlatform management message
        drop(platform); // Temporary - need to solve platform injection

        Ok(troupe)
    }

    #[cfg(target_os = "linux")]
    pub fn with_config(config: EngineConfig) -> Result<Self> {
        use crate::platform::linux::LinuxOps;

        let troupe = Self::new();
        let dir = troupe.directory();

        let ops = LinuxOps::new(dir.engine.clone()).context("Failed to create Linux ops")?;
        let platform = PlatformActor::new(ops);

        dir.vsync
            .send(Message::Management(VsyncManagement::SetConfig {
                config: VsyncConfig {
                    refresh_rate: config.performance.target_fps as f64,
                },
                engine_handle: dir.engine.clone(),
                self_handle: dir.vsync.clone(),
            }))?;

        drop(platform);
        Ok(troupe)
    }
}
