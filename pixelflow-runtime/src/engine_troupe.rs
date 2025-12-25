//! Engine Troupe - Render pipeline actor coordination using troupe! macro.

use crate::api::private::{EngineControl, EngineData};
use crate::api::public::AppManagement;
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use crate::display::platform::PlatformActor;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{VsyncActor, VsyncConfig, VsyncManagement};
use crate::error::RuntimeError;
use actor_scheduler::{Actor, ActorTypes, Message, ParkHint, TroupeActor};

/// Engine handler - coordinates app, rendering, display.
pub struct EngineHandler;

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
    fn handle_data(&mut self, _data: EngineData<PlatformPixel>) {
        // TODO: Implement rendering logic
    }

    fn handle_control(&mut self, _ctrl: EngineControl<PlatformPixel>) {
        // TODO: Implement control logic
    }

    fn handle_management(&mut self, _mgmt: AppManagement) {
        // TODO: Implement management logic
    }

    fn park(&mut self, hint: ParkHint) -> ParkHint {
        hint
    }
}

// Implement TroupeActor for EngineHandler
impl<'a> TroupeActor<'a, Directory> for EngineHandler {
    fn new(_dir: &'a Directory) -> Self {
        Self
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
