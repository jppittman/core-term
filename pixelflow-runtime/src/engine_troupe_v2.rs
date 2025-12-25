//! Engine Troupe - Coordinated lifecycle for render pipeline actors.
//!
//! The EngineTroupe manages three cooperating actors using the troupe! macro:
//! - **DriverActor** [main]: Platform display driver, runs on main/GUI thread  
//! - **EngineHandler** [expose]: Coordinates app, rendering, and display
//! - **VsyncActor**: Generates frame timing signals
//!
//! # Usage
//!
//! ```ignore
//! let troupe = EngineTroupe::with_config(config)?;
//! let engine_handle = troupe.exposed().engine;
//!
//! let app = MyApp::new(engine_handle.clone());
//! engine_handle.send(EngineManagement::SetApplication(Box::new(app)))?;
//!
//! troupe.play()?;  // Blocks on main thread
//! ```

use crate::api::private::{EngineActorHandle, EngineControl, EngineData, WindowId};
use crate::api::public::{AppManagement, Application};
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use crate::display::platform::PlatformActor;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{
    RenderedResponse, VsyncActor, VsyncCommand, VsyncConfig, VsyncManagement,
};
use crate::error::RuntimeError;
use actor_scheduler::{ActorHandle, Message};
use log::info;

// Use the troupe! macro to generate Directory, ExposedHandles, Troupe, and run()
actor_scheduler::troupe! {
    driver: DriverActor<ActivePlatform> [main],
    engine: EngineHandler [expose],
    vsync: VsyncActor,
}

// TODO: Implement TroupeActor for each actor with #[actor_impl]
// This requires refactoring EngineHandler to take &Directory in new()
// For now, keeping the manual implementation as a reference

/// Engine handler actor - coordinates app, rendering, and display.
///
/// TODO: Convert to TroupeActor pattern with #[actor_impl]
pub struct EngineHandler {
    // Will be refactored to use Directory reference
}

impl Troupe {
    /// Create troupe with platform-specific configuration.
    #[cfg(target_os = "macos")]
    pub fn with_config(config: EngineConfig) -> Result<Self, RuntimeError> {
        use crate::platform::{waker, MetalOps};

        info!("EngineTroupe::with_config() - Creating render pipeline (macOS)");

        // Phase 1: Create troupe (macro handles this)
        let troupe = Self::new();

        // Phase 2: Platform-specific initialization
        // Create Metal ops using engine handle from directory
        let ops = MetalOps::new(troupe.directory().engine.clone())
            .map_err(|e| RuntimeError::InitError(format!("Failed to create Metal platform ops: {}", e)))?;
        let platform = PlatformActor::new(ops);

        // TODO: Send platform to driver via Management message
        // For now, we have a chicken-egg problem - need to solve this

        // Configure vsync
        let vsync_config = VsyncConfig {
            refresh_rate: config.performance.target_fps as f64,
        };
        troupe
            .directory()
            .vsync
            .send(Message::Management(VsyncManagement::SetConfig {
                config: vsync_config,
                engine_handle: troupe.directory().engine.clone(),
                self_handle: troupe.directory().vsync.clone(),
            })).map_err(|e| RuntimeError::InitError(format!("Failed to configure vsync: {}", e)))?;

        Ok(troupe)
    }

    #[cfg(target_os = "linux")]
    pub fn with_config(config: EngineConfig) -> Result<Self, RuntimeError> {
        use crate::platform::linux::LinuxOps;

        info!("EngineTroupe::with_config() - Creating render pipeline (Linux)");

        let troupe = Self::new();

        let ops = LinuxOps::new(troupe.directory().engine.clone())
            .map_err(|e| RuntimeError::InitError(format!("Failed to create Linux platform ops: {}", e)))?;
        let platform = PlatformActor::new(ops);

        // TODO: Same platform initialization issue

        let vsync_config = VsyncConfig {
            refresh_rate: config.performance.target_fps as f64,
        };
        troupe
            .directory()
            .vsync
            .send(Message::Management(VsyncManagement::SetConfig {
                config: vsync_config,
                engine_handle: troupe.directory().engine.clone(),
                self_handle: troupe.directory().vsync.clone(),
            })).map_err(|e| RuntimeError::InitError(format!("Failed to configure vsync: {}", e)))?;

        Ok(troupe)
    }
}
