//! Engine Troupe - Coordinated lifecycle for render pipeline actors.
//!
//! The EngineTroupe manages three cooperating actors:
//! - **DriverActor** [main]: Platform display driver, runs on main/GUI thread
//! - **EngineHandler**: Coordinates app, rendering, and display
//! - **VsyncActor**: Generates frame timing signals
//!
//! # Two-Phase Initialization
//!
//! ```ignore
//! // Phase 1: Create troupe (channels ready, no threads yet)
//! let troupe = EngineTroupe::new(config)?;
//!
//! // Phase 2: Get engine handle for app
//! let engine_handle = troupe.engine_handle();
//!
//! // Create app with engine handle
//! let app = MyApp::new(engine_handle);
//!
//! // Phase 3: Run troupe with app (blocks on main thread)
//! troupe.play(app)?;
//! ```

use crate::api::private::{EngineActorHandle, EngineActorScheduler, EngineControl, EngineData, WindowId};
use crate::api::public::{AppManagement, Application};
use crate::config::EngineConfig;
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use crate::display::platform::PlatformActor;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{RenderedResponse, VsyncActor, VsyncCommand, VsyncManagement};
use actor_scheduler::{ActorHandle, ActorScheduler, Message};
use anyhow::{Context, Result};
use log::info;

/// Directory of handles for all actors in the engine troupe.
pub struct EngineDirectory {
    /// Handle to send to the engine (for external app)
    pub engine: EngineActorHandle<PlatformPixel>,
    /// Handle to send to the driver
    pub driver: ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>,
    /// Handle to send to vsync actor
    pub vsync: ActorHandle<RenderedResponse, VsyncCommand, VsyncManagement>,
}

/// Handles exposed to external code (the app).
pub struct ExposedHandles {
    /// Engine handle - app sends render surfaces here
    pub engine: EngineActorHandle<PlatformPixel>,
}

/// Engine Troupe manages the render pipeline lifecycle.
///
/// Owns all schedulers and the platform driver. Use two-phase init:
/// 1. `new()` - creates all channels
/// 2. `engine_handle()` - get handle for app
/// 3. `play(app)` - run actors
pub struct EngineTroupe {
    // Directory of handles (for actor communication)
    directory: EngineDirectory,

    // Schedulers (owned, consumed in play())
    engine_scheduler: EngineActorScheduler<PlatformPixel>,
    // Note: driver_scheduler is owned by the DriverActor
    vsync_scheduler: ActorScheduler<RenderedResponse, VsyncCommand, VsyncManagement>,

    // Platform-specific driver actor (runs on main thread, owns its scheduler)
    driver: DriverActor<ActivePlatform>,

    // Configuration
    config: EngineConfig,
}

impl EngineTroupe {
    /// Create a new engine troupe. Phase 1: channels are created, no threads spawned.
    #[cfg(target_os = "macos")]
    pub fn new(config: EngineConfig) -> Result<Self> {
        use crate::platform::{waker, MetalOps};

        info!("EngineTroupe::new() - Creating render pipeline (macOS)");

        // Phase 1: Create all handles and schedulers upfront

        // Engine actor (receives from driver and app)
        let (engine_handle, engine_scheduler) = actor_scheduler::create_actor::<
            EngineData<PlatformPixel>,
            EngineControl<PlatformPixel>,
            AppManagement,
        >(256, None);

        // VSync actor (receives timing feedback and commands)
        let (vsync_handle, vsync_scheduler) = actor_scheduler::create_actor::<
            RenderedResponse,
            VsyncCommand,
            VsyncManagement,
        >(64, None);

        // Platform ops and driver
        let ops = MetalOps::new(engine_handle.clone())
            .context("Failed to create Metal platform ops")?;
        let platform = PlatformActor::new(ops);

        // Driver with Cocoa waker for main thread wake
        let waker = std::sync::Arc::new(waker::CocoaWaker::new());
        let (driver_handle, driver_scheduler) = actor_scheduler::create_actor::<
            DisplayData<PlatformPixel>,
            DisplayControl,
            DisplayMgmt,
        >(1024, Some(waker));

        let driver = DriverActor::new(driver_scheduler, platform);

        // Build directory
        let directory = EngineDirectory {
            engine: engine_handle,
            driver: driver_handle,
            vsync: vsync_handle,
        };

        Ok(Self {
            directory,
            engine_scheduler,
            vsync_scheduler,
            driver,
            config,
        })
    }

    /// Create a new engine troupe. Phase 1: channels are created, no threads spawned.
    #[cfg(target_os = "linux")]
    pub fn new(config: EngineConfig) -> Result<Self> {
        use crate::platform::linux::LinuxOps;

        info!("EngineTroupe::new() - Creating render pipeline (Linux)");

        // Phase 1: Create all handles and schedulers upfront

        // Engine actor (receives from driver and app)
        let (engine_handle, engine_scheduler) = actor_scheduler::create_actor::<
            EngineData<PlatformPixel>,
            EngineControl<PlatformPixel>,
            AppManagement,
        >(256, None);

        // VSync actor (receives timing feedback and commands)
        let (vsync_handle, vsync_scheduler) = actor_scheduler::create_actor::<
            RenderedResponse,
            VsyncCommand,
            VsyncManagement,
        >(64, None);

        // Platform ops and driver
        let ops = LinuxOps::new(engine_handle.clone())
            .context("Failed to create Linux platform ops")?;
        let platform = PlatformActor::new(ops);

        // Driver (no special waker on Linux)
        let (driver_handle, driver_scheduler) = actor_scheduler::create_actor::<
            DisplayData<PlatformPixel>,
            DisplayControl,
            DisplayMgmt,
        >(1024, None);

        let driver = DriverActor::new(driver_scheduler, platform);

        // Build directory
        let directory = EngineDirectory {
            engine: engine_handle,
            driver: driver_handle,
            vsync: vsync_handle,
        };

        Ok(Self {
            directory,
            engine_scheduler,
            vsync_scheduler,
            driver,
            config,
        })
    }

    /// Get the engine handle for the app. Call after new(), before play().
    pub fn engine_handle(&self) -> EngineActorHandle<PlatformPixel> {
        self.directory.engine.clone()
    }

    /// Get exposed handles for external code.
    pub fn exposed(&self) -> ExposedHandles {
        ExposedHandles {
            engine: self.directory.engine.clone(),
        }
    }

    /// Run the troupe with the given application. Blocks on main thread.
    ///
    /// Phase 3: Spawns engine and vsync threads, runs driver on main thread.
    pub fn play<A: Application + Send + 'static>(self, app: A) -> Result<()> {
        info!("EngineTroupe::play() - Starting render pipeline");

        let Self {
            directory,
            mut engine_scheduler,
            mut vsync_scheduler,
            mut driver,
            config,
        } = self;

        // Spawn VSync actor thread
        let vsync_handle = directory.vsync.clone();
        let engine_handle_for_vsync = directory.engine.clone();
        let target_fps = config.performance.target_fps as f64;

        std::thread::Builder::new()
            .name("vsync".to_string())
            .spawn(move || {
                // VsyncActor::new takes the handle to itself (for clock thread)
                let mut actor = VsyncActor::new(
                    target_fps,
                    engine_handle_for_vsync,
                    vsync_handle,
                );
                vsync_scheduler.run(&mut actor);
            })
            .context("Failed to spawn vsync thread")?;

        // Start vsync
        let _ = directory.vsync.send(Message::Control(VsyncCommand::Start));

        // Spawn engine thread
        let driver_handle = directory.driver.clone();
        let engine_handle = directory.engine.clone();
        let vsync_handle_for_engine = directory.vsync.clone();
        let render_threads = config.performance.render_threads;

        std::thread::Builder::new()
            .name("engine".to_string())
            .spawn(move || {
                let mut handler = super::platform::EngineHandler::new(
                    app,
                    engine_handle,
                    driver_handle,
                    Some(vsync_handle_for_engine),
                    render_threads,
                );
                engine_scheduler.run(&mut handler);
            })
            .context("Failed to spawn engine thread")?;

        // Send CreateWindow command to driver
        let width = (config.window.columns as usize * config.window.cell_width_px) as u32;
        let height = (config.window.rows as usize * config.window.cell_height_px) as u32;

        let _ = directory.driver.send(Message::Management(DisplayMgmt::Create {
            id: WindowId::PRIMARY,
            settings: crate::api::public::WindowDescriptor {
                title: config.window.title.clone(),
                width,
                height,
                ..Default::default()
            },
        }));

        // Run driver on main thread [main] - blocks until shutdown
        driver.run()
    }
}
