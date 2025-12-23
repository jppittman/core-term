pub mod waker;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
mod linux;

use crate::api::private::{EngineActorHandle, EngineActorScheduler, EngineControl, EngineData};

use crate::api::private::WindowId;
use crate::api::public::AppManagement;
use crate::api::public::{
    AppData, Application, EngineEvent, EngineEventControl, EngineEventData, EngineEventManagement,
};
use crate::config::EngineConfig;
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt};
use crate::input::MouseButton;
use crate::render_pool::render_parallel;
use actor_scheduler::{Actor, Message, Troupe};
use std::time::Instant;

#[cfg(target_os = "macos")]
pub use macos::*;

#[cfg(target_os = "linux")]
pub use linux::*;

use crate::display::driver::DriverActor;
use anyhow::{Context, Result};
use log::info;
use pixelflow_core::Scale;
use pixelflow_graphics::render::rasterizer::{Rasterize, TensorShape};
use pixelflow_graphics::render::Frame;

// Platform Logic
use crate::display::platform::PlatformActor;

#[cfg(target_os = "macos")]
pub type ActivePlatform = PlatformActor<MetalOps>;

#[cfg(target_os = "linux")]
pub type ActivePlatform = PlatformActor<linux::LinuxOps>;

// Type Aliases - use platform-appropriate pixel format
#[cfg(target_os = "macos")]
pub type PlatformPixel = pixelflow_graphics::render::color::Rgba8;

#[cfg(target_os = "linux")]
pub type PlatformPixel = pixelflow_graphics::render::color::Bgra8;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub type PlatformPixel = pixelflow_graphics::render::color::Rgba8;

pub type PlatformDriver = DriverActor<ActivePlatform>;

pub struct EnginePlatform {
    driver: PlatformDriver,
    driver_handle:
        std::sync::Arc<actor_scheduler::ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>>,
    config: EngineConfig,
    #[allow(dead_code)]
    handle: EngineActorHandle<PlatformPixel>,
    troupe: Troupe,
}

impl EnginePlatform {
    #[cfg(target_os = "macos")]
    pub fn new(
        app: impl Application + Send + 'static,
        engine_handle: EngineActorHandle<PlatformPixel>,
        scheduler: EngineActorScheduler<PlatformPixel>,
        config: EngineConfig,
    ) -> Result<Self> {
        info!("EnginePlatform::new() - Creating ActorScheduler-based platform with app");

        // 1. Create Platform Ops (Metal)
        let ops = MetalOps::new(engine_handle.clone()).context("Failed to create platform ops")?;
        let platform = PlatformActor::new(ops);

        // 2. Create Scheduler for the Driver with CocoaWaker
        let waker = std::sync::Arc::new(waker::CocoaWaker::new());
        let (driver_handle, driver_scheduler) = actor_scheduler::create_actor::<
            DisplayData<PlatformPixel>,
            DisplayControl,
            DisplayMgmt,
        >(1024, Some(waker));

        let driver_handle = std::sync::Arc::new(driver_handle);

        Self::new_with_platform(app, engine_handle, scheduler, config, platform, driver_handle, driver_scheduler)
    }

    #[cfg(target_os = "linux")]
    pub fn new(
        app: impl Application + Send + 'static,
        engine_handle: EngineActorHandle<PlatformPixel>,
        scheduler: EngineActorScheduler<PlatformPixel>,
        config: EngineConfig,
    ) -> Result<Self> {
        info!("EnginePlatform::new() - Creating ActorScheduler-based platform with app (Linux)");

        // 1. Create Platform Ops (Linux/X11)
        let ops = linux::LinuxOps::new(engine_handle.clone()).context("Failed to create platform ops")?;
        let platform = PlatformActor::new(ops);

        // 2. Create Scheduler for the Driver (no special waker needed for Linux yet)
        let (driver_handle, driver_scheduler) = actor_scheduler::create_actor::<
            DisplayData<PlatformPixel>,
            DisplayControl,
            DisplayMgmt,
        >(1024, None);

        let driver_handle = std::sync::Arc::new(driver_handle);

        Self::new_with_platform(app, engine_handle, scheduler, config, platform, driver_handle, driver_scheduler)
    }

    fn new_with_platform(
        app: impl Application + Send + 'static,
        engine_handle: EngineActorHandle<PlatformPixel>,
        engine_scheduler: EngineActorScheduler<PlatformPixel>,
        config: EngineConfig,
        platform: ActivePlatform,
        driver_handle: std::sync::Arc<actor_scheduler::ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>>,
        driver_scheduler: actor_scheduler::ActorScheduler<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>,
    ) -> Result<Self> {
        let mut troupe = Troupe::new();

        // 3. Create DriverActor
        let driver = DriverActor::new(driver_scheduler, platform);

        // 4. Create VsyncActor
        let target_fps = config.performance.target_fps;
        let (vsync_handle, vsync_scheduler) = actor_scheduler::create_actor::<
            crate::vsync_actor::RenderedResponse,
            crate::vsync_actor::VsyncCommand,
            crate::vsync_actor::VsyncManagement,
        >(1024, None);
        let vsync_handle = std::sync::Arc::new(vsync_handle);

        let vsync_actor = crate::vsync_actor::VsyncActor::new(
            target_fps as f64,
            engine_handle.clone(),
            vsync_handle.clone(),
        );

        // Spawn Vsync Actor
        troupe.spawn_named("vsync-actor", vsync_scheduler, vsync_actor);
        let _ = vsync_handle.send(crate::vsync_actor::VsyncCommand::Start);

        // 5. Create EngineHandler
        let render_threads = config.performance.render_threads;
        let driver_handle_clone = driver_handle.clone();

        let engine_handler = EngineHandler {
            app,
            engine_handle: engine_handle.clone(),
            driver_handle: driver_handle_clone,
            framebuffer: None,
            physical_width: 0,
            physical_height: 0,
            scale_factor: 1.0,
            vsync_actor: Some(vsync_handle),
            render_threads,
            frame_count: 0,
        };

        // Spawn Engine Actor
        troupe.spawn_named("engine-actor", engine_scheduler, engine_handler);

        Ok(Self {
            driver,
            driver_handle,
            config,
            handle: engine_handle,
            troupe,
        })
    }

    /// Run the engine (driver loop on main thread).
    pub fn run(self) -> Result<()> {
        info!("EnginePlatform::run() - Starting driver on main thread");

        // Destructure self to get ownership of parts
        let EnginePlatform { driver, driver_handle, config, troupe, .. } = self;

        // Send CreateWindow command
        // DriverCommand is internal API enum, we need to map to DisplayMgmt
        let width = (config.window.columns as usize * config.window.cell_width_px) as u32;
        let height = (config.window.rows as usize * config.window.cell_height_px) as u32;

        // Manual mapping to DisplayMgmt::Create
        let _ = driver_handle
            .send(Message::Management(DisplayMgmt::Create {
                id: WindowId::PRIMARY,
                settings: crate::api::public::WindowDescriptor {
                    title: config.window.title.clone(),
                    width,
                    height,
                    ..Default::default()
                },
            }));

        // Explicitly drop driver_handle to ensure the driver's channel can be closed if needed
        // (though DriverActor runs on the scheduler which owns the receiver, so dropping sender
        // here is good practice to avoid holding it unnecessarily during the blocking run).
        // More critically, if the driver waits for all senders to disconnect to shut down,
        // we MUST drop this handle.
        drop(driver_handle);

        // Run driver on main thread (blocks)
        let res = driver.run();

        // Wait for other actors to finish when driver exits
        info!("EnginePlatform: Driver exited, waiting for troupe shutdown...");
        troupe.wait();

        res
    }
}

// Engine handler - processes events and coordinates app/driver communication
struct EngineHandler<A: Application> {
    app: A,
    engine_handle: EngineActorHandle<PlatformPixel>,
    // driver: PlatformDriver, // EngineHandler does not need the DriverActor, only the handle
    driver_handle:
        std::sync::Arc<actor_scheduler::ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>>,
    framebuffer: Option<Frame<PlatformPixel>>,
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    vsync_actor: Option<
        std::sync::Arc<actor_scheduler::ActorHandle<
            crate::vsync_actor::RenderedResponse,
            crate::vsync_actor::VsyncCommand,
            crate::vsync_actor::VsyncManagement,
        >>,
    >,
    render_threads: usize,
    frame_count: u64,
}

impl<A: Application> Actor<EngineData<PlatformPixel>, EngineControl<PlatformPixel>, AppManagement>
    for EngineHandler<A>
{
    fn handle_data(&mut self, data: EngineData<PlatformPixel>) {
        match data {
            EngineData::FromDriver(evt) => {
                // Track physical dimensions and scale factor from window events
                match &evt {
                    DisplayEvent::WindowCreated {
                        width_px,
                        height_px,
                        scale,
                        ..
                    } => {
                        self.physical_width = *width_px;
                        self.physical_height = *height_px;
                        self.scale_factor = *scale;
                        self.framebuffer = None;
                    }
                    DisplayEvent::Resized {
                        width_px,
                        height_px,
                        ..
                    } => {
                        self.physical_width = *width_px;
                        self.physical_height = *height_px;
                        self.framebuffer = None;
                    }
                    DisplayEvent::ScaleChanged { scale, .. } => {
                        self.scale_factor = *scale;
                        self.framebuffer = None;
                    }
                    _ => {}
                }

                // Forward display event to app
                if let Some(engine_evt) = map_display_event(&evt, self.scale_factor) {
                    let _ = self.app.send(engine_evt);
                }
            }
            EngineData::FromApp(AppData::RenderSurface(surface)) => {
                log::trace!("Engine: Received RenderSurface (continuous)");
                // App sent a rendered surface (continuous) - present it
                let mut frame = self
                    .framebuffer
                    .take()
                    .unwrap_or_else(|| Frame::new(self.physical_width, self.physical_height));

                // Use Rasterize combinator to adapt continuous surface to discrete grid
                let scaled = Scale::uniform(surface, self.scale_factor);
                let rasterized = Rasterize(scaled);

                let width = frame.width as usize;
                let height = frame.height as usize;
                let shape = TensorShape::new(width, height);
                let options = crate::render_pool::RenderOptions {
                    num_threads: self.render_threads,
                };
                render_parallel(&rasterized, frame.as_slice_mut(), shape, options);

                // Send frame to driver
                let _ = self.driver_handle.send(Message::Data(DisplayData::Present {
                    id: WindowId::PRIMARY,
                    frame,
                }));

                // Send feedback to VSync actor
                if let Some(ref vsync) = self.vsync_actor {
                    let _ = vsync.send(crate::vsync_actor::RenderedResponse {
                        frame_number: self.frame_count,
                        rendered_at: Instant::now(),
                    });
                }
                self.frame_count += 1;
            }
            EngineData::FromApp(AppData::RenderSurfaceU32(surface)) => {
                log::trace!("Engine: Received RenderSurfaceU32 (discrete)");
                // App sent a rendered surface (discrete) - present it
                let mut frame = self
                    .framebuffer
                    .take()
                    .unwrap_or_else(|| Frame::new(self.physical_width, self.physical_height));

                // Scale discrete surface using nearest neighbor
                let scaled = Scale::uniform(surface, self.scale_factor);

                let width = frame.width as usize;
                let height = frame.height as usize;
                let shape = TensorShape::new(width, height);
                let options = crate::render_pool::RenderOptions {
                    num_threads: self.render_threads,
                };
                render_parallel(&scaled, frame.as_slice_mut(), shape, options);

                // Send frame to driver
                let _ = self.driver_handle.send(Message::Data(DisplayData::Present {
                    id: WindowId::PRIMARY,
                    frame,
                }));

                // Send feedback to VSync actor
                if let Some(ref vsync) = self.vsync_actor {
                    let _ = vsync.send(crate::vsync_actor::RenderedResponse {
                        frame_number: self.frame_count,
                        rendered_at: Instant::now(),
                    });
                }
                self.frame_count += 1;
            }
            EngineData::FromApp(AppData::Skipped) => {
                log::trace!("Engine: Received Skipped frame");
                // App skipped frame, but we still need to return token to VSync
                if let Some(ref vsync) = self.vsync_actor {
                    let _ = vsync.send(crate::vsync_actor::RenderedResponse {
                        frame_number: self.frame_count,
                        rendered_at: Instant::now(),
                    });
                }
            }
            EngineData::FromApp(AppData::_Phantom(_)) => {
                unreachable!("_Phantom variant should never be constructed")
            }
        }
    }

    fn handle_control(&mut self, ctrl: EngineControl<PlatformPixel>) {
        match ctrl {
            EngineControl::PresentComplete(frame) => {
                // Store frame for reuse
                self.framebuffer = Some(frame);
            }
            EngineControl::VSync {
                timestamp,
                target_timestamp,
                refresh_interval,
            } => {
                log::trace!("Engine: Received VSync, forwarding to App");
                // Forward VSync as RequestFrame to app (push model)
                let event = EngineEvent::Data(EngineEventData::RequestFrame {
                    timestamp,
                    target_timestamp,
                    refresh_interval,
                });
                let _ = self.app.send(event);
            }
            EngineControl::UpdateRefreshRate(refresh_rate) => {
                if let Some(ref vsync) = self.vsync_actor {
                    // Update existing VSync actor
                    info!(
                        "Engine: Updating VSync refresh rate to {:.2} Hz",
                        refresh_rate
                    );
                    let _ = vsync.send(crate::vsync_actor::VsyncCommand::UpdateRefreshRate(
                        refresh_rate,
                    ));
                } else {
                    // This path should ideally be unreachable now that Vsync is spawned in new()
                    // But if we support dynamic respawning, we'd need access to Troupe or handle.
                    // For now, logging error.
                    log::error!("Engine: Cannot spawn new VSync actor - missing Troupe access");
                }
            }
            EngineControl::VsyncActorReady(actor) => {
                info!("Engine: VSync actor received from platform");
                let _ = actor.send(crate::vsync_actor::VsyncCommand::Start);
                self.vsync_actor = Some(std::sync::Arc::new(actor));
            }
            EngineControl::Quit => {
                info!("Engine: Quit requested from app");
                let _ = self
                    .driver_handle
                    .send(Message::Control(DisplayControl::Shutdown));
            }
            EngineControl::DriverAck => {
                // Ignore driver acks
            }
        }
    }

    fn handle_management(&mut self, mgmt: AppManagement) {
        match mgmt {
            AppManagement::SetTitle(title) => {
                let _ = self
                    .driver_handle
                    .send(Message::Control(DisplayControl::SetTitle {
                        id: WindowId::PRIMARY,
                        title,
                    }));
            }
            AppManagement::CopyToClipboard(_text) => {
                // Not supported in DisplayMgmt yet?
                // Or map to something else.
                // For now, logging.
                log::warn!("Clipboard copy not implemented in DisplayMgmt yet");
                // let _ = self.driver_handle.send(DisplayMgmt::CopyToClipboard(text));
            }
            AppManagement::RequestPaste => {
                log::warn!("Clipboard paste not implemented in DisplayMgmt yet");
                // let _ = self.driver_handle.send(DisplayMgmt::RequestPaste);
            }
            AppManagement::ResizeRequest(width, height) => {
                let _ = self
                    .driver_handle
                    .send(Message::Control(DisplayControl::SetSize {
                        id: WindowId::PRIMARY,
                        width,
                        height,
                    }));
            }
            AppManagement::SetCursorIcon(icon) => {
                let _ = self
                    .driver_handle
                    .send(Message::Control(DisplayControl::SetCursor {
                        id: WindowId::PRIMARY,
                        cursor: icon,
                    }));
            }
            AppManagement::Quit => {
                info!("Engine: App requested Quit");
                let _ = self
                    .driver_handle
                    .send(Message::Control(DisplayControl::Shutdown));
            }
        }
    }

    fn park(&mut self, _hint: actor_scheduler::ParkHint) {
        // Engine loop doesn't have periodic tasks, it reacts to messages
    }
}

/// Convert DisplayEvent to EngineEvent, converting physical to logical pixels.
fn map_display_event(evt: &DisplayEvent, scale_factor: f64) -> Option<EngineEvent> {
    match evt {
        DisplayEvent::WindowCreated {
            width_px,
            height_px,
            ..
        }
        | DisplayEvent::Resized {
            width_px,
            height_px,
            ..
        } => {
            // Convert physical pixels to logical pixels
            let logical_w = (*width_px as f64 / scale_factor) as u32;
            let logical_h = (*height_px as f64 / scale_factor) as u32;
            log::info!(
                "Engine: Mapping display resize {}x{} -> {}x{}",
                width_px,
                height_px,
                logical_w,
                logical_h
            );
            Some(EngineEvent::Control(EngineEventControl::Resize(
                logical_w, logical_h,
            )))
        }
        DisplayEvent::WindowDestroyed { id } => {
            log::info!("Engine: WindowDestroyed {:?}", id);
            None
        }
        DisplayEvent::CloseRequested { .. } => {
            Some(EngineEvent::Control(EngineEventControl::CloseRequested))
        }
        DisplayEvent::ScaleChanged { scale, .. } => Some(EngineEvent::Control(
            EngineEventControl::ScaleChanged(*scale),
        )),
        DisplayEvent::Key {
            symbol,
            modifiers,
            text,
            ..
        } => Some(EngineEvent::Management(EngineEventManagement::KeyDown {
            key: *symbol,
            mods: *modifiers,
            text: text.clone(),
        })),
        DisplayEvent::MouseButtonPress { button, x, y, .. } => {
            // Convert physical pixels to logical pixels
            let logical_x = (*x as f64 / scale_factor).max(0.0) as u32;
            let logical_y = (*y as f64 / scale_factor).max(0.0) as u32;
            let btn = match button {
                1 => MouseButton::Left,
                2 => MouseButton::Middle,
                3 => MouseButton::Right,
                _ => MouseButton::Other(*button),
            };
            Some(EngineEvent::Management(EngineEventManagement::MouseClick {
                x: logical_x,
                y: logical_y,
                button: btn,
            }))
        }
        DisplayEvent::MouseButtonRelease { button, x, y, .. } => {
            let logical_x = (*x as f64 / scale_factor).max(0.0) as u32;
            let logical_y = (*y as f64 / scale_factor).max(0.0) as u32;
            let btn = match button {
                1 => MouseButton::Left,
                2 => MouseButton::Middle,
                3 => MouseButton::Right,
                _ => MouseButton::Other(*button),
            };
            Some(EngineEvent::Management(
                EngineEventManagement::MouseRelease {
                    x: logical_x,
                    y: logical_y,
                    button: btn,
                },
            ))
        }
        DisplayEvent::MouseMove {
            x, y, modifiers, ..
        } => {
            let logical_x = (*x as f64 / scale_factor).max(0.0) as u32;
            let logical_y = (*y as f64 / scale_factor).max(0.0) as u32;
            Some(EngineEvent::Management(EngineEventManagement::MouseMove {
                x: logical_x,
                y: logical_y,
                mods: *modifiers,
            }))
        }
        DisplayEvent::MouseScroll {
            dx,
            dy,
            x,
            y,
            modifiers,
            ..
        } => {
            let logical_x = (*x as f64 / scale_factor).max(0.0) as u32;
            let logical_y = (*y as f64 / scale_factor).max(0.0) as u32;
            Some(EngineEvent::Management(
                EngineEventManagement::MouseScroll {
                    x: logical_x,
                    y: logical_y,
                    dx: *dx,
                    dy: *dy,
                    mods: *modifiers,
                },
            ))
        }
        DisplayEvent::FocusGained { .. } => {
            Some(EngineEvent::Management(EngineEventManagement::FocusGained))
        }
        DisplayEvent::FocusLost { .. } => {
            Some(EngineEvent::Management(EngineEventManagement::FocusLost))
        }
        DisplayEvent::PasteData { text } => Some(EngineEvent::Management(
            EngineEventManagement::Paste(text.clone()),
        )),
        DisplayEvent::ClipboardDataRequested => {
            log::trace!("Engine: ClipboardDataRequested (ignored)");
            None
        }
    }
}
