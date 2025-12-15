pub mod waker;

#[cfg(target_os = "macos")]
pub mod macos;

use crate::api::private::{
    create_engine_actor, DriverCommand, EngineActorHandle, EngineActorScheduler, EngineControl,
    EngineData,
};
use crate::api::public::AppManagement;
use crate::api::public::{
    AppData, Application, EngineEvent, EngineEventControl, EngineEventData, EngineEventManagement,
};
use crate::config::EngineConfig;
// use crate::display::driver::DisplayDriver; // Gone
use crate::api::private::WindowId;
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt};
use crate::display::DriverActor;
use crate::input::MouseButton;
use crate::render_pool::render_parallel;
use actor_scheduler::{Actor, ActorHandle, ActorScheduler, Message};
use anyhow::{Context, Result};
use log::info;
use pixelflow_core::Scale;
use pixelflow_render::Frame;
use std::time::Instant;

// Platform Logic
#[cfg(target_os = "macos")]
use macos::MetalPlatform as ActivePlatform;

// Type Aliases
pub type PlatformPixel = <ActivePlatform as crate::display::Platform>::Pixel;
pub type PlatformDriver = DriverActor<ActivePlatform>;

pub struct EnginePlatform {
    driver: PlatformDriver,
    driver_handle:
        actor_scheduler::ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>,
    config: EngineConfig,
    handle: EngineActorHandle<PlatformPixel>,
}

impl EnginePlatform {
    pub fn new(
        app: impl Application + Send + 'static,
        engine_handle: EngineActorHandle<PlatformPixel>,
        scheduler: EngineActorScheduler<PlatformPixel>,
        config: EngineConfig,
    ) -> Result<Self> {
        info!("EnginePlatform::new() - Creating ActorScheduler-based platform with app");

        // 1. Create Platform (Metal)
        // We pass engine_handle so platform can send events back
        let platform =
            ActivePlatform::new(engine_handle.clone()).context("Failed to create platform")?;

        // 2. Create Scheduler for the Driver

        // Actually, `driver_scheduler` runs on main thread inside `DriverActor::run`.
        // We need a Waker if other threads push to it and it sleeps.
        // `MetalPlatform` uses `sys` blocking calls, so `ParkHint` handles sleep.
        // But if we push to channel, we need to wake it up from `nextEventMatchingMask`.
        // So we NEED `CocoaWaker`.

        let waker = std::sync::Arc::new(waker::CocoaWaker::new());
        // We can't set waker easily on created scheduler?
        // `create_actor_scheduler` takes `Option<Arc<dyn Waker>>`.
        // So:
        let (driver_handle, driver_scheduler) = actor_scheduler::create_actor::<
            DisplayData<PlatformPixel>,
            DisplayControl,
            DisplayMgmt,
        >(1024, Some(waker));

        // 3. Create DriverActor
        let driver = DriverActor::new(driver_scheduler, platform);

        // Spawn engine thread with app
        // Driver must be CLONED? No, DriverActor is not clone.
        // `driver` runs on main thread.
        // The *Handle* (`driver_handle`) is what we give to the engine to talk to the driver.
        // But `EnginePlatform::driver` field is `PlatformDriver` (DriverActor).
        // Wait, `EnginePlatform` stores `driver` and calls `run` on it.
        // `driver.send`?? `DriverActor` does not have `send`.
        // `driver_handle` has `send`.

        // Refactor `EnginePlatform` to hold `driver` (to run it) and `driver_handle` (to pass to engine).

        // Spawn engine thread
        let target_fps = config.performance.target_fps;
        let render_threads = config.performance.render_threads;

        let driver_handle_clone = driver_handle.clone(); // For engine thread
        let engine_handle_clone = engine_handle.clone();

        std::thread::spawn(move || {
            if let Err(e) = engine_loop(
                app,
                engine_handle_clone,
                driver_handle_clone,
                scheduler,
                target_fps,
                render_threads,
            ) {
                log::error!("Engine loop error: {}", e);
            }
        });

        Ok(Self {
            driver,
            driver_handle,
            config,
            handle: engine_handle,
        })
    }

    /// Run the engine (driver loop on main thread).
    pub fn run(mut self) -> Result<()> {
        info!("EnginePlatform::run() - Starting driver on main thread");

        // Send CreateWindow command
        // Send CreateWindow command
        // DriverCommand is internal API enum, we need to map to DisplayMgmt
        let width = (self.config.window.columns as usize * self.config.window.cell_width_px) as u32;
        let height = (self.config.window.rows as usize * self.config.window.cell_height_px) as u32;

        // Manual mapping to DisplayMgmt::Create
        let _ = self
            .driver_handle
            .send(Message::Management(DisplayMgmt::Create {
                id: WindowId::PRIMARY,
                settings: crate::api::public::WindowDescriptor {
                    title: self.config.window.title.clone(),
                    width,
                    height,
                    ..Default::default()
                },
            }));

        // Run driver on main thread (blocks)
        self.driver.run()
    }
}

// Engine handler - processes events and coordinates app/driver communication
struct EngineHandler<A: Application> {
    app: A,
    engine_handle: EngineActorHandle<PlatformPixel>,
    // driver: PlatformDriver, // EngineHandler does not need the DriverActor, only the handle
    driver_handle:
        actor_scheduler::ActorHandle<DisplayData<PlatformPixel>, DisplayControl, DisplayMgmt>,
    framebuffer: Option<Frame<PlatformPixel>>,
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    vsync_actor: Option<
        actor_scheduler::ActorHandle<
            crate::vsync_actor::RenderedResponse,
            crate::vsync_actor::VsyncCommand,
            crate::vsync_actor::VsyncManagement,
        >,
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
                // App sent a rendered surface (continuous) - present it
                let mut frame = self
                    .framebuffer
                    .take()
                    .unwrap_or_else(|| Frame::new(self.physical_width, self.physical_height));

                // Use Rasterize combinator to adapt continuous surface to discrete grid
                let scaled = Scale::uniform(surface, self.scale_factor);
                let rasterized = pixelflow_core::surfaces::raster::Rasterize(scaled);

                let width = frame.width as usize;
                let height = frame.height as usize;
                render_parallel(
                    &rasterized,
                    frame.as_slice_mut(),
                    width,
                    height,
                    self.render_threads,
                );

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
                // App sent a rendered surface (discrete) - present it
                let mut frame = self
                    .framebuffer
                    .take()
                    .unwrap_or_else(|| Frame::new(self.physical_width, self.physical_height));

                // Scale discrete surface using nearest neighbor (Scale<u32> does this)
                let scaled = Scale::uniform(surface, self.scale_factor);

                let width = frame.width as usize;
                let height = frame.height as usize;
                // Render directly, no rasterization needed for u32 surface
                render_parallel(
                    &scaled,
                    frame.as_slice_mut(),
                    width,
                    height,
                    self.render_threads,
                );

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
                    // Spawn VSync actor for the first time
                    info!("Engine: Spawning VSync actor with {:.2} Hz", refresh_rate);
                    let vsync_actor = crate::vsync_actor::VsyncActor::spawn(
                        refresh_rate,
                        self.engine_handle.clone(),
                    );
                    let _ = vsync_actor.send(crate::vsync_actor::VsyncCommand::Start);
                    self.vsync_actor = Some(vsync_actor);
                }
            }
            EngineControl::VsyncActorReady(actor) => {
                info!("Engine: VSync actor received from platform");
                let _ = actor.send(crate::vsync_actor::VsyncCommand::Start);
                self.vsync_actor = Some(actor);
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
        // Engine has no periodic tasks
    }
}

fn engine_loop<A: Application>(
    app: A,
    engine_handle: EngineActorHandle<PlatformPixel>,
    driver_handle: actor_scheduler::ActorHandle<
        DisplayData<PlatformPixel>,
        DisplayControl,
        DisplayMgmt,
    >,
    mut scheduler: EngineActorScheduler<PlatformPixel>,
    _target_fps: u32,
    render_threads: usize,
) -> Result<()> {
    info!(
        "Engine loop started (scheduler model, {} threads)",
        render_threads
    );

    let mut handler = EngineHandler {
        app,
        engine_handle,
        driver_handle,
        framebuffer: None,
        physical_width: 0,
        physical_height: 0,
        scale_factor: 1.0,
        vsync_actor: None,
        render_threads,
        frame_count: 0,
    };

    scheduler.run(&mut handler);
    Ok(())
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
