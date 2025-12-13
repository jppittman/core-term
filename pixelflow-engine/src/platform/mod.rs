pub mod waker;

use crate::api::private::{
    create_engine_actor, DriverCommand, EngineActorHandle, EngineActorScheduler, EngineControl,
    EngineData,
};
use crate::api::public::AppManagement;
use crate::api::public::{
    AppData, Application, EngineEvent, EngineEventControl, EngineEventData, EngineEventManagement,
};
use crate::config::EngineConfig;
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, WindowId};
use crate::input::MouseButton;
use crate::render_pool::render_parallel;
use anyhow::{Context, Result};
use log::info;
use pixelflow_core::Scale;
use pixelflow_render::Frame;
use actor_scheduler::{Actor, Message};
use std::time::Instant;

// Platform-specific driver type alias
#[cfg(use_x11_display)]
type PlatformDriver = crate::display::drivers::X11DisplayDriver;

#[cfg(use_cocoa_display)]
type PlatformDriver = crate::display::drivers::MetalDisplayDriver;

#[cfg(use_headless_display)]
type PlatformDriver = crate::display::drivers::HeadlessDisplayDriver;

#[cfg(use_web_display)]
type PlatformDriver = crate::display::drivers::WebDisplayDriver;

/// The platform's native pixel type (determined by the display driver).
pub type PlatformPixel = <PlatformDriver as DisplayDriver>::Pixel;

pub struct EnginePlatform {
    driver: PlatformDriver,
    config: EngineConfig,
    handle: EngineActorHandle<PlatformPixel>,
    scheduler: EngineActorScheduler<PlatformPixel>,
}

impl EnginePlatform {
    pub fn new(
        app: impl Application + Send + 'static,
        engine_handle: EngineActorHandle<PlatformPixel>,
        config: EngineConfig,
    ) -> Result<Self> {
        info!("EnginePlatform::new() - Creating ActorScheduler-based platform with app");

        // Create engine channels with platform-specific wake handler (None for now, macOS later)
        let (handle, scheduler) = create_engine_actor::<PlatformPixel>(None);

        let driver = PlatformDriver::new(handle.clone())
            .context("Failed to create display driver")?;

        // Spawn engine thread with app
        let driver_clone = driver.clone();
        let target_fps = config.performance.target_fps;
        let render_threads = config.performance.render_threads;

        std::thread::spawn(move || {
            if let Err(e) = engine_loop(app, engine_handle, driver_clone, scheduler, target_fps, render_threads) {
                log::error!("Engine loop error: {}", e);
            }
        });

        Ok(Self {
            driver,
            config,
            handle,
            scheduler: create_engine_actor::<PlatformPixel>(None).1, // Dummy, moved to thread
        })
    }

    /// Run the engine (driver loop on main thread).
    pub fn run(mut self) -> Result<()> {
        info!("EnginePlatform::run() - Starting driver on main thread");

        // Send CreateWindow command
        let width = (self.config.window.columns as usize * self.config.window.cell_width_px) as u32;
        let height = (self.config.window.rows as usize * self.config.window.cell_height_px) as u32;
        self.driver.send(DriverCommand::CreateWindow {
            id: WindowId::PRIMARY,
            width,
            height,
            title: self.config.window.title.clone(),
        })?;

        // Run driver on main thread (blocks)
        self.driver.run()
    }
}

// Engine handler - processes events and coordinates app/driver communication
struct EngineHandler<A: Application> {
    app: A,
    engine_handle: EngineActorHandle<PlatformPixel>,
    driver: PlatformDriver,
    framebuffer: Option<Frame<PlatformPixel>>,
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    vsync_actor: Option<actor_scheduler::ActorHandle<
        crate::vsync_actor::RenderedResponse,
        crate::vsync_actor::VsyncCommand,
        crate::vsync_actor::VsyncManagement,
    >>,
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
                    DisplayEvent::WindowCreated { width_px, height_px, scale, .. } => {
                        self.physical_width = *width_px;
                        self.physical_height = *height_px;
                        self.scale_factor = *scale;
                        self.framebuffer = None;
                    }
                    DisplayEvent::Resized { width_px, height_px, .. } => {
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
                // App sent a rendered surface - present it
                let mut frame = self
                    .framebuffer
                    .take()
                    .unwrap_or_else(|| Frame::new(self.physical_width, self.physical_height));

                // Render with parallel rasterization
                // Note: Scale holds Arc<dyn Surface...>, which implements Surface
                // But Scale<Arc<dyn Surface>> needs to implement Surface too.
                // pixelflow-core implements Surface for Scale<S> where S: Surface.
                // However, Arc<dyn Surface> doesn't automatically implement Surface.
                // We need to verify if pixelflow-core provides Surface impl for Arc<dyn Surface>.
                // It seems it does not based on the error.
                // We should wrap the surface in a struct that implements Surface or fix the upstream.
                // For now, let's assume we can clone the Arc and use it if we had a struct.

                // Workaround: We need a type that implements Surface and delegates to Arc<dyn Surface>.
                // Or maybe just pass the surface directly if scale is 1.0 (optimization).

                // Let's create a wrapper struct locally if needed, or check if we can fix imports.
                // The error says: `Scale<Arc<dyn Surface...>>` doesn't implement `Surface`.
                // This means `Arc<dyn Surface...>` doesn't implement `Surface`.

                // Render with parallel rasterization
                // Note: Scale holds Arc<dyn Surface...>, which implements Surface
                // and pixelflow-core implements Surface for Scale<S> where S: Surface.
                // However, render_parallel expects S: Surface<P>.
                // Scale<S> implements Surface<T, f32> (since Arc<dyn Surface<P, f32>> has C=f32).
                // render_parallel doesn't specify C, so it defaults to u32?
                // Let's check render_pool.rs: pub fn render_parallel<P, S>(... S: Surface<P> ...)
                // Surface<P> defaults C=u32.
                // But Scale<S> where S uses f32 coordinates implements Surface<P, f32>.
                // It does NOT implement Surface<P, u32>.

                // We need to bridge from discrete (u32) coordinates used by render_parallel
                // to continuous (f32) coordinates used by Scale.

                // pixelflow-core usually provides a "Discrete" combinator or similar,
                // or we need to look at how Scale handles this.
                // Scale has:
                // impl<T, S> Surface<T, u32> for Scale<S> where S: Surface<T, u32>
                // impl<T, S> Surface<T, f32> for Scale<S> where S: Surface<T, f32>

                // Our 'surface' is Surface<P, f32>.
                // So 'scaled' is Surface<P, f32>.
                // But render_parallel iterates over u32 pixels and calls eval(Batch<u32>, Batch<u32>).
                // So we need something that implements Surface<P, u32> but wraps Surface<P, f32>.

                // That is the `Discrete` combinator mentioned in memory.
                // "pixelflow-core provides a `Discrete<S>` combinator that adapts a `Surface<P, u32>` to `Surface<P, f32>`"
                // Wait, it adapts "Surface<P, u32> to Surface<P, f32>"? Or the other way around?
                // Usually it takes a continuous surface and makes it discrete by casting coords.

                // Let's assume there is a Discrete combinator in pixelflow_core::surfaces::transform or similar.
                // I will search for it.

                // Use Rasterize combinator to adapt continuous surface to discrete grid
                let scaled = Scale::uniform(surface, self.scale_factor);
                let rasterized = pixelflow_core::surfaces::raster::Rasterize(scaled);

                let width = frame.width as usize;
                let height = frame.height as usize;
                render_parallel(&rasterized, frame.as_slice_mut(), width, height, self.render_threads);

                // Send frame to driver
                let _ = self.driver.send(DriverCommand::Present {
                    id: WindowId::PRIMARY,
                    frame,
                });

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
                    info!("Engine: Updating VSync refresh rate to {:.2} Hz", refresh_rate);
                    let _ = vsync.send(crate::vsync_actor::VsyncCommand::UpdateRefreshRate(refresh_rate));
                } else {
                    // Spawn VSync actor for the first time
                    info!("Engine: Spawning VSync actor with {:.2} Hz", refresh_rate);
                    let vsync_actor = crate::vsync_actor::VsyncActor::spawn(
                        refresh_rate,
                        self.engine_handle.clone()
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
                let _ = self.driver.send(DriverCommand::Shutdown);
            }
            EngineControl::DriverAck => {
                // Ignore driver acks
            }
        }
    }

    fn handle_management(&mut self, mgmt: AppManagement) {
        match mgmt {
            AppManagement::SetTitle(title) => {
                let _ = self.driver.send(DriverCommand::SetTitle {
                    id: WindowId::PRIMARY,
                    title,
                });
            }
            AppManagement::CopyToClipboard(text) => {
                let _ = self.driver.send(DriverCommand::CopyToClipboard(text));
            }
            AppManagement::RequestPaste => {
                let _ = self.driver.send(DriverCommand::RequestPaste);
            }
            AppManagement::ResizeRequest(width, height) => {
                let _ = self.driver.send(DriverCommand::SetSize {
                    id: WindowId::PRIMARY,
                    width,
                    height,
                });
            }
            AppManagement::SetCursorIcon(icon) => {
                let _ = self.driver.send(DriverCommand::SetCursorIcon { icon });
            }
        }
    }

    fn park(&mut self) {
        // Engine has no periodic tasks
    }
}

fn engine_loop<A: Application>(
    app: A,
    engine_handle: EngineActorHandle<PlatformPixel>,
    driver: PlatformDriver,
    mut scheduler: EngineActorScheduler<PlatformPixel>,
    _target_fps: u32,
    render_threads: usize,
) -> Result<()> {
    info!("Engine loop started (scheduler model, {} threads)", render_threads);

    let mut handler = EngineHandler {
        app,
        engine_handle,
        driver,
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
