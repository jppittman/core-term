pub mod waker;

use crate::api::private::{
    DriverCommand, EngineControl,
    EngineData, DisplayEvent, WindowId, DISPLAY_EVENT_BURST_LIMIT, DISPLAY_EVENT_BUFFER_SIZE,
};

// Re-export for convenience
pub use crate::api::private::EngineActorHandle;
use actor_scheduler::{Message, spawn_with_config};
use crate::api::public::AppManagement;
use crate::api::public::{
    AppData, Application, EngineEvent, EngineEventControl, EngineEventData, EngineEventManagement,
};
use crate::config::EngineConfig;
use crate::display::driver::DisplayDriver;
use crate::input::MouseButton;
use crate::render_pool::render_parallel;
use anyhow::{Context, Result};
use log::info;
use pixelflow_core::Scale;
use pixelflow_render::Frame;
use actor_scheduler::Actor;
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
}

impl EnginePlatform {
    /// Create a new platform (no spawning).
    pub fn new(config: EngineConfig) -> Result<Self> {
        info!("EnginePlatform::new() - Creating platform");

        // Create driver WITHOUT handle
        let driver = PlatformDriver::new()
            .context("Failed to create display driver")?;

        Ok(Self {
            driver,
            config,
        })
    }

    /// Initialize the platform (spawns engine actor, returns handle).
    pub fn init(self) -> Result<(Self, EngineActorHandle<PlatformPixel>)> {
        info!("EnginePlatform::init() - Spawning engine actor");

        let render_threads = self.config.performance.render_threads;

        let handler = EngineHandler {
            app: None,
            engine_handle: None,
            driver: self.driver.clone(),
            framebuffer: None,
            physical_width: 0,
            physical_height: 0,
            scale_factor: 1.0,
            vsync_actor: None,
            render_threads,
        };

        // Spawn engine actor - returns handle
        let engine_handle = spawn_with_config(
            handler,
            DISPLAY_EVENT_BURST_LIMIT,
            DISPLAY_EVENT_BUFFER_SIZE,
            None,  // No wake handler for now
        );

        // Inject weak handle into engine actor for VSync creation
        engine_handle.send(actor_scheduler::Message::Control(
            EngineControl::SetEngineHandle(engine_handle.downgrade())
        ))?;

        // Inject weak handle into driver
        self.driver.send(DriverCommand::SetEngineHandle(engine_handle.downgrade()))?;

        Ok((self, engine_handle))
    }

    /// Run the engine (driver loop on main thread).
    ///
    /// Takes the application handle and registers it with the engine.
    pub fn run(
        self,
        app_handle: actor_scheduler::ActorHandle<EngineEvent, (), AppManagement>,
        engine_handle: &EngineActorHandle<PlatformPixel>,
    ) -> Result<()> {
        info!("EnginePlatform::run() - Registering app and starting driver");

        // Register app with engine
        engine_handle.send(actor_scheduler::Message::Control(
            EngineControl::RegisterApp(app_handle.downgrade())
        ))?;

        // Send CreateWindow command
        self.driver.send(DriverCommand::CreateWindow {
            id: WindowId::PRIMARY,
            width: self.config.window.width,
            height: self.config.window.height,
            title: self.config.window.title.clone(),
        })?;

        // Run driver on main thread (blocks)
        self.driver.run()
    }
}

// Engine handler - processes events and coordinates app/driver communication
struct EngineHandler<D>
where
    D: DisplayDriver,
{
    app: Option<actor_scheduler::WeakActorHandle<EngineEvent, (), AppManagement>>,
    engine_handle: Option<actor_scheduler::WeakActorHandle<EngineData<D::Pixel>, EngineControl<D::Pixel>, AppManagement>>,
    driver: D,
    framebuffer: Option<Frame<D::Pixel>>,
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    vsync_actor: Option<crate::vsync_actor::VsyncActorHandle>,
    render_threads: usize,
}

impl<D> Actor<EngineData<D::Pixel>, EngineControl<D::Pixel>, AppManagement>
    for EngineHandler<D>
where
    D: DisplayDriver,
{
    fn handle_data(&mut self, data: EngineData<D::Pixel>) {
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
                    // All other events don't affect physical dimensions
                    DisplayEvent::WindowDestroyed { .. } => {}
                    DisplayEvent::CloseRequested { .. } => {}
                    DisplayEvent::Key { .. } => {}
                    DisplayEvent::MouseButtonPress { .. } => {}
                    DisplayEvent::MouseButtonRelease { .. } => {}
                    DisplayEvent::MouseMove { .. } => {}
                    DisplayEvent::MouseScroll { .. } => {}
                    DisplayEvent::FocusGained { .. } => {}
                    DisplayEvent::FocusLost { .. } => {}
                    DisplayEvent::PasteData { .. } => {}
                    DisplayEvent::ClipboardDataRequested => {}
                }

                // Forward display event to app (if registered)
                if let Some(engine_evt) = map_display_event(&evt, self.scale_factor) {
                    if let Some(app) = &self.app {
                        let _ = app.send(actor_scheduler::Message::Data(engine_evt));
                    }
                }
            }
            EngineData::FromApp(AppData::RenderSurface { frame_id, surface, app_submit_time }) => {
                let now = std::time::Instant::now();
                log::info!("FRAME_TIMING: Frame {} - App->Engine: {:.3}ms",
                    frame_id,
                    (now - app_submit_time).as_secs_f64() * 1000.0
                );

                // App sent a rendered surface - present it
                let mut frame = self
                    .framebuffer
                    .take()
                    .unwrap_or_else(|| Frame::new(self.physical_width, self.physical_height));

                log::trace!("Engine: Rendering surface {}x{} (physical) with {} threads",
                    self.physical_width, self.physical_height, self.render_threads);

                // Render with parallel rasterization
                let execute_start = std::time::Instant::now();
                let scaled = Scale::uniform(surface, self.scale_factor);
                let width = frame.width as usize;
                let height = frame.height as usize;
                render_parallel(&scaled, frame.as_slice_mut(), width, height, self.render_threads);
                let execute_end = std::time::Instant::now();

                log::info!("FRAME_TIMING: Frame {} - Execute: {:.3}ms",
                    frame_id,
                    (execute_end - execute_start).as_secs_f64() * 1000.0
                );

                log::trace!("Engine: Sending Present command to driver");
                // Send frame to driver
                let _ = self.driver.send(DriverCommand::Present {
                    id: WindowId::PRIMARY,
                    frame_id,
                    frame,
                    engine_submit_time: std::time::Instant::now(),
                });
            }
            EngineData::FromVSync(vsync_msg) => {
                let now = std::time::Instant::now();
                log::info!("FRAME_TIMING: Frame {} - VSync->Engine: {:.3}ms",
                    vsync_msg.frame_id,
                    (now - vsync_msg.timestamp).as_secs_f64() * 1000.0
                );

                // Forward VSync as RequestFrame to app (push model)
                let event = EngineEvent::Data(EngineEventData::RequestFrame {
                    frame_id: vsync_msg.frame_id,
                    timestamp: vsync_msg.timestamp,
                    target_timestamp: vsync_msg.target_timestamp,
                    refresh_interval: vsync_msg.refresh_interval,
                });
                if let Some(app) = &self.app {
                    if let Err(e) = app.send(actor_scheduler::Message::Data(event)) {
                        log::error!("Engine: Failed to send RequestFrame to app: {:?}", e);
                    } else {
                        log::trace!("Engine: RequestFrame sent to app successfully");
                    }
                }
            }
        }
    }

    fn handle_control(&mut self, ctrl: EngineControl<D::Pixel>) {
        match ctrl {
            EngineControl::PresentComplete(frame) => {
                // Notify VSync to refill token bucket
                if let Some(vsync) = &self.vsync_actor {
                    let _ = vsync.send(crate::vsync_actor::VsyncRequest::RenderedResponse);
                }
                // Store frame for reuse
                self.framebuffer = Some(frame);
            }
            EngineControl::UpdateRefreshRate(refresh_rate) => {
                info!("Engine: Refresh rate update to {:.2} Hz", refresh_rate);

                if self.vsync_actor.is_some() {
                    // Update existing VSync actor
                    if let Some(vsync) = &self.vsync_actor {
                        let _ = vsync.send(crate::vsync_actor::VsyncRequest::UpdateRefreshRate(refresh_rate));
                    }
                } else {
                    // Spawn new VSync actor
                    if let Some(weak_handle) = &self.engine_handle {
                        let weak_handle = weak_handle.clone();
                        let vsync = crate::vsync_actor::spawn_vsync_actor(
                            move |msg| {
                                if let Some(handle) = weak_handle.upgrade() {
                                    handle.send(actor_scheduler::Message::Data(EngineData::FromVSync(msg)))
                                } else {
                                    Err(actor_scheduler::SendError)
                                }
                            },
                            refresh_rate,
                        );
                        info!("Engine: VSync actor spawned at {:.2} Hz", refresh_rate);
                        self.vsync_actor = Some(vsync);
                    } else {
                        log::error!("Engine: Cannot spawn VSync - engine handle not set");
                    }
                }
            }
            EngineControl::Quit => {
                info!("Engine: Quit requested from app");
                let _ = self.driver.send(DriverCommand::Shutdown);
            }
            EngineControl::DriverAck => {
                // Ignore driver acks
            }
            EngineControl::SetEngineHandle(handle) => {
                info!("Engine: Engine handle set for VSync creation");
                self.engine_handle = Some(handle);
            }
            EngineControl::RegisterApp(app_handle) => {
                info!("Engine: Application registered");
                self.app = Some(app_handle);
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
            AppManagement::ResizeRequest(_, _) => {
                // TODO: Implement window resize
            }
            AppManagement::SetCursorIcon(_) => {
                // TODO: Implement cursor icon change
            }
        }
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

#[cfg(test)]
mod tests;
