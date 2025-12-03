pub mod waker;

use crate::channel::{create_engine_channels, DriverCommand, EngineCommand, EngineSender};
use crate::config::EngineConfig;
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, WindowId};
use crate::input::MouseButton;
use crate::traits::{AppAction, AppState, Application, EngineEvent};
use anyhow::{Context, Result};
use log::info;
use pixelflow_core::Scale;
use pixelflow_render::rasterizer::render;
use pixelflow_render::Frame;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

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
    engine_sender: EngineSender<PlatformPixel>,
    control_rx: Receiver<EngineCommand<PlatformPixel>>,
    display_rx: Receiver<EngineCommand<PlatformPixel>>,
}

struct PlatformWaker {
    sender: EngineSender<PlatformPixel>,
}

impl crate::platform::waker::EventLoopWaker for PlatformWaker {
    fn wake(&self) -> Result<()> {
        self.sender
            .send(EngineCommand::Doorbell)
            .map_err(|e| anyhow::anyhow!("Failed to wake engine: {}", e))
    }
}

impl EnginePlatform {
    pub fn new(config: EngineConfig) -> Result<Self> {
        info!("EnginePlatform::new() - Creating channel-based platform");

        let channels = create_engine_channels::<PlatformPixel>(64);
        let engine_sender = channels.engine_sender.clone();
        let driver = PlatformDriver::new(channels.engine_sender)
            .context("Failed to create display driver")?;

        Ok(Self {
            driver,
            config,
            engine_sender,
            control_rx: channels.control_rx,
            display_rx: channels.display_rx,
        })
    }

    pub fn create_waker(&self) -> Box<dyn crate::platform::waker::EventLoopWaker> {
        Box::new(PlatformWaker {
            sender: self.engine_sender.clone(),
        })
    }

    /// Get a clone of the engine sender for external wake signaling.
    /// External code can call `sender.send(EngineCommand::Doorbell)` to wake the engine.
    pub fn engine_sender(&self) -> EngineSender<PlatformPixel> {
        self.engine_sender.clone()
    }

    /// Run the engine with the given application.
    ///
    /// The application's pixel type must match the platform's pixel type
    /// (e.g., `Rgba` for Cocoa, `Bgra` for X11).
    pub fn run(self, app: impl Application<PlatformPixel> + Send + 'static) -> Result<()> {
        info!("EnginePlatform::run() - Starting");

        let driver_handle = self.driver.clone();
        let control_rx = self.control_rx;
        let display_rx = self.display_rx;
        let target_fps = self.config.performance.target_fps;

        // Spawn engine thread
        std::thread::spawn(move || {
            if let Err(e) = engine_loop(app, driver_handle, control_rx, display_rx, target_fps) {
                log::error!("Engine loop error: {}", e);
            }
        });

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

fn engine_loop<A: Application<PlatformPixel>>(
    mut app: A,
    driver: PlatformDriver,
    control_rx: Receiver<EngineCommand<PlatformPixel>>,
    display_rx: Receiver<EngineCommand<PlatformPixel>>,
    target_fps: u32,
) -> Result<()> {
    info!("Engine loop started (pull model, {} FPS)", target_fps);

    let frame_duration = Duration::from_secs_f64(1.0 / target_fps as f64);
    // Typed framebuffer - no alignment games, just Vec<PlatformPixel>
    let mut framebuffer: Option<Frame<PlatformPixel>> = None;
    // Physical dimensions (for framebuffer/display)
    let mut physical_width = 0u32;
    let mut physical_height = 0u32;
    // Scale factor for physical <-> logical conversion
    let mut scale_factor = 1.0f64;

    loop {
        let frame_start = Instant::now();
        let deadline = frame_start + frame_duration;

        // 1. Process events until next frame deadline
        loop {
            let timeout = deadline.saturating_duration_since(Instant::now());
            if timeout.is_zero() {
                break;
            }

            // Try control channel first (high priority)
            match control_rx.recv_timeout(timeout) {
                Ok(EngineCommand::Doorbell) => {
                    let action = app.on_event(EngineEvent::Wake);
                    if let ActionResult::Shutdown = handle_action(action, &driver)? {
                        return Ok(());
                    }
                }
                Ok(EngineCommand::PresentComplete(frame)) => {
                    // Reuse the returned frame for next render
                    framebuffer = Some(frame);
                }
                Ok(EngineCommand::DriverAck) => {}
                Ok(EngineCommand::DisplayEvent(_)) => {
                    // Shouldn't happen on control channel
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Deadline reached, time to render
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    info!("Engine loop: control channel closed");
                    return Ok(());
                }
            }

            // Drain display events (low priority, bounded)
            for _ in 0..10 {
                match display_rx.try_recv() {
                    Ok(EngineCommand::DisplayEvent(evt)) => {
                        // Track physical dimensions and scale factor from window events
                        match &evt {
                            DisplayEvent::WindowCreated {
                                width_px: w,
                                height_px: h,
                                scale: sf,
                                ..
                            } => {
                                physical_width = *w;
                                physical_height = *h;
                                scale_factor = *sf;
                                framebuffer = None;
                            }
                            DisplayEvent::Resized {
                                width_px: w,
                                height_px: h,
                                ..
                            } => {
                                physical_width = *w;
                                physical_height = *h;
                                framebuffer = None;
                            }
                            DisplayEvent::ScaleChanged { scale: sf, .. } => {
                                scale_factor = *sf;
                                framebuffer = None;
                            }
                            _ => {}
                        }

                        if let Some(engine_evt) = map_display_event(&evt, scale_factor) {
                            let action = app.on_event(engine_evt);
                            if let ActionResult::Shutdown = handle_action(action, &driver)? {
                                info!("Engine loop: shutdown requested");
                                return Ok(());
                            }
                        }

                        // Handle CloseRequested
                        if matches!(evt, DisplayEvent::CloseRequested { .. }) {
                            info!("Engine loop: close requested");
                            driver.send(DriverCommand::Shutdown)?;
                            return Ok(());
                        }
                    }
                    _ => break,
                }
            }
        }

        // 2. Pull frame from app at vsync tick
        if physical_width > 0 && physical_height > 0 {
            let logical_width = (physical_width as f64 / scale_factor) as u32;
            let logical_height = (physical_height as f64 / scale_factor) as u32;

            let app_state = AppState {
                width_px: logical_width,
                height_px: logical_height,
            };

            // Pull model: app returns None if nothing to render
            if let Some(surface) = app.render(&app_state) {
                // Get or create typed framebuffer - no alignment issues!
                let mut frame = framebuffer
                    .take()
                    .unwrap_or_else(|| Frame::new(physical_width, physical_height));

                // Render directly into the typed frame
                let scaled = Scale::new(surface, scale_factor);
                render::<PlatformPixel, _>(&scaled, &mut frame);

                // Send typed frame to driver
                driver.send(DriverCommand::Present {
                    id: WindowId::PRIMARY,
                    frame,
                })?;
            }
        }
    }
}

enum ActionResult {
    Continue,
    Shutdown,
}

fn handle_action(action: AppAction, driver: &PlatformDriver) -> Result<ActionResult> {
    match action {
        AppAction::Continue => Ok(ActionResult::Continue),
        AppAction::Redraw => {
            // Pull model: engine calls render() at vsync rate, app returns None if nothing to show
            log::warn!("AppAction::Redraw is deprecated - engine uses pull model");
            Ok(ActionResult::Continue)
        }
        AppAction::Quit => Ok(ActionResult::Shutdown),
        AppAction::SetTitle(title) => {
            driver.send(DriverCommand::SetTitle {
                id: WindowId::PRIMARY,
                title,
            })?;
            Ok(ActionResult::Continue)
        }
        AppAction::CopyToClipboard(text) => {
            driver.send(DriverCommand::CopyToClipboard(text))?;
            Ok(ActionResult::Continue)
        }
        AppAction::RequestPaste => {
            driver.send(DriverCommand::RequestPaste)?;
            Ok(ActionResult::Continue)
        }
        AppAction::ResizeRequest(_, _) => Ok(ActionResult::Continue),
        AppAction::SetCursorIcon(_) => Ok(ActionResult::Continue),
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
            Some(EngineEvent::Resize(logical_w, logical_h))
        }
        DisplayEvent::WindowDestroyed { id } => {
            log::info!("Engine: WindowDestroyed {:?}", id);
            None
        }
        DisplayEvent::CloseRequested { .. } => Some(EngineEvent::CloseRequested),
        DisplayEvent::ScaleChanged { scale, .. } => Some(EngineEvent::ScaleChanged(*scale)),
        DisplayEvent::Key {
            symbol,
            modifiers,
            text,
            ..
        } => Some(EngineEvent::KeyDown {
            key: *symbol,
            mods: *modifiers,
            text: text.clone(),
        }),
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
            Some(EngineEvent::MouseClick {
                x: logical_x,
                y: logical_y,
                button: btn,
            })
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
            Some(EngineEvent::MouseRelease {
                x: logical_x,
                y: logical_y,
                button: btn,
            })
        }
        DisplayEvent::MouseMove {
            x, y, modifiers, ..
        } => {
            let logical_x = (*x as f64 / scale_factor).max(0.0) as u32;
            let logical_y = (*y as f64 / scale_factor).max(0.0) as u32;
            Some(EngineEvent::MouseMove {
                x: logical_x,
                y: logical_y,
                mods: *modifiers,
            })
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
            Some(EngineEvent::MouseScroll {
                x: logical_x,
                y: logical_y,
                dx: *dx,
                dy: *dy,
                mods: *modifiers,
            })
        }
        DisplayEvent::FocusGained { .. } => Some(EngineEvent::FocusGained),
        DisplayEvent::FocusLost { .. } => Some(EngineEvent::FocusLost),
        DisplayEvent::PasteData { text } => Some(EngineEvent::Paste(text.clone())),
        DisplayEvent::ClipboardDataRequested => {
            log::trace!("Engine: ClipboardDataRequested (ignored)");
            None
        }
    }
}
