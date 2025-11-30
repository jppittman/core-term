pub mod waker;

use crate::channel::{create_engine_channels, DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, DriverConfig, RenderSnapshot};
use crate::input::MouseButton;
use crate::traits::{AppAction, AppState, Application, EngineEvent};
use anyhow::{Context, Result};
use log::info;
use pixelflow_core::Scale;
use pixelflow_render::rasterizer::render_u32;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

// Platform-specific driver type alias
#[cfg(use_x11_display)]
type PlatformDriver = crate::display::drivers::X11DisplayDriver;

#[cfg(use_cocoa_display)]
type PlatformDriver = crate::display::drivers::CocoaDisplayDriver;

#[cfg(use_headless_display)]
type PlatformDriver = crate::display::drivers::HeadlessDisplayDriver;

#[cfg(use_web_display)]
type PlatformDriver = crate::display::drivers::WebDisplayDriver;

pub struct EnginePlatform {
    driver: PlatformDriver,
    config: DriverConfig,
    engine_sender: EngineSender,
    control_rx: Receiver<EngineCommand>,
    display_rx: Receiver<EngineCommand>,
}

struct PlatformWaker {
    sender: EngineSender,
}

impl crate::platform::waker::EventLoopWaker for PlatformWaker {
    fn wake(&self) -> Result<()> {
        self.sender
            .send(EngineCommand::Doorbell)
            .map_err(|e| anyhow::anyhow!("Failed to wake engine: {}", e))
    }
}

impl EnginePlatform {
    pub fn new(config: DriverConfig) -> Result<Self> {
        info!("EnginePlatform::new() - Creating channel-based platform");

        let channels = create_engine_channels(64);
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
    pub fn engine_sender(&self) -> crate::channel::EngineSender {
        self.engine_sender.clone()
    }

    pub fn run(self, app: impl Application + Send + 'static) -> Result<()> {
        info!("EnginePlatform::run() - Starting");

        let driver_handle = self.driver.clone();
        let control_rx = self.control_rx;
        let display_rx = self.display_rx;
        let config = self.config.clone();
        let target_fps = config.target_fps;

        // Spawn engine thread
        std::thread::spawn(move || {
            if let Err(e) = engine_loop(app, driver_handle, control_rx, display_rx, target_fps) {
                log::error!("Engine loop error: {}", e);
            }
        });

        // Send Configure before running
        self.driver.send(DriverCommand::Configure(config))?;

        // Run driver on main thread (blocks)
        self.driver.run()
    }
}

fn engine_loop(
    mut app: impl Application,
    driver: PlatformDriver,
    control_rx: Receiver<EngineCommand>,
    display_rx: Receiver<EngineCommand>,
    target_fps: u32,
) -> Result<()> {
    info!("Engine loop started (pull model, {} FPS)", target_fps);

    let frame_duration = Duration::from_secs_f64(1.0 / target_fps as f64);
    let mut framebuffer: Option<Box<[u8]>> = None;
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
                Ok(EngineCommand::PresentComplete(snap)) => {
                    framebuffer = Some(snap.framebuffer);
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
                        // Track physical dimensions and scale factor
                        if let DisplayEvent::Resize { width_px: w, height_px: h, scale_factor: sf } = &evt {
                            physical_width = *w;
                            physical_height = *h;
                            scale_factor = *sf;
                        }
                        // Also track scale factor from mouse events (fallback)
                        if let DisplayEvent::MouseButtonPress { scale_factor: sf, .. }
                            | DisplayEvent::MouseButtonRelease { scale_factor: sf, .. }
                            | DisplayEvent::MouseMove { scale_factor: sf, .. } = &evt
                        {
                            scale_factor = *sf;
                        }

                        if let Some(engine_evt) = map_display_event(&evt, scale_factor) {
                            let action = app.on_event(engine_evt);
                            if let ActionResult::Shutdown = handle_action(action, &driver)? {
                                info!("Engine loop: shutdown requested");
                                return Ok(());
                            }
                        }

                        // Handle CloseRequested
                        if matches!(evt, DisplayEvent::CloseRequested) {
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
                let mut fb = framebuffer.take().unwrap_or_else(|| {
                    vec![0u8; (physical_width * physical_height * 4) as usize].into_boxed_slice()
                });

                let (prefix, pixels, suffix) = unsafe { fb.align_to_mut::<u32>() };
                if prefix.is_empty() && suffix.is_empty() {
                    let scaled = Scale::new(surface, scale_factor);
                    render_u32(&scaled, pixels, physical_width as usize, physical_height as usize);
                }

                let snapshot = RenderSnapshot {
                    framebuffer: fb,
                    width_px: physical_width,
                    height_px: physical_height,
                };
                driver.send(DriverCommand::Present(snapshot))?;
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
            driver.send(DriverCommand::SetTitle(title))?;
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
        DisplayEvent::Resize { width_px, height_px, .. } => {
            // Convert physical pixels to logical pixels
            let logical_w = (*width_px as f64 / scale_factor) as u32;
            let logical_h = (*height_px as f64 / scale_factor) as u32;
            Some(EngineEvent::Resize(logical_w, logical_h))
        }
        DisplayEvent::Key { symbol, modifiers, text } => Some(EngineEvent::KeyDown {
            key: symbol.clone(),
            mods: *modifiers,
            text: text.clone(),
        }),
        DisplayEvent::MouseButtonPress { button, x, y, .. } => {
            // Convert physical pixels to logical pixels
            let logical_x = (*x as f64 / scale_factor) as u32;
            let logical_y = (*y as f64 / scale_factor) as u32;
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
            let logical_x = (*x as f64 / scale_factor) as u32;
            let logical_y = (*y as f64 / scale_factor) as u32;
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
        DisplayEvent::MouseMove { x, y, modifiers, .. } => {
            let logical_x = (*x as f64 / scale_factor) as u32;
            let logical_y = (*y as f64 / scale_factor) as u32;
            Some(EngineEvent::MouseMove {
                x: logical_x,
                y: logical_y,
                mods: *modifiers,
            })
        }
        DisplayEvent::CloseRequested => Some(EngineEvent::CloseRequested),
        DisplayEvent::FocusGained => Some(EngineEvent::FocusGained),
        DisplayEvent::FocusLost => Some(EngineEvent::FocusLost),
        DisplayEvent::PasteData { text } => Some(EngineEvent::Paste(text.clone())),
        _ => None,
    }
}
