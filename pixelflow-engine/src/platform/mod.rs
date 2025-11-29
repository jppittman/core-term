pub mod waker;
pub mod vsync;

use crate::channel::{create_engine_channels, DriverCommand, EngineCommand, EngineSender};
use crate::config::EngineConfig;
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, DriverConfig, RenderSnapshot};
use crate::input::MouseButton;
use crate::platform::vsync::VsyncActor;
use crate::traits::{AppAction, AppState, Application, EngineEvent};
use anyhow::{Context, Result};
use log::info;
use pixelflow_render::rasterizer::{materialize_into, ScreenViewMut};
use std::sync::mpsc::Receiver;

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
    control_rx: Receiver<EngineCommand>,
    display_rx: Receiver<EngineCommand>,
    engine_sender: EngineSender,
    _vsync_actor: VsyncActor,
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
    pub fn new(config: EngineConfig) -> Result<Self> {
        info!("EnginePlatform::new() - Creating channel-based platform");

        let channels = create_engine_channels(64);
        let driver = PlatformDriver::new(channels.engine_sender.clone())
            .context("Failed to create display driver")?;

        let vsync_actor = VsyncActor::spawn(channels.engine_sender.clone(), config.performance.target_fps)?;

        Ok(Self {
            driver,
            config: config.into(),
            control_rx: channels.control_rx,
            display_rx: channels.display_rx,
            engine_sender: channels.engine_sender,
            _vsync_actor: vsync_actor,
        })
    }

    pub fn create_waker(&self) -> Box<dyn crate::platform::waker::EventLoopWaker> {
        Box::new(PlatformWaker {
            sender: self.engine_sender.clone(),
        })
    }

    pub fn run(self, app: impl Application + Send + 'static) -> Result<()> {
        info!("EnginePlatform::run() - Starting");

        let driver_handle = self.driver.clone();
        let control_rx = self.control_rx;
        let display_rx = self.display_rx;
        let config = self.config.clone();

        // Spawn engine thread
        std::thread::spawn(move || {
            if let Err(e) = engine_loop(app, driver_handle, control_rx, display_rx) {
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
) -> Result<()> {
    info!("Engine loop started");

    let mut framebuffer: Option<Box<[u8]>> = None;
    let mut width_px = 0u32;
    let mut height_px = 0u32;
    let mut scale_factor = 1.0f64;
    let mut needs_redraw = false;

    loop {
        // 1. Drain control channel (high priority, unbounded)
        loop {
            match control_rx.try_recv() {
                Ok(EngineCommand::Doorbell) => continue,
                Ok(EngineCommand::Tick) => {
                    let action = app.on_event(EngineEvent::Tick);
                    match handle_action(action, &driver)? {
                        ActionResult::Continue => {}
                        ActionResult::Redraw => needs_redraw = true,
                        ActionResult::Shutdown => return Ok(()),
                    }
                }
                Ok(EngineCommand::PresentComplete(snap)) => {
                    framebuffer = Some(snap.framebuffer);
                }
                Ok(EngineCommand::DriverAck) => {}
                Ok(EngineCommand::DisplayEvent(_)) => {
                    // Shouldn't happen on control channel, but handle it
                }
                Err(_) => break,
            }
        }

        // 2. Batch display events (low priority, bounded)
        let mut budget = 10;
        while budget > 0 {
            match display_rx.try_recv() {
                Ok(EngineCommand::DisplayEvent(evt)) => {
                    // Track resize
                    if let DisplayEvent::Resize { width_px: w, height_px: h } = &evt {
                        width_px = *w;
                        height_px = *h;
                        needs_redraw = true;
                    }
                    // Track scale factor from mouse events
                    if let DisplayEvent::MouseButtonPress { scale_factor: sf, .. }
                        | DisplayEvent::MouseButtonRelease { scale_factor: sf, .. }
                        | DisplayEvent::MouseMove { scale_factor: sf, .. } = &evt
                    {
                        scale_factor = *sf;
                    }

                    if let Some(engine_evt) = map_display_event(&evt) {
                        let action = app.on_event(engine_evt);
                        match handle_action(action, &driver)? {
                            ActionResult::Continue => {}
                            ActionResult::Redraw => needs_redraw = true,
                            ActionResult::Shutdown => {
                                info!("Engine loop: shutdown requested");
                                return Ok(());
                            }
                        }
                    }

                    // Handle CloseRequested
                    if matches!(evt, DisplayEvent::CloseRequested) {
                        info!("Engine loop: close requested");
                        driver.send(DriverCommand::Shutdown)?;
                        return Ok(());
                    }

                    budget -= 1;
                }
                _ => break,
            }
        }

        // 3. Render if needed
        if needs_redraw && width_px > 0 && height_px > 0 {
            let app_state = AppState {
                width_px,
                height_px,
                scale_factor,
            };

            if let Some(surface) = app.render(&app_state) {
                let mut fb = framebuffer.take().unwrap_or_else(|| {
                    vec![0u8; (width_px * height_px * 4) as usize].into_boxed_slice()
                });

                // Convert u8 framebuffer to u32 slice
                let (prefix, pixels, suffix) = unsafe { fb.align_to_mut::<u32>() };
                if prefix.is_empty() && suffix.is_empty() {
                    let mut screen = ScreenViewMut::new(
                        pixels,
                        width_px as usize,
                        height_px as usize,
                        0,
                        0,
                    );
                    materialize_into(&mut screen, surface.as_ref());
                }

                // Convert back to Box<[u8]>
                let fb = unsafe {
                    let ptr = pixels.as_mut_ptr() as *mut u8;
                    let len = pixels.len() * 4;
                    Box::from_raw(std::slice::from_raw_parts_mut(ptr, len))
                };

                let snapshot = RenderSnapshot {
                    framebuffer: fb,
                    width_px,
                    height_px,
                };
                driver.send(DriverCommand::Present(snapshot))?;
            }

            needs_redraw = false;
        }

        // 4. Sleep on control channel (doorbell wakes us)
        if budget > 0 {
            match control_rx.recv() {
                Ok(EngineCommand::Doorbell) => {}
                Ok(EngineCommand::Tick) => {
                    let action = app.on_event(EngineEvent::Tick);
                    match handle_action(action, &driver)? {
                        ActionResult::Continue => {}
                        ActionResult::Redraw => needs_redraw = true,
                        ActionResult::Shutdown => return Ok(()),
                    }
                }
                Ok(EngineCommand::PresentComplete(snap)) => {
                    framebuffer = Some(snap.framebuffer);
                }
                Ok(EngineCommand::DriverAck) => {}
                Ok(EngineCommand::DisplayEvent(_)) => {}
                Err(_) => {
                    info!("Engine loop: control channel closed");
                    return Ok(());
                }
            }
        }
    }
}

enum ActionResult {
    Continue,
    Redraw,
    Shutdown,
}

fn handle_action(action: AppAction, driver: &PlatformDriver) -> Result<ActionResult> {
    match action {
        AppAction::Continue => Ok(ActionResult::Continue),
        AppAction::Redraw => Ok(ActionResult::Redraw),
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

fn map_display_event(evt: &DisplayEvent) -> Option<EngineEvent> {
    match evt {
        DisplayEvent::Resize { width_px, height_px } => {
            Some(EngineEvent::Resize(*width_px, *height_px))
        }
        DisplayEvent::Key { symbol, modifiers, text } => Some(EngineEvent::KeyDown {
            key: symbol.clone(),
            mods: *modifiers,
            text: text.clone(),
        }),
        DisplayEvent::MouseButtonPress { button, x, y, .. } => {
            let btn = match button {
                1 => MouseButton::Left,
                2 => MouseButton::Middle,
                3 => MouseButton::Right,
                _ => MouseButton::Other(*button),
            };
            Some(EngineEvent::MouseClick {
                x: *x as u32,
                y: *y as u32,
                button: btn,
            })
        }
        DisplayEvent::CloseRequested => Some(EngineEvent::CloseRequested),
        DisplayEvent::FocusGained => Some(EngineEvent::FocusGained),
        DisplayEvent::FocusLost => Some(EngineEvent::FocusLost),
        DisplayEvent::PasteData { text } => Some(EngineEvent::Paste(text.clone())),
        _ => None,
    }
}
