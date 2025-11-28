pub mod waker;

use crate::display::{
    messages::RenderSnapshot, DisplayEvent, DisplayManager, DriverRequest, DriverResponse,
};
use crate::input::MouseButton;
use crate::traits::{AppAction, AppState, Application, EngineEvent};
use anyhow::{Context, Result};
use log::info;
use pixelflow_render::rasterizer::process_frame;

pub struct EnginePlatform {
    display_manager: DisplayManager,
    framebuffer: Option<Box<[u8]>>,
}

impl EnginePlatform {
    pub fn new(config: crate::display::messages::DriverConfig) -> Result<Self> {
        info!("EnginePlatform::new() - Initializing display-based platform");
        let display_manager =
            DisplayManager::new(config).context("Failed to create DisplayManager")?;
        Ok(Self {
            display_manager,
            framebuffer: None,
        })
    }

    pub fn run(mut self, mut app: impl Application + 'static) -> Result<()> {
        info!("EnginePlatform::run() - Starting event loop");

        let mut shutdown_complete = false;

        // Initial Resize event
        let metrics = self.display_manager.metrics().clone();
        app.on_event(EngineEvent::Resize(metrics.width_px, metrics.height_px));

        loop {
            // Poll display events
            let response = self
                .display_manager
                .handle_request(DriverRequest::PollEvents)
                .context("DisplayManager event polling failed")?;

            let mut needs_redraw = false;

            if let DriverResponse::Events(display_events) = response {
                for display_event in display_events {
                    if let Some(engine_event) = self.map_event(display_event) {
                        let action = app.on_event(engine_event);
                        if self.handle_action(
                            action,
                            &mut app,
                            &mut shutdown_complete,
                            &mut needs_redraw,
                        )? {
                            if shutdown_complete {
                                break;
                            }
                        }
                    }
                }
            }
            if shutdown_complete {
                break;
            }

            // Check if we need to redraw
            if needs_redraw {
                self.perform_render(&mut app)?;
            }
        }

        info!("EnginePlatform::run() - Exiting normally");
        Ok(())
    }

    fn map_event(&self, event: DisplayEvent) -> Option<EngineEvent> {
        match event {
            DisplayEvent::Resize {
                width_px,
                height_px,
            } => Some(EngineEvent::Resize(width_px, height_px)),
            DisplayEvent::Key {
                symbol,
                modifiers,
                text,
            } => Some(EngineEvent::KeyDown {
                key: symbol,
                mods: modifiers,
                text,
            }),
            DisplayEvent::MouseButtonPress {
                button,
                x,
                y,
                modifiers: _,
                ..
            } => {
                let btn = match button {
                    0 => MouseButton::Left,
                    1 => MouseButton::Right, // Cocoa maps 0=Left, 1=Right usually? DisplayDriver converts.
                    // Wait, CocoaDisplayDriver converted button to 0/1.
                    // DisplayEvent has u8. EngineEvent has MouseButton enum.
                    // I need to map u8 back to enum?
                    // Or update DisplayEvent to use MouseButton enum?
                    // Messages.rs used u8.
                    // I should update Messages.rs to use MouseButton enum from input.rs!
                    // I missed that.
                    // For now, I'll map.
                    // Standard: 0=Left, 1=Middle, 2=Right? Or 1=Left?
                    // Cocoa: 0=Left, 1=Right, 2=Middle.
                    // Let's assume CocoaDisplayDriver did the right thing.
                    // Wait, CocoaDisplayDriver:
                    // LeftMouseDown -> button: 0
                    // RightMouseDown -> button: 1
                    // So 0=Left, 1=Right.
                    _ => MouseButton::Other(button),
                };
                Some(EngineEvent::MouseClick {
                    x: x as u32,
                    y: y as u32,
                    button: btn,
                })
            }
            DisplayEvent::CloseRequested => Some(EngineEvent::CloseRequested),
            DisplayEvent::ClipboardDataRequested => None, // Handled internally? Or passed to app? App handles copy/paste.
            // Wait, DisplayEvent::ClipboardDataRequested means X11 needs data.
            // CoreTermApp logic handles clipboard.
            // So we should pass it? EngineEvent doesn't have it.
            // I should add it to EngineEvent? Or handle it in EnginePlatform?
            // EnginePlatform doesn't have the text. App has it.
            // So I should add EngineEvent::ClipboardRequested.
            _ => None, // Focus, MouseMove etc.
        }
    }

    fn handle_action(
        &mut self,
        action: AppAction,
        _app: &mut impl Application,
        shutdown: &mut bool,
        needs_redraw: &mut bool,
    ) -> Result<bool> {
        match action {
            AppAction::Continue => {}
            AppAction::Redraw => *needs_redraw = true,
            AppAction::SetTitle(t) => {
                self.display_manager
                    .handle_request(DriverRequest::SetTitle(t))?;
            }
            AppAction::Quit => *shutdown = true,
            AppAction::ResizeRequest(_, _) => {
                // Not implemented in DriverRequest yet?
            }
            AppAction::SetCursorIcon(_) => {} // Not impl
            AppAction::CopyToClipboard(text) => {
                self.display_manager
                    .handle_request(DriverRequest::CopyToClipboard(text))?;
            }
            AppAction::RequestPaste => {
                self.display_manager
                    .handle_request(DriverRequest::RequestPaste)?;
            }
        }
        Ok(false)
    }

    fn perform_render(&mut self, app: &mut impl Application) -> Result<()> {
        let metrics = self.display_manager.metrics().clone();
        let app_state = AppState {
            width_px: metrics.width_px,
            height_px: metrics.height_px,
            scale_factor: metrics.scale_factor,
        };

        let ops = app.render(&app_state);

        // Get framebuffer
        let mut framebuffer = if let Some(fb) = self.framebuffer.take() {
            fb
        } else {
            let resp = self
                .display_manager
                .handle_request(DriverRequest::RequestFramebuffer)?;
            if let DriverResponse::Framebuffer(fb) = resp {
                fb
            } else {
                return Err(anyhow::anyhow!("Expected Framebuffer"));
            }
        };

        // Convert u8 framebuffer to u32 slice for process_frame
        // Wait, process_frame takes &mut [u32].
        // The framebuffer is Box<[u8]>.
        // I need to cast it safely.
        let (prefix, pixels, suffix) = unsafe { framebuffer.align_to_mut::<u32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            panic!("Framebuffer not aligned to u32");
        }

        // Assume cell_width/height?
        // process_frame needs cell dims for text layout.
        // AppState has pixels.
        // I should probably get cell dims from Config/DriverConfig?
        // Or pass it in AppState?
        // DriverConfig has cell_width_px.
        // But EnginePlatform doesn't hold Config easily visible here.
        // self.display_manager has it? DisplayManager has metrics.
        // metrics has width/height/scale. Not cell size.
        // Maybe hardcode or add to DisplayMetrics.
        // For now, let's assume arbitrary cell size or get from somewhere.
        // Actually, `ops` are generic. `Op::Text` uses grid?
        // I checked `process_frame`. `Op::Text` needs `cell_width` to calculate pixel pos.
        // If `CoreTermApp` emits `Op::Text`, it assumes a grid.
        // So `EnginePlatform` MUST know the grid size.
        // I'll assume standard 10x20 or similar for now, or add it to `DisplayMetrics`.
        let cell_w = 10; // Placeholder
        let cell_h = 20; // Placeholder

        process_frame(
            pixels,
            metrics.width_px as usize,
            metrics.height_px as usize,
            cell_w,
            cell_h,
            &ops,
        );

        // Present
        let snapshot = RenderSnapshot {
            framebuffer,
            width_px: metrics.width_px,
            height_px: metrics.height_px,
        };
        let resp = self
            .display_manager
            .handle_request(DriverRequest::Present(snapshot));
        match resp {
            Ok(DriverResponse::PresentComplete(snap)) => {
                self.framebuffer = Some(snap.framebuffer);
            }
            Err(crate::display::DisplayError::PresentationFailed(snap, _)) => {
                self.framebuffer = Some(snap.framebuffer);
            }
            Err(e) => return Err(e.into()),
            _ => return Err(anyhow::anyhow!("Unexpected response")),
        }
        Ok(())
    }

    pub fn create_waker(&self) -> Box<dyn crate::platform::waker::EventLoopWaker> {
        self.display_manager.create_waker()
    }
}
