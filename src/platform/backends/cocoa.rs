#![cfg(target_os = "macos")]

//! Minimal CocoaDriver using raw Cocoa bindings.
//!
//! This implements the RISC philosophy: the driver is dumb and minimal,
//! it just pushes pixels to the screen. All text rendering, styling, etc.
//! is done in software by the rasterizer.

use crate::platform::backends::{
    BackendEvent, CursorVisibility, Driver, DriverCommand, FocusState, PlatformState, RenderCommand,
};
use anyhow::{Context, Result};
use log::*;
use std::os::unix::io::RawFd;

#[cfg(target_os = "macos")]
use cocoa::appkit::{
    NSApp, NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreType, NSWindow,
    NSWindowStyleMask,
};
#[cfg(target_os = "macos")]
use cocoa::base::{id, nil, YES};
#[cfg(target_os = "macos")]
use cocoa::foundation::{NSAutoreleasePool, NSPoint, NSRect, NSSize, NSString};
#[cfg(target_os = "macos")]
use core_graphics::base::CGFloat;
#[cfg(target_os = "macos")]
use objc::{msg_send, sel, sel_impl};

pub struct CocoaDriver {
    #[cfg(target_os = "macos")]
    _pool: id, // NSAutoreleasePool
    #[cfg(target_os = "macos")]
    window: id,
    #[cfg(target_os = "macos")]
    _view: id,
    window_width_px: usize,
    window_height_px: usize,
    cell_width_px: usize,
    cell_height_px: usize,
    framebuffer: Vec<u8>, // RGBA pixels
}

#[cfg(target_os = "macos")]
impl CocoaDriver {
    /// Create the NSApplication singleton
    fn init_app() -> Result<id> {
        unsafe {
            let app = NSApp();
            if app == nil {
                return Err(anyhow::anyhow!("Failed to get NSApplication"));
            }
            app.setActivationPolicy_(NSApplicationActivationPolicyRegular);
            let () = msg_send![app, finishLaunching];
            Ok(app)
        }
    }

    /// Create an NSWindow
    fn create_window(width: usize, height: usize) -> Result<id> {
        unsafe {
            let window_rect = NSRect::new(
                NSPoint::new(100.0, 100.0),
                NSSize::new(width as CGFloat, height as CGFloat),
            );

            let style_mask = NSWindowStyleMask::NSTitledWindowMask
                | NSWindowStyleMask::NSClosableWindowMask
                | NSWindowStyleMask::NSMiniaturizableWindowMask
                | NSWindowStyleMask::NSResizableWindowMask;

            let window = NSWindow::alloc(nil).initWithContentRect_styleMask_backing_defer_(
                window_rect,
                style_mask,
                NSBackingStoreType::NSBackingStoreBuffered,
                false,
            );

            if window == nil {
                return Err(anyhow::anyhow!("Failed to create NSWindow"));
            }

            let title = NSString::alloc(nil).init_str("core-term");
            window.setTitle_(title);
            window.center();
            window.makeKeyAndOrderFront_(nil);

            Ok(window)
        }
    }

    /// Create an NSView for rendering
    fn create_view(width: usize, height: usize) -> Result<id> {
        unsafe {
            use cocoa::appkit::NSView;

            let view_rect = NSRect::new(
                NSPoint::new(0.0, 0.0),
                NSSize::new(width as CGFloat, height as CGFloat),
            );

            let view = NSView::alloc(nil).initWithFrame_(view_rect);
            if view == nil {
                return Err(anyhow::anyhow!("Failed to create NSView"));
            }

            Ok(view)
        }
    }
}

impl Driver for CocoaDriver {
    fn new() -> Result<Self>
    where
        Self: Sized,
    {
        #[cfg(target_os = "macos")]
        {
            info!("CocoaDriver: Initializing minimal driver with real Cocoa window");

            unsafe {
                let pool = NSAutoreleasePool::new(nil);
                let window_width_px = 800;
                let window_height_px = 600;
                let cell_width_px = 8;
                let cell_height_px = 16;

                Self::init_app().context("Failed to initialize NSApplication")?;
                let window = Self::create_window(window_width_px, window_height_px)
                    .context("Failed to create window")?;
                let view = Self::create_view(window_width_px, window_height_px)
                    .context("Failed to create view")?;

                let () = msg_send![window, setContentView: view];

                // Create framebuffer
                let framebuffer = vec![0u8; window_width_px * window_height_px * 4];

                info!("CocoaDriver: Window created at {}x{}", window_width_px, window_height_px);

                Ok(Self {
                    _pool: pool,
                    window,
                    _view: view,
                    window_width_px,
                    window_height_px,
                    cell_width_px,
                    cell_height_px,
                    framebuffer,
                })
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(anyhow::anyhow!("CocoaDriver only supported on macOS"))
        }
    }

    fn get_event_fd(&self) -> Option<RawFd> {
        // Cocoa uses its own event loop, not file descriptors
        None
    }

    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        // TODO: Process Cocoa events non-blocking using:
        // NSApp.nextEventMatchingMask(_:untilDate:inMode:dequeue:)
        // with untilDate = nil for non-blocking
        Ok(vec![])
    }

    fn get_platform_state(&self) -> PlatformState {
        PlatformState {
            event_fd: None,
            font_cell_width_px: self.cell_width_px,
            font_cell_height_px: self.cell_height_px,
            scale_factor: 1.0, // TODO: Get actual scale from NSScreen
            display_width_px: self.window_width_px as u16,
            display_height_px: self.window_height_px as u16,
        }
    }

    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()> {
        // Old-style interface - just log for now
        debug!("CocoaDriver: Ignoring {} old-style RenderCommands (use DriverCommand instead)", commands.len());
        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        #[cfg(target_os = "macos")]
        unsafe {
            // Flush display
            let () = msg_send![self.window, display];
        }
        Ok(())
    }

    fn set_title(&mut self, title: &str) {
        #[cfg(target_os = "macos")]
        unsafe {
            let ns_title = NSString::alloc(nil).init_str(title);
            let () = msg_send![self.window, setTitle: ns_title];
        }
    }

    fn bell(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            let () = msg_send![cocoa::appkit::NSApplication::sharedApplication(nil), beep];
        }
    }

    fn set_cursor_visibility(&mut self, _visibility: CursorVisibility) {
        // TODO: Implement cursor visibility with NSCursor
    }

    fn set_focus(&mut self, _focus_state: FocusState) {
        // Focus is handled by window manager
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("CocoaDriver: Cleanup");
        Ok(())
    }

    fn get_framebuffer_mut(&mut self) -> &mut [u8] {
        &mut self.framebuffer
    }

    fn get_framebuffer_size(&self) -> (usize, usize) {
        (self.window_width_px, self.window_height_px)
    }
}

// New minimal driver interface
#[cfg(target_os = "macos")]
impl CocoaDriver {
    /// Execute minimal RISC-like driver commands
    pub fn execute_driver_commands(&mut self, commands: Vec<DriverCommand>) -> Result<()> {
        for cmd in commands {
            match cmd {
                DriverCommand::SetTitle { title } => {
                    self.set_title(&title);
                }
                DriverCommand::Bell => {
                    self.bell();
                }
                DriverCommand::Present => {
                    self.present()?;
                }
            }
        }
        Ok(())
    }
}

impl Drop for CocoaDriver {
    fn drop(&mut self) {
        debug!("CocoaDriver: Dropping");
    }
}
