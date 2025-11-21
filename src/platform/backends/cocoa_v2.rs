// src/platform/backends/cocoa_v2.rs

//! Minimal Cocoa driver using raw Cocoa bindings.
//!
//! This is a RISC-like driver that only implements pixel blitting.
//! All text rendering, styling, etc. is done in software.

#![cfg(target_os = "macos")]

use crate::platform::backends::{BackendEvent, CursorVisibility, Driver, DriverCommand, FocusState, PlatformState, RenderCommand};
use anyhow::{Context, Result};
use cocoa::appkit::{NSApp, NSApplication, NSApplicationActivationPolicyRegular, NSWindow, NSWindowStyleMask, NSBackingStoreType, NSView};
use cocoa::base::{id, nil};
use cocoa::foundation::{NSAutoreleasePool, NSPoint, NSRect, NSSize, NSString};
use core_graphics::base::CGFloat;
use core_graphics::color_space::CGColorSpace;
use core_graphics::context::CGContext;
use core_graphics::data_provider::CGDataProvider;
use core_graphics::image::CGImage;
use log::*;
use objc::runtime::{Class, Object, Sel};
use objc::{class, msg_send, sel, sel_impl};
use std::os::unix::io::RawFd;
use std::sync::{Arc, Mutex};

/// Shared state between the app and the view
struct CocoaState {
    pixel_buffer: Vec<u8>,
    buffer_width: usize,
    buffer_height: usize,
    needs_display: bool,
}

pub struct CocoaDriver {
    _pool: id, // NSAutoreleasePool
    window: id,
    view: id,
    state: Arc<Mutex<CocoaState>>,
    window_width_px: usize,
    window_height_px: usize,
    cell_width_px: usize,
    cell_height_px: usize,
}

impl CocoaDriver {
    fn create_window(width: usize, height: usize) -> Result<id> {
        unsafe {
            // Create the NSApplication if it doesn't exist
            let app = NSApp();
            if app == nil {
                return Err(anyhow::anyhow!("Failed to get NSApplication"));
            }

            app.setActivationPolicy_(NSApplicationActivationPolicyRegular);

            // Create window
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

    fn create_view(width: usize, height: usize, state: Arc<Mutex<CocoaState>>) -> Result<id> {
        unsafe {
            let view_rect = NSRect::new(
                NSPoint::new(0.0, 0.0),
                NSSize::new(width as CGFloat, height as CGFloat),
            );

            // For now, create a basic NSView
            // TODO: Create custom view class that can handle our pixel blitting
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
        info!("CocoaDriver: Initializing with real Cocoa window");

        unsafe {
            let pool = NSAutoreleasePool::new(nil);

            let window_width_px = 800;
            let window_height_px = 600;
            let cell_width_px = 8;
            let cell_height_px = 16;

            let state = Arc::new(Mutex::new(CocoaState {
                pixel_buffer: vec![0; window_width_px * window_height_px * 4],
                buffer_width: window_width_px,
                buffer_height: window_height_px,
                needs_display: true,
            }));

            let window = Self::create_window(window_width_px, window_height_px)
                .context("Failed to create Cocoa window")?;

            let view = Self::create_view(window_width_px, window_height_px, state.clone())
                .context("Failed to create Cocoa view")?;

            let () = msg_send![window, setContentView: view];

            info!("CocoaDriver: Window and view created successfully");

            Ok(Self {
                _pool: pool,
                window,
                view,
                state,
                window_width_px,
                window_height_px,
                cell_width_px,
                cell_height_px,
            })
        }
    }

    fn get_event_fd(&self) -> Option<RawFd> {
        // Cocoa uses an event loop, not file descriptors
        None
    }

    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        // TODO: Process Cocoa events (NSEvent)
        // For now, return empty
        Ok(vec![])
    }

    fn get_platform_state(&self) -> PlatformState {
        PlatformState {
            event_fd: None,
            font_cell_width_px: self.cell_width_px,
            font_cell_height_px: self.cell_height_px,
            scale_factor: 1.0, // TODO: Get from NSScreen
            display_width_px: self.window_width_px as u16,
            display_height_px: self.window_height_px as u16,
        }
    }

    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()> {
        // This is the old interface - for now, just log
        info!("CocoaDriver: Received {} old-style render commands", commands.len());
        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        unsafe {
            // Tell the view to redraw
            let () = msg_send![self.view, setNeedsDisplay: true];
        }
        Ok(())
    }

    fn set_title(&mut self, title: &str) {
        unsafe {
            let ns_title = NSString::alloc(nil).init_str(title);
            let () = msg_send![self.window, setTitle: ns_title];
        }
    }

    fn bell(&mut self) {
        info!("CocoaDriver: Bell (beep)");
        // TODO: NSBeep()
    }

    fn set_cursor_visibility(&mut self, _visibility: CursorVisibility) {
        // TODO: Implement cursor visibility
    }

    fn set_focus(&mut self, _focus_state: FocusState) {
        // Focus is handled by window manager
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("CocoaDriver: Cleanup");
        Ok(())
    }
}

// For the new minimal driver commands
impl CocoaDriver {
    pub fn execute_driver_commands(&mut self, commands: Vec<DriverCommand>) -> Result<()> {
        for cmd in commands {
            match cmd {
                DriverCommand::Clear { r, g, b, a } => {
                    let mut state = self.state.lock().unwrap();
                    for pixel in state.pixel_buffer.chunks_exact_mut(4) {
                        pixel[0] = r;
                        pixel[1] = g;
                        pixel[2] = b;
                        pixel[3] = a;
                    }
                    state.needs_display = true;
                }
                DriverCommand::BlitPixels { x_px, y_px, width_px, height_px, rgba_data } => {
                    let mut state = self.state.lock().unwrap();
                    let buf_width = state.buffer_width;

                    for row in 0..height_px as usize {
                        let src_row_offset = row * width_px as usize * 4;
                        let dst_y = y_px as usize + row;
                        if dst_y >= state.buffer_height {
                            break;
                        }

                        let dst_row_offset = (dst_y * buf_width + x_px as usize) * 4;
                        let copy_width = (width_px as usize).min(buf_width - x_px as usize);

                        if dst_row_offset + copy_width * 4 <= state.pixel_buffer.len() &&
                           src_row_offset + copy_width * 4 <= rgba_data.len() {
                            state.pixel_buffer[dst_row_offset..dst_row_offset + copy_width * 4]
                                .copy_from_slice(&rgba_data[src_row_offset..src_row_offset + copy_width * 4]);
                        }
                    }
                    state.needs_display = true;
                }
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
        info!("CocoaDriver: Dropping");
        // Cocoa objects are reference counted, they'll clean up automatically
    }
}
