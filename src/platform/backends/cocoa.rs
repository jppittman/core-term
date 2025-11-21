#![cfg(target_os = "macos")]

//! Minimal CocoaDriver using raw Cocoa bindings.
//!
//! This implements the RISC philosophy: the driver is dumb and minimal,
//! it just pushes pixels to the screen. All text rendering, styling, etc.
//! is done in software by the rasterizer.

use crate::platform::backends::{
    BackendEvent, CursorVisibility, Driver, DriverCommand, FocusState, MouseButton, PlatformState,
    RenderCommand,
};
use anyhow::{Context, Result};
use log::*;
use std::os::unix::io::RawFd;
use std::sync::Arc;

#[cfg(target_os = "macos")]
use cocoa::appkit::{
    NSApp, NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreType,
    NSWindow, NSWindowStyleMask,
};
#[cfg(target_os = "macos")]
use cocoa::base::{id, nil, YES};
#[cfg(target_os = "macos")]
use cocoa::foundation::{NSAutoreleasePool, NSPoint, NSRect, NSSize, NSString};
#[cfg(target_os = "macos")]
use core_graphics::base::CGFloat;
#[cfg(target_os = "macos")]
use core_graphics::color_space::CGColorSpace;
#[cfg(target_os = "macos")]
use core_graphics::data_provider::CGDataProvider;
#[cfg(target_os = "macos")]
use core_graphics::image::CGImage;
#[cfg(target_os = "macos")]
use objc::{class, msg_send, sel, sel_impl};

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
            let () = msg_send![app, activateIgnoringOtherApps: YES];
            Ok(app)
        }
    }

    /// Extract modifier flags from an NSEvent and convert to our Modifiers type
    unsafe fn extract_modifiers(event: id) -> crate::keys::Modifiers {
        use crate::keys::Modifiers;
        use cocoa::appkit::NSEventModifierFlags;

        let flags: u64 = msg_send![event, modifierFlags];
        let mut modifiers = Modifiers::empty();

        if flags & NSEventModifierFlags::NSShiftKeyMask.bits() != 0 {
            modifiers |= Modifiers::SHIFT;
        }
        if flags & NSEventModifierFlags::NSControlKeyMask.bits() != 0 {
            modifiers |= Modifiers::CONTROL;
        }
        if flags & NSEventModifierFlags::NSAlternateKeyMask.bits() != 0 {
            modifiers |= Modifiers::ALT;
        }
        if flags & NSEventModifierFlags::NSCommandKeyMask.bits() != 0 {
            modifiers |= Modifiers::SUPER;
        }

        modifiers
    }

    /// Map macOS keycode to our KeySymbol abstraction
    fn map_keycode_to_symbol(keycode: u16) -> crate::keys::KeySymbol {
        use crate::keys::KeySymbol;

        match keycode {
            0x00 => KeySymbol::Char('a'),
            0x01 => KeySymbol::Char('s'),
            0x02 => KeySymbol::Char('d'),
            0x03 => KeySymbol::Char('f'),
            0x04 => KeySymbol::Char('h'),
            0x05 => KeySymbol::Char('g'),
            0x06 => KeySymbol::Char('z'),
            0x07 => KeySymbol::Char('x'),
            0x08 => KeySymbol::Char('c'),
            0x09 => KeySymbol::Char('v'),
            0x0B => KeySymbol::Char('b'),
            0x0C => KeySymbol::Char('q'),
            0x0D => KeySymbol::Char('w'),
            0x0E => KeySymbol::Char('e'),
            0x0F => KeySymbol::Char('r'),
            0x10 => KeySymbol::Char('y'),
            0x11 => KeySymbol::Char('t'),
            0x12 => KeySymbol::Char('1'),
            0x13 => KeySymbol::Char('2'),
            0x14 => KeySymbol::Char('3'),
            0x15 => KeySymbol::Char('4'),
            0x16 => KeySymbol::Char('6'),
            0x17 => KeySymbol::Char('5'),
            0x18 => KeySymbol::Char('='),
            0x19 => KeySymbol::Char('9'),
            0x1A => KeySymbol::Char('7'),
            0x1B => KeySymbol::Char('-'),
            0x1C => KeySymbol::Char('8'),
            0x1D => KeySymbol::Char('0'),
            0x1E => KeySymbol::Char(']'),
            0x1F => KeySymbol::Char('o'),
            0x20 => KeySymbol::Char('u'),
            0x21 => KeySymbol::Char('['),
            0x22 => KeySymbol::Char('i'),
            0x23 => KeySymbol::Char('p'),
            0x24 => KeySymbol::Enter,
            0x25 => KeySymbol::Char('l'),
            0x26 => KeySymbol::Char('j'),
            0x27 => KeySymbol::Char('\''),
            0x28 => KeySymbol::Char('k'),
            0x29 => KeySymbol::Char(';'),
            0x2A => KeySymbol::Char('\\'),
            0x2B => KeySymbol::Char(','),
            0x2C => KeySymbol::Char('/'),
            0x2D => KeySymbol::Char('n'),
            0x2E => KeySymbol::Char('m'),
            0x2F => KeySymbol::Char('.'),
            0x30 => KeySymbol::Tab,
            0x31 => KeySymbol::Char(' '),
            0x32 => KeySymbol::Char('`'),
            0x33 => KeySymbol::Backspace,
            0x35 => KeySymbol::Escape,
            0x7B => KeySymbol::Left,
            0x7C => KeySymbol::Right,
            0x7D => KeySymbol::Down,
            0x7E => KeySymbol::Up,
            0x73 => KeySymbol::Home,
            0x77 => KeySymbol::End,
            0x74 => KeySymbol::PageUp,
            0x79 => KeySymbol::PageDown,
            0x75 => KeySymbol::Delete,
            _ => KeySymbol::Char('\0'), // Unknown key
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
            let () = msg_send![window, setOpaque: YES];
            let black_color: id = msg_send![class!(NSColor), blackColor];
            let () = msg_send![window, setBackgroundColor: black_color];
            window.center();
            window.makeKeyAndOrderFront_(nil);
            let () = msg_send![window, orderFrontRegardless];

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

            let () = msg_send![view, setAutoresizingMask: 18u64];

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

                // Create framebuffer (initialized to black)
                let framebuffer = vec![0u8; window_width_px * window_height_px * 4];

                info!(
                    "CocoaDriver: Window created at {}x{}",
                    window_width_px, window_height_px
                );

                let mut driver = Self {
                    _pool: pool,
                    window,
                    _view: view,
                    window_width_px,
                    window_height_px,
                    cell_width_px,
                    cell_height_px,
                    framebuffer,
                };

                driver.present().context("Failed initial present")?;

                Ok(driver)
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
        #[cfg(target_os = "macos")]
        unsafe {
            use cocoa::appkit::{NSEventMask, NSEventType};

            let mut backend_events = Vec::new();
            let app = NSApp();

            // Use distantPast for non-blocking poll (returns immediately if no events)
            let distant_past: id = msg_send![class!(NSDate), distantPast];

            loop {
                let event: id = msg_send![
                    app,
                    nextEventMatchingMask: NSEventMask::NSAnyEventMask.bits()
                    untilDate: distant_past
                    inMode: cocoa::foundation::NSDefaultRunLoopMode
                    dequeue: YES
                ];

                if event == nil {
                    break;
                }

                let event_type: u64 = msg_send![event, type];

                match event_type {
                    t if t == NSEventType::NSKeyDown as u64 => {
                        let chars: id = msg_send![event, characters];
                        let text = if chars != nil {
                            let s = cocoa::foundation::NSString::UTF8String(chars);
                            if !s.is_null() {
                                std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned()
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };

                        let key_code: u16 = msg_send![event, keyCode];
                        let symbol = Self::map_keycode_to_symbol(key_code);
                        let modifiers = Self::extract_modifiers(event);

                        backend_events.push(BackendEvent::Key {
                            symbol,
                            modifiers,
                            text,
                        });
                    }
                    t if t == NSEventType::NSLeftMouseDown as u64 => {
                        let location: NSPoint = msg_send![event, locationInWindow];
                        let modifiers = Self::extract_modifiers(event);
                        backend_events.push(BackendEvent::MouseButtonPress {
                            button: MouseButton::Left,
                            x: location.x as u16,
                            y: (self.window_height_px as f64 - location.y) as u16,
                            modifiers,
                        });
                    }
                    t if t == NSEventType::NSRightMouseDown as u64 => {
                        let location: NSPoint = msg_send![event, locationInWindow];
                        let modifiers = Self::extract_modifiers(event);
                        backend_events.push(BackendEvent::MouseButtonPress {
                            button: MouseButton::Right,
                            x: location.x as u16,
                            y: (self.window_height_px as f64 - location.y) as u16,
                            modifiers,
                        });
                    }
                    t if t == NSEventType::NSLeftMouseUp as u64 => {
                        let location: NSPoint = msg_send![event, locationInWindow];
                        let modifiers = Self::extract_modifiers(event);
                        backend_events.push(BackendEvent::MouseButtonRelease {
                            button: MouseButton::Left,
                            x: location.x as u16,
                            y: (self.window_height_px as f64 - location.y) as u16,
                            modifiers,
                        });
                    }
                    t if t == NSEventType::NSRightMouseUp as u64 => {
                        let location: NSPoint = msg_send![event, locationInWindow];
                        let modifiers = Self::extract_modifiers(event);
                        backend_events.push(BackendEvent::MouseButtonRelease {
                            button: MouseButton::Right,
                            x: location.x as u16,
                            y: (self.window_height_px as f64 - location.y) as u16,
                            modifiers,
                        });
                    }
                    t if t == NSEventType::NSMouseMoved as u64
                        || t == NSEventType::NSLeftMouseDragged as u64
                        || t == NSEventType::NSRightMouseDragged as u64 =>
                    {
                        let location: NSPoint = msg_send![event, locationInWindow];
                        let modifiers = Self::extract_modifiers(event);
                        backend_events.push(BackendEvent::MouseMove {
                            x: location.x as u16,
                            y: (self.window_height_px as f64 - location.y) as u16,
                            modifiers,
                        });
                    }
                    _ => {
                        let () = msg_send![app, sendEvent: event];
                    }
                }
            }

            Ok(backend_events)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(vec![])
        }
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
        debug!(
            "CocoaDriver: Ignoring {} old-style RenderCommands (use DriverCommand instead)",
            commands.len()
        );
        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        #[cfg(target_os = "macos")]
        unsafe {
            self.draw_framebuffer_to_view()
                .context("Failed to draw framebuffer to view")?;
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
    /// Converts the framebuffer to a CGImage and draws it to the NSView
    unsafe fn draw_framebuffer_to_view(&mut self) -> Result<()> {
        use core_graphics::geometry::{CGPoint, CGRect, CGSize};

        // Create CGImage from framebuffer
        // The framebuffer is RGBA, 8 bits per component, 32 bits per pixel
        let width = self.window_width_px;
        let height = self.window_height_px;
        let bytes_per_row = width * 4;

        // CGDataProvider needs Arc<Vec<u8>> for shared ownership
        // TODO(performance): This copies the entire framebuffer. Consider using
        // a double-buffer strategy or CALayer for better performance
        let data = Arc::new(self.framebuffer.clone());

        let provider = CGDataProvider::from_buffer(data);
        let color_space = CGColorSpace::create_device_rgb();

        // Create CGImage from raw pixel data
        // CGImageAlphaInfo values:
        // kCGImageAlphaNone = 0
        // kCGImageAlphaPremultipliedLast = 1  (RGBA, alpha premultiplied)
        // kCGImageAlphaLast = 2  (RGBA, straight alpha)
        // kCGImageAlphaNoneSkipLast = 6  (RGB_, ignore A)
        // For RGBA with straight alpha: use 2
        let bitmap_info = 2u32; // kCGImageAlphaLast = RGBA with straight alpha

        let image = CGImage::new(
            width,
            height,
            8,  // bits per component (RGBA = 8 bits each)
            32, // bits per pixel (RGBA = 4 * 8 = 32)
            bytes_per_row,
            &color_space,
            bitmap_info,
            &provider,
            false, // should_interpolate
            0,     // rendering_intent (0 = default)
        );

        // Get the view's graphics context and draw the image
        // lockFocus makes the view the current drawing context
        let _: () = msg_send![self._view, lockFocus];

        // Get the current NSGraphicsContext
        let ns_context: id = msg_send![class!(NSGraphicsContext), currentContext];
        if ns_context == nil {
            let _: () = msg_send![self._view, unlockFocus];
            return Err(anyhow::anyhow!("Failed to get NSGraphicsContext"));
        }

        // Get the CGContext from NSGraphicsContext
        let cg_context: *mut core_graphics::sys::CGContext = msg_send![ns_context, CGContext];
        if cg_context.is_null() {
            let _: () = msg_send![self._view, unlockFocus];
            return Err(anyhow::anyhow!(
                "Failed to get CGContext from NSGraphicsContext"
            ));
        }

        // Draw the image to fill the entire view
        let rect = CGRect::new(
            &CGPoint::new(0.0, 0.0),
            &CGSize::new(width as CGFloat, height as CGFloat),
        );

        let context = core_graphics::context::CGContext::from_existing_context_ptr(cg_context);
        context.draw_image(rect, &image);

        // Flush and unlock
        let _: () = msg_send![ns_context, flushGraphics];
        let _: () = msg_send![self._view, unlockFocus];

        // Tell the window to display
        let _: () = msg_send![self.window, display];

        Ok(())
    }

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
