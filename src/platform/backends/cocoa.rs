#![cfg(target_os = "macos")]

//! Minimal CocoaDriver using raw Cocoa bindings with CALayer, HiDPI, and proper coordinate system.
//!
//! This implements the RISC philosophy: the driver is dumb and minimal,
//! it just pushes pixels to the screen. All text rendering, styling, etc.
//! is done in software by the rasterizer.

use crate::platform::backends::{
    BackendEvent, CursorVisibility, Driver, DriverCommand, FocusState, MouseButton, PlatformState,
    RenderCommand,
};
use anyhow::{Context, Result};
use log::{debug, info, trace, warn};
use std::os::unix::io::RawFd;
use std::sync::{Arc, Once};

// Default window dimensions
const DEFAULT_WINDOW_WIDTH_PX: usize = 800;
const DEFAULT_WINDOW_HEIGHT_PX: usize = 600;
const DEFAULT_WINDOW_X: f64 = 100.0;
const DEFAULT_WINDOW_Y: f64 = 100.0;

// Default cell (character) dimensions
const DEFAULT_CELL_WIDTH_PX: usize = 8;
const DEFAULT_CELL_HEIGHT_PX: usize = 16;

// RGBA pixel format constants
const BYTES_PER_PIXEL: usize = 4;
const BITS_PER_COMPONENT: usize = 8;
const BITS_PER_PIXEL: usize = 32;

// CGImageAlphaInfo values for bitmap format specification
// These determine how Core Graphics interprets RGBA pixel data
const CG_IMAGE_ALPHA_NONE: u32 = 0;
const CG_IMAGE_ALPHA_PREMULTIPLIED_LAST: u32 = 1;  // RGBA, premultiplied
const CG_IMAGE_ALPHA_PREMULTIPLIED_FIRST: u32 = 2; // ARGB, premultiplied
const CG_IMAGE_ALPHA_LAST: u32 = 3;                // RGBA, straight alpha
const CG_IMAGE_ALPHA_NONE_SKIP_LAST: u32 = 6;      // RGB_, ignore alpha

// NSView autoresizing mask: width sizable (2) | height sizable (16)
const NS_VIEW_AUTORESIZE_MASK: u64 = 18;

// Custom NSView subclass name
const VIEW_CLASS_NAME: &str = "CoreTermView";

#[cfg(target_os = "macos")]
use cocoa::appkit::{
    NSApp, NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreType,
    NSWindow, NSWindowStyleMask,
};
#[cfg(target_os = "macos")]
use cocoa::base::{id, nil, YES, NO};
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
use objc::declare::ClassDecl;
#[cfg(target_os = "macos")]
use objc::runtime::{Class, Object, Sel, BOOL};
#[cfg(target_os = "macos")]
use objc::{class, msg_send, sel, sel_impl};

pub struct CocoaDriver {
    #[cfg(target_os = "macos")]
    _pool: id, // NSAutoreleasePool
    #[cfg(target_os = "macos")]
    window: id,
    #[cfg(target_os = "macos")]
    _view: id,
    window_width_pts: f64,  // Logical size in points
    window_height_pts: f64, // Logical size in points
    backing_scale: f64,     // Retina scale factor (1.0 or 2.0)
    cell_width_px: usize,
    cell_height_px: usize,
    framebuffer: Vec<u8>, // RGBA pixels at physical resolution
    cleanup_once: Once,     // Ensures cleanup runs exactly once
}

#[cfg(target_os = "macos")]
impl CocoaDriver {
    /// Registers a custom NSView subclass that handles coordinate flipping and keyboard focus.
    /// This runs only once using Once for thread safety.
    fn register_view_class() {
        static REGISTER_ONCE: Once = Once::new();
        REGISTER_ONCE.call_once(|| {
            unsafe {
                let superclass = Class::get("NSView").unwrap();
                let mut decl = ClassDecl::new(VIEW_CLASS_NAME, superclass).unwrap();

                // Override isFlipped to return YES
                // This makes the coordinate system top-left (0,0), matching our terminal grid
                // and eliminating the need for manual Y-axis flipping
                extern "C" fn is_flipped(_this: &Object, _sel: Sel) -> BOOL {
                    use log::trace;
                    trace!("CoreTermView::isFlipped called - returning YES");
                    YES
                }
                decl.add_method(
                    sel!(isFlipped),
                    is_flipped as extern "C" fn(&Object, Sel) -> BOOL,
                );

                // Override acceptsFirstResponder to return YES
                // This allows the view to receive keyboard events
                extern "C" fn accepts_first_responder(_this: &Object, _sel: Sel) -> BOOL {
                    use log::debug;
                    debug!("CoreTermView::acceptsFirstResponder called - returning YES");
                    YES
                }
                decl.add_method(
                    sel!(acceptsFirstResponder),
                    accepts_first_responder as extern "C" fn(&Object, Sel) -> BOOL,
                );

                // Override becomeFirstResponder to log when we gain focus
                extern "C" fn become_first_responder(_this: &mut Object, _sel: Sel) -> BOOL {
                    use log::info;
                    info!("CoreTermView::becomeFirstResponder called - view gained focus");
                    YES
                }
                decl.add_method(
                    sel!(becomeFirstResponder),
                    become_first_responder as extern "C" fn(&mut Object, Sel) -> BOOL,
                );

                // Override keyDown: to handle keyboard events WITHOUT calling [super keyDown:]
                // This prevents the system beep when we handle keys ourselves
                extern "C" fn key_down(_this: &Object, _sel: Sel, _event: id) {
                    use log::trace;
                    trace!("CoreTermView::keyDown: called - event handled, NOT calling super");
                    // Explicitly do NOT call [super keyDown:event] to silence the beep
                    // The event will be processed via process_events() polling instead
                }
                decl.add_method(
                    sel!(keyDown:),
                    key_down as extern "C" fn(&Object, Sel, id),
                );

                decl.register();
                debug!("Registered custom NSView subclass: {}", VIEW_CLASS_NAME);
            }
        });
    }

    /// Registers a custom NSWindowDelegate that handles window close events.
    /// This runs only once using Once for thread safety.
    fn register_window_delegate_class() {
        static REGISTER_ONCE: Once = Once::new();
        REGISTER_ONCE.call_once(|| {
            unsafe {
                let superclass = Class::get("NSObject").unwrap();
                let mut decl = ClassDecl::new("CoreTermWindowDelegate", superclass).unwrap();

                // windowShouldClose: - called when user clicks the red X close button
                // Return YES to allow the window to close, NO to prevent it
                extern "C" fn window_should_close(_this: &Object, _sel: Sel, _sender: id) -> BOOL {
                    use log::info;
                    info!("CoreTermWindowDelegate::windowShouldClose: - allowing window to close");
                    YES
                }
                decl.add_method(
                    sel!(windowShouldClose:),
                    window_should_close as extern "C" fn(&Object, Sel, id) -> BOOL,
                );

                // windowWillClose: - called just before window closes
                // This is where we terminate the app when the window closes
                extern "C" fn window_will_close(_this: &Object, _sel: Sel, _notification: id) {
                    use log::info;
                    info!("CoreTermWindowDelegate::windowWillClose: - terminating application");
                    unsafe {
                        let app = NSApp();
                        let _: () = msg_send![app, terminate: nil];
                    }
                }
                decl.add_method(
                    sel!(windowWillClose:),
                    window_will_close as extern "C" fn(&Object, Sel, id),
                );

                decl.register();
                debug!("Registered NSWindowDelegate subclass: CoreTermWindowDelegate");
            }
        });
    }

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
                NSPoint::new(DEFAULT_WINDOW_X, DEFAULT_WINDOW_Y),
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

            // Register and set the window delegate to handle close button
            Self::register_window_delegate_class();
            let delegate_class = Class::get("CoreTermWindowDelegate").unwrap();
            let delegate: id = msg_send![delegate_class, new];
            let () = msg_send![window, setDelegate: delegate];

            window.center();
            window.makeKeyAndOrderFront_(nil);
            let () = msg_send![window, orderFrontRegardless];

            Ok(window)
        }
    }

    /// Create an NSView for rendering with CALayer backing
    /// Uses our custom CoreTermView subclass for proper coordinate system and keyboard handling
    fn create_view(width: f64, height: f64) -> Result<id> {
        unsafe {
            // Register the custom view class (runs once)
            Self::register_view_class();

            let class = Class::get(VIEW_CLASS_NAME).unwrap();
            let view_rect = NSRect::new(
                NSPoint::new(0.0, 0.0),
                NSSize::new(width as CGFloat, height as CGFloat),
            );

            // Allocate our CUSTOM view subclass
            let view: id = msg_send![class, alloc];
            let view: id = msg_send![view, initWithFrame: view_rect];

            if view == nil {
                return Err(anyhow::anyhow!("Failed to create custom view"));
            }

            let () = msg_send![view, setAutoresizingMask: NS_VIEW_AUTORESIZE_MASK];

            // CRITICAL: Enable layer-backing for modern rendering
            // This makes the view use a CALayer, which is the proper way to render on macOS
            let () = msg_send![view, setWantsLayer: YES];

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
            info!("CocoaDriver: Initializing with CALayer, HiDPI support, and custom view");

            unsafe {
                let pool = NSAutoreleasePool::new(nil);

                // Logical dimensions in points (not pixels)
                let window_width_pts = DEFAULT_WINDOW_WIDTH_PX as f64;
                let window_height_pts = DEFAULT_WINDOW_HEIGHT_PX as f64;

                Self::init_app().context("Failed to initialize NSApplication")?;
                let window = Self::create_window(window_width_pts as usize, window_height_pts as usize)
                    .context("Failed to create window")?;
                let view = Self::create_view(window_width_pts, window_height_pts)
                    .context("Failed to create view")?;

                let () = msg_send![window, setContentView: view];

                // Query the backing scale factor for HiDPI/Retina support
                let backing_scale: CGFloat = msg_send![window, backingScaleFactor];
                info!("CocoaDriver: Backing scale factor = {}", backing_scale);

                // Calculate PHYSICAL pixel size for framebuffer
                let buffer_width_px = (window_width_pts * backing_scale) as usize;
                let buffer_height_px = (window_height_pts * backing_scale) as usize;

                // Scale font cell size to physical resolution for Retina displays
                let cell_width_px = (DEFAULT_CELL_WIDTH_PX as f64 * backing_scale) as usize;
                let cell_height_px = (DEFAULT_CELL_HEIGHT_PX as f64 * backing_scale) as usize;

                info!(
                    "CocoaDriver: Window {} x {} points ({} x {} physical px), font cells {} x {} px",
                    window_width_pts, window_height_pts, buffer_width_px, buffer_height_px,
                    cell_width_px, cell_height_px
                );

                // Create framebuffer at physical resolution
                let framebuffer = vec![0u8; buffer_width_px * buffer_height_px * BYTES_PER_PIXEL];

                let mut driver = Self {
                    _pool: pool,
                    window,
                    _view: view,
                    window_width_pts,
                    window_height_pts,
                    backing_scale,
                    cell_width_px,
                    cell_height_px,
                    framebuffer,
                    cleanup_once: Once::new(),
                };

                driver.present().context("Failed initial present")?;

                // Force the application and window to the front and capture keyboard focus
                // Make the view the first responder to receive keyboard events
                let app = NSApp();
                let () = msg_send![app, activateIgnoringOtherApps: YES];
                let () = msg_send![window, makeKeyAndOrderFront: nil];

                info!("Attempting to make view the first responder...");
                let success: BOOL = msg_send![window, makeFirstResponder: view];
                info!("makeFirstResponder returned: {}", if success == YES { "YES" } else { "NO" });

                // Verify which view is actually the first responder
                let first_responder: id = msg_send![window, firstResponder];
                if first_responder == view {
                    info!("SUCCESS: View is now the first responder");
                } else {
                    warn!("WARNING: First responder is NOT our view (it's {:?})", first_responder);
                }

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

            // CRITICAL: We must call sendEvent for ALL events we process, not just unhandled ones.
            // macOS window management (close button, resize, move, etc.) depends on events being
            // dispatched through NSApp even if we also handle them for terminal input.
            // Without this, the window freezes and becomes unresponsive.

            // CRITICAL FIX: Use a very short timeout instead of distantPast
            // Cocoa's event system needs the app to actually WAIT for events, not just poll
            // A 1ms timeout allows events to be delivered while keeping the loop responsive
            let timeout: id = msg_send![class!(NSDate), dateWithTimeIntervalSinceNow: 0.001];

            loop {
                let event: id = msg_send![
                    app,
                    nextEventMatchingMask: NSEventMask::NSAnyEventMask.bits()
                    untilDate: timeout
                    inMode: cocoa::foundation::NSDefaultRunLoopMode
                    dequeue: YES
                ];

                if event == nil {
                    break;
                }

                let event_type: u64 = msg_send![event, type];

                // Log ALL events to see what we're actually receiving
                debug!("cocoa: Received NSEvent type={}", event_type);

                match event_type {
                    t if t == NSEventType::NSKeyDown as u64 => {
                        debug!("cocoa: Received NSKeyDown event");
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

                        debug!("cocoa: KeyDown - keyCode={}, text='{}', symbol={:?}", key_code, text, symbol);

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
                            y: (self.window_height_pts - location.y) as u16,
                            modifiers,
                        });
                    }
                    t if t == NSEventType::NSRightMouseDown as u64 => {
                        let location: NSPoint = msg_send![event, locationInWindow];
                        let modifiers = Self::extract_modifiers(event);
                        backend_events.push(BackendEvent::MouseButtonPress {
                            button: MouseButton::Right,
                            x: location.x as u16,
                            y: (self.window_height_pts - location.y) as u16,
                            modifiers,
                        });
                    }
                    t if t == NSEventType::NSLeftMouseUp as u64 => {
                        let location: NSPoint = msg_send![event, locationInWindow];
                        let modifiers = Self::extract_modifiers(event);
                        backend_events.push(BackendEvent::MouseButtonRelease {
                            button: MouseButton::Left,
                            x: location.x as u16,
                            y: (self.window_height_pts - location.y) as u16,
                            modifiers,
                        });
                    }
                    t if t == NSEventType::NSRightMouseUp as u64 => {
                        let location: NSPoint = msg_send![event, locationInWindow];
                        let modifiers = Self::extract_modifiers(event);
                        backend_events.push(BackendEvent::MouseButtonRelease {
                            button: MouseButton::Right,
                            x: location.x as u16,
                            y: (self.window_height_pts - location.y) as u16,
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
                            y: (self.window_height_pts - location.y) as u16,
                            modifiers,
                        });
                    }
                    _ => {}
                }

                // CRITICAL: Forward ALL events to NSApp for window management.
                // Without this, macOS never processes window close/resize/move operations,
                // causing the window to freeze (symptom: can't click red X button).
                // See: https://developer.apple.com/documentation/appkit/nsapplication/1428359-sendevent
                trace!("cocoa: Dispatching event type={} to NSApp via sendEvent", event_type);
                let () = msg_send![app, sendEvent: event];
            }

            Ok(backend_events)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(vec![])
        }
    }

    fn get_platform_state(&self) -> PlatformState {
        // Return PHYSICAL pixel dimensions for the rasterizer
        let physical_width = (self.window_width_pts * self.backing_scale) as u16;
        let physical_height = (self.window_height_pts * self.backing_scale) as u16;

        PlatformState {
            event_fd: None,
            font_cell_width_px: self.cell_width_px,
            font_cell_height_px: self.cell_height_px,
            scale_factor: self.backing_scale,
            display_width_px: physical_width,
            display_height_px: physical_height,
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
        // Sample framebuffer to check if we have actual color data
        let has_non_black = self.framebuffer.chunks_exact(4).take(100).any(|p| {
            p[0] > 0 || p[1] > 0 || p[2] > 0 || p[3] > 0
        });

        let physical_width = (self.window_width_pts * self.backing_scale) as usize;
        let physical_height = (self.window_height_pts * self.backing_scale) as usize;

        debug!(
            "CocoaDriver::present() - framebuffer size={} bytes ({}x{} physical px, scale={}), has_non_black_pixels={}",
            self.framebuffer.len(),
            physical_width,
            physical_height,
            self.backing_scale,
            has_non_black
        );

        #[cfg(target_os = "macos")]
        unsafe {
            self.draw_framebuffer_to_view()
                .context("Failed to draw framebuffer to view")?;
        }

        debug!("CocoaDriver::present() completed successfully");
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
        info!("CocoaDriver: Cleanup called");
        self.cleanup_once.call_once(|| {
            self.do_cleanup();
        });
        Ok(())
    }

    fn get_framebuffer_mut(&mut self) -> &mut [u8] {
        &mut self.framebuffer
    }

    fn get_framebuffer_size(&self) -> (usize, usize) {
        // Return PHYSICAL pixel dimensions
        (
            (self.window_width_pts * self.backing_scale) as usize,
            (self.window_height_pts * self.backing_scale) as usize,
        )
    }
}

// New minimal driver interface
#[cfg(target_os = "macos")]
impl CocoaDriver {
    /// Converts the framebuffer to a CGImage and sets it as the CALayer contents
    unsafe fn draw_framebuffer_to_view(&mut self) -> Result<()> {
        // Calculate physical pixel dimensions
        let width = (self.window_width_pts * self.backing_scale) as usize;
        let height = (self.window_height_pts * self.backing_scale) as usize;

        debug!(
            "draw_framebuffer_to_view: {} x {} physical pixels (scale={})",
            width, height, self.backing_scale
        );

        let bytes_per_row = width * BYTES_PER_PIXEL;

        // CGDataProvider needs Arc<Vec<u8>> for shared ownership
        let data = Arc::new(self.framebuffer.clone());

        let provider = CGDataProvider::from_buffer(data);
        let color_space = CGColorSpace::create_device_rgb();

        // Use kCGImageAlphaPremultipliedLast (1) for best CALayer compatibility
        let bitmap_info = 1u32;

        let image = CGImage::new(
            width,
            height,
            BITS_PER_COMPONENT,
            BITS_PER_PIXEL,
            bytes_per_row,
            &color_space,
            bitmap_info,
            &provider,
            false, // should_interpolate
            0,     // rendering_intent (0 = default)
        );

        // Get the view's CALayer
        let layer: id = msg_send![self._view, layer];
        if layer == nil {
            return Err(anyhow::anyhow!("View has no layer - did setWantsLayer: YES fail?"));
        }

        // Extract the raw CGImageRef pointer from the wrapper
        // SAFETY: CGImage is repr(transparent) over *mut sys::CGImage
        let image_ref: *mut core_graphics::sys::CGImage = std::mem::transmute_copy(&image);

        // Set the layer's contents
        let _: () = msg_send![layer, setContents: image_ref];

        // CRITICAL: Set the contentsScale to match the backing scale
        // Without this, the layer assumes 1x content and stretches it (fuzziness)
        let _: () = msg_send![layer, setContentsScale: self.backing_scale];

        debug!("draw_framebuffer_to_view: CGImage set (scale={})", self.backing_scale);

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

    /// Internal cleanup logic - closes the window.
    /// This is called exactly once via `Once` from `cleanup()`.
    fn do_cleanup(&self) {
        info!("CocoaDriver: Performing cleanup (closing window)");
        #[cfg(target_os = "macos")]
        unsafe {
            // Close the window - this is all we need to do
            // The app will terminate naturally when main() exits
            let _: () = msg_send![self.window, close];
        }
    }

    /// Public wrapper for backward compatibility. Calls cleanup().
    pub fn close_window(&mut self) {
        let _ = self.cleanup();
    }
}

impl Drop for CocoaDriver {
    fn drop(&mut self) {
        debug!("CocoaDriver: Dropping");
    }
}
