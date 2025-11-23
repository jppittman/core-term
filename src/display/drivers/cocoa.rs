#![cfg(target_os = "macos")]

//! Cocoa DisplayDriver implementation using objc2.
//!
//! This is a minimal RISC-style driver that provides only platform-specific primitives.
//! All common logic lives in DisplayManager.

use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, DriverRequest, DriverResponse};
use anyhow::{Context, Result};
use core_graphics::base::CGFloat;
use core_graphics::color_space::CGColorSpace;
use core_graphics::data_provider::CGDataProvider;
use core_graphics::image::CGImage;
use log::{debug, info, trace, warn};
use objc2::rc::{autoreleasepool, Id, Retained};
use objc2::runtime::{AnyObject, Bool, ProtocolObject, Sel};
use objc2::{class, msg_send, msg_send_id, sel, ClassType, DeclaredClass, MainThreadOnly};

// Explicit imports from objc2-app-kit
use objc2_app_kit::{
    NSApp, NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSEvent, NSEventMask,
    NSEventModifierFlags, NSEventType, NSPasteboard, NSView, NSWindow, NSWindowStyleMask,
};

// Explicit imports from objc2-foundation
use objc2_foundation::{
    MainThreadMarker,
    NSDate,
    NSDefaultRunLoopMode,
    NSObject,
    NSObjectProtocol,
    NSPoint,
    NSRect,
    NSSize, // Use NS* types instead of CG* types
    NSString,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

// Default window dimensions
const DEFAULT_WINDOW_X: f64 = 100.0;
const DEFAULT_WINDOW_Y: f64 = 100.0;

// Default cell (character) dimensions
const DEFAULT_CELL_WIDTH_PX: usize = 8;
const DEFAULT_CELL_HEIGHT_PX: usize = 16;

// RGBA pixel format constants
const BYTES_PER_PIXEL: usize = 4;
const BITS_PER_COMPONENT: usize = 8;
const BITS_PER_PIXEL: usize = 32;

const MAX_DRAW_LATENCY_SECONDS: f64 = 0.016; // ~60 FPS (16ms)

// Custom NSView subclass name
const VIEW_CLASS_NAME: &str = "CoreTermView";
const DELEGATE_CLASS_NAME: &str = "CoreTermWindowDelegate";

/// Platform-specific Cocoa display driver.
pub struct CocoaDisplayDriver {
    mtm: MainThreadMarker,
    window: Option<Retained<NSWindow>>,
    view: Option<Retained<NSObject>>,
    delegate: Option<Retained<NSObject>>,
    window_width_pts: f64,
    window_height_pts: f64,
    backing_scale: f64,
    framebuffer: Option<Box<[u8]>>,
}

impl DisplayDriver for CocoaDisplayDriver {
    fn new() -> Result<Self> {
        let mtm =
            MainThreadMarker::new().context("CocoaDisplayDriver must be created on main thread")?;

        info!("CocoaDisplayDriver::new() - Pure initialization");

        // Register custom classes
        Self::register_view_class();
        Self::register_delegate_class();

        // Initialize NSApplication
        Self::init_app(mtm)?;

        Ok(Self {
            mtm,
            window: None,
            view: None,
            delegate: None,
            window_width_pts: 0.0,
            window_height_pts: 0.0,
            backing_scale: 1.0,
            framebuffer: None,
        })
    }

    fn handle_request(&mut self, request: DriverRequest) -> Result<DriverResponse> {
        match request {
            DriverRequest::Init => self.handle_init(),
            DriverRequest::PollEvents => self.handle_poll_events(),
            DriverRequest::RequestFramebuffer => self.handle_request_framebuffer(),
            DriverRequest::Present(buffer) => self.handle_present(buffer),
            DriverRequest::SetTitle(title) => self.handle_set_title(&title),
            DriverRequest::Bell => self.handle_bell(),
            DriverRequest::SetCursorVisibility(visible) => {
                self.handle_set_cursor_visibility(visible)
            }
            DriverRequest::CopyToClipboard(text) => self.handle_copy_to_clipboard(&text),
            DriverRequest::RequestPaste => self.handle_request_paste(),
        }
    }
}

impl CocoaDisplayDriver {
    /// Register custom NSView subclass
    fn register_view_class() {
        use objc2::declare::ClassBuilder;
        use std::sync::Once;

        static REGISTER_ONCE: Once = Once::new();
        REGISTER_ONCE.call_once(|| {
            let mut builder = ClassBuilder::new(VIEW_CLASS_NAME, class!(NSView))
                .expect("Failed to create CoreTermView class");

            // Override isFlipped to return YES (top-left origin)
            unsafe extern "C" fn is_flipped(_this: &NSObject, _cmd: Sel) -> Bool {
                trace!("CoreTermView::isFlipped called - returning YES");
                Bool::YES
            }
            unsafe {
                builder.add_method(
                    sel!(isFlipped),
                    is_flipped as unsafe extern "C" fn(&NSObject, Sel) -> Bool,
                );
            }

            // Override acceptsFirstResponder to return YES
            unsafe extern "C" fn accepts_first_responder(_this: &NSObject, _cmd: Sel) -> Bool {
                debug!("CoreTermView::acceptsFirstResponder called - returning YES");
                Bool::YES
            }
            unsafe {
                builder.add_method(
                    sel!(acceptsFirstResponder),
                    accepts_first_responder as unsafe extern "C" fn(&NSObject, Sel) -> Bool,
                );
            }

            // Override becomeFirstResponder
            unsafe extern "C" fn become_first_responder(_this: &mut NSObject, _cmd: Sel) -> Bool {
                info!("CoreTermView::becomeFirstResponder called - view gained focus");
                Bool::YES
            }
            unsafe {
                builder.add_method(
                    sel!(becomeFirstResponder),
                    become_first_responder as unsafe extern "C" fn(&mut NSObject, Sel) -> Bool,
                );
            }

            // Override keyDown: to prevent system beep
            unsafe extern "C" fn key_down(_this: &NSObject, _cmd: Sel, _event: *mut NSEvent) {
                trace!("CoreTermView::keyDown: called - event handled, NOT calling super");
                // Don't call [super keyDown:event] to silence beep
            }
            unsafe {
                builder.add_method(
                    sel!(keyDown:),
                    key_down as unsafe extern "C" fn(&NSObject, Sel, *mut NSEvent),
                );
            }

            let _cls = builder.register();
            debug!("Registered custom NSView subclass: {}", VIEW_CLASS_NAME);
        });
    }

    /// Register custom NSWindowDelegate subclass
    fn register_delegate_class() {
        use objc2::declare::ClassBuilder;
        use std::sync::Once;

        static REGISTER_ONCE: Once = Once::new();
        REGISTER_ONCE.call_once(|| {
            let mut builder = ClassBuilder::new(DELEGATE_CLASS_NAME, class!(NSObject))
                .expect("Failed to create CoreTermWindowDelegate class");

            // windowShouldClose:
            unsafe extern "C" fn window_should_close(
                _this: &NSObject,
                _cmd: Sel,
                _sender: *mut NSWindow,
            ) -> Bool {
                info!("CoreTermWindowDelegate::windowShouldClose: - allowing window to close");
                Bool::YES
            }
            unsafe {
                builder.add_method(
                    sel!(windowShouldClose:),
                    window_should_close
                        as unsafe extern "C" fn(&NSObject, Sel, *mut NSWindow) -> Bool,
                );
            }

            // windowWillClose:
            unsafe extern "C" fn window_will_close(
                _this: &NSObject,
                _cmd: Sel,
                _notification: *mut NSObject,
            ) {
                info!("CoreTermWindowDelegate::windowWillClose: - window closing");
            }
            unsafe {
                builder.add_method(
                    sel!(windowWillClose:),
                    window_will_close as unsafe extern "C" fn(&NSObject, Sel, *mut NSObject),
                );
            }

            let _cls = builder.register();
            debug!(
                "Registered NSWindowDelegate subclass: {}",
                DELEGATE_CLASS_NAME
            );
        });
    }

    /// Initialize NSApplication
    fn init_app(mtm: MainThreadMarker) -> Result<()> {
        unsafe {
            let app = NSApplication::sharedApplication(mtm);
            app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
            let _: () = msg_send![&app, finishLaunching];
            let _: () = msg_send![&app, activateIgnoringOtherApps: Bool::YES];
            info!("NSApplication initialized");
            Ok(())
        }
    }

    /// Handle Init request - create window and discover metrics
    fn handle_init(&mut self) -> Result<DriverResponse> {
        info!("CocoaDisplayDriver: Handling Init request");

        // Calculate initial window size (80x24 terminal)
        let cols = 80usize;
        let rows = 24usize;
        let window_width_pts = (cols * DEFAULT_CELL_WIDTH_PX) as f64;
        let window_height_pts = (rows * DEFAULT_CELL_HEIGHT_PX) as f64;

        // Create window
        let window = self.create_window(window_width_pts as usize, window_height_pts as usize)?;

        // Create view
        let view = self.create_view(window_width_pts, window_height_pts)?;

        // Set view as window content
        unsafe {
            let _: () = msg_send![&window, setContentView: &view];
        }

        // Get backing scale factor
        let backing_scale: CGFloat = unsafe { msg_send![&window, backingScaleFactor] };
        info!(
            "CocoaDisplayDriver: Backing scale factor = {}",
            backing_scale
        );

        // Calculate physical dimensions
        let cell_width_px = (DEFAULT_CELL_WIDTH_PX as f64 * backing_scale) as u32;
        let cell_height_px = (DEFAULT_CELL_HEIGHT_PX as f64 * backing_scale) as u32;
        let width_px = (window_width_pts * backing_scale) as u32;
        let height_px = (window_height_pts * backing_scale) as u32;

        info!(
            "CocoaDisplayDriver: Window {} x {} points ({} x {} physical px), font cells {} x {} px",
            window_width_pts, window_height_pts, width_px, height_px, cell_width_px, cell_height_px
        );

        // Allocate initial framebuffer
        let buffer_size = (width_px as usize) * (height_px as usize) * BYTES_PER_PIXEL;
        let framebuffer = vec![0u8; buffer_size].into_boxed_slice();

        // Store state
        self.window = Some(window.clone());
        self.view = Some(view.clone());
        self.window_width_pts = window_width_pts;
        self.window_height_pts = window_height_pts;
        self.backing_scale = backing_scale;
        self.framebuffer = Some(framebuffer);

        // Make window visible and focused
        unsafe {
            let app = NSApp(self.mtm);
            let _: () = msg_send![&app, activateIgnoringOtherApps: Bool::YES];
            let _: () = msg_send![&window, makeKeyAndOrderFront: None::<&NSObject>];

            // Make view first responder
            info!("Attempting to make view the first responder...");
            let success: Bool = msg_send![&window, makeFirstResponder: &view];
            info!(
                "makeFirstResponder returned: {}",
                if success.as_bool() { "YES" } else { "NO" }
            );

            // Verify first responder was set correctly
            let first_responder: *mut AnyObject = msg_send![&window, firstResponder];
            let view_ptr: *const NSObject = &**view;
            if first_responder as *const NSObject == view_ptr {
                info!("SUCCESS: View is now the first responder");
            } else {
                warn!(
                    "WARNING: First responder is NOT our view (it's {:?})",
                    first_responder
                );
            }
        }

        Ok(DriverResponse::InitComplete {
            width_px,
            height_px,
            scale_factor: backing_scale,
        })
    }

    /// Create an NSWindow
    fn create_window(&self, width: usize, height: usize) -> Result<Retained<NSWindow>> {
        unsafe {
            let content_rect = NSRect::new(
                NSPoint::new(DEFAULT_WINDOW_X, DEFAULT_WINDOW_Y),
                NSSize::new(width as f64, height as f64),
            );

            let style_mask = NSWindowStyleMask::Titled
                | NSWindowStyleMask::Closable
                | NSWindowStyleMask::Miniaturizable
                | NSWindowStyleMask::Resizable;

            let window = NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(self.mtm),
                content_rect,
                style_mask,
                NSBackingStoreType::Buffered,
                false,
            );

            // Set title
            let title = NSString::from_str("CoreTerm");
            window.setTitle(&title);

            // Center window on screen
            window.center();

            // Configure window for optimal rendering
            let _: () = msg_send![&window, setOpaque: Bool::YES];

            // Set black background color
            let black_color: *mut AnyObject = msg_send![class!(NSColor), blackColor];
            let _: () = msg_send![&window, setBackgroundColor: black_color];

            // Create and set delegate
            let delegate_class = class!(CoreTermWindowDelegate);
            let delegate: Retained<NSObject> = msg_send![delegate_class, new];
            window.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));

            // Store delegate to keep it alive
            self.delegate = Some(delegate);

            info!("NSWindow created successfully");
            Ok(window)
        }
    }

    /// Create a custom NSView
    fn create_view(&self, width: f64, height: f64) -> Result<Retained<NSObject>> {
        unsafe {
            let view_class = class!(CoreTermView);
            let view: Retained<NSObject> = msg_send![view_class, alloc];

            let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(width, height));
            let view: Retained<NSObject> = msg_send![view, initWithFrame: frame];

            // Enable CALayer backing
            let _: () = msg_send![&view, setWantsLayer: Bool::YES];

            // Set autoresizing mask
            let autoresizing_mask: u64 = 18; // width sizable (2) | height sizable (16)
            let _: () = msg_send![&view, setAutoresizingMask: autoresizing_mask];

            info!("CoreTermView created successfully");
            Ok(view)
        }
    }

    /// Handle PollEvents request
    fn handle_poll_events(&mut self) -> Result<DriverResponse> {
        let mut events = Vec::new();

        unsafe {
            let app = NSApplication::sharedApplication(self.mtm);

            // Poll with timeout
            let timeout = NSDate::dateWithTimeIntervalSinceNow(MAX_DRAW_LATENCY_SECONDS);

            loop {
                let event = app.nextEventMatchingMask_untilDate_inMode_dequeue(
                    NSEventMask::Any,
                    Some(&timeout),
                    &NSDefaultRunLoopMode,
                    true,
                );

                let Some(event) = event else {
                    break;
                };

                let event_type = event.r#type();

                // Skip NSApplicationDefined events
                if event_type == NSEventType::ApplicationDefined {
                    debug!("cocoa: Skipping NSApplicationDefined event");
                    let _: () = msg_send![&app, sendEvent: &event];
                    continue;
                }

                debug!("cocoa: Received NSEvent type={:?}", event_type);

                // Convert to DisplayEvent
                if let Some(display_event) = self.convert_event(&event) {
                    events.push(display_event);
                }

                // Forward to NSApp for window management
                trace!("cocoa: Dispatching event to NSApp via sendEvent");
                let _: () = msg_send![&app, sendEvent: &event];
            }
        }

        Ok(DriverResponse::Events(events))
    }

    /// Convert NSEvent to DisplayEvent
    fn convert_event(&self, event: &NSEvent) -> Option<DisplayEvent> {
        use crate::platform::backends::{KeySymbol, Modifiers};

        unsafe {
            match event.r#type() {
                NSEventType::KeyDown => {
                    let chars = event.characters();
                    let text = chars.map(|s| s.to_string());

                    let key_code = event.keyCode();
                    let symbol = Self::map_keycode_to_symbol(key_code);
                    let modifiers = Self::extract_modifiers(event);

                    debug!(
                        "cocoa: KeyDown - keyCode={}, text={:?}, symbol={:?}",
                        key_code, text, symbol
                    );

                    Some(DisplayEvent::Key {
                        symbol,
                        modifiers,
                        text,
                    })
                }
                NSEventType::LeftMouseDown => {
                    let location = event.locationInWindow();
                    let modifiers = Self::extract_modifiers(event);
                    Some(DisplayEvent::MouseButtonPress {
                        button: 0,
                        x: location.x as i32,
                        y: (self.window_height_pts - location.y) as i32,
                        modifiers,
                    })
                }
                NSEventType::RightMouseDown => {
                    let location = event.locationInWindow();
                    let modifiers = Self::extract_modifiers(event);
                    Some(DisplayEvent::MouseButtonPress {
                        button: 1,
                        x: location.x as i32,
                        y: (self.window_height_pts - location.y) as i32,
                        modifiers,
                    })
                }
                NSEventType::LeftMouseUp => {
                    let location = event.locationInWindow();
                    let modifiers = Self::extract_modifiers(event);
                    Some(DisplayEvent::MouseButtonRelease {
                        button: 0,
                        x: location.x as i32,
                        y: (self.window_height_pts - location.y) as i32,
                        modifiers,
                    })
                }
                NSEventType::RightMouseUp => {
                    let location = event.locationInWindow();
                    let modifiers = Self::extract_modifiers(event);
                    Some(DisplayEvent::MouseButtonRelease {
                        button: 1,
                        x: location.x as i32,
                        y: (self.window_height_pts - location.y) as i32,
                        modifiers,
                    })
                }
                NSEventType::MouseMoved
                | NSEventType::LeftMouseDragged
                | NSEventType::RightMouseDragged => {
                    let location = event.locationInWindow();
                    let modifiers = Self::extract_modifiers(event);
                    Some(DisplayEvent::MouseMove {
                        x: location.x as i32,
                        y: (self.window_height_pts - location.y) as i32,
                        modifiers,
                    })
                }
                _ => None,
            }
        }
    }

    /// Extract modifier flags
    fn extract_modifiers(event: &NSEvent) -> crate::platform::backends::Modifiers {
        use crate::platform::backends::Modifiers;

        let flags = event.modifierFlags();
        let mut modifiers = Modifiers::empty();

        if flags.contains(NSEventModifierFlags::Shift) {
            modifiers |= Modifiers::SHIFT;
        }
        if flags.contains(NSEventModifierFlags::Control) {
            modifiers |= Modifiers::CONTROL;
        }
        if flags.contains(NSEventModifierFlags::Option) {
            modifiers |= Modifiers::ALT;
        }
        if flags.contains(NSEventModifierFlags::Command) {
            modifiers |= Modifiers::SUPER;
        }

        modifiers
    }

    /// Map macOS keycode to KeySymbol
    fn map_keycode_to_symbol(keycode: u16) -> crate::platform::backends::KeySymbol {
        use crate::platform::backends::KeySymbol;

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
            _ => KeySymbol::Char('\0'),
        }
    }

    /// Handle RequestFramebuffer
    fn handle_request_framebuffer(&mut self) -> Result<DriverResponse> {
        let buffer = self
            .framebuffer
            .take()
            .context("Framebuffer already transferred")?;
        Ok(DriverResponse::Framebuffer(buffer))
    }

    /// Handle Present
    fn handle_present(&mut self, buffer: Box<[u8]>) -> Result<DriverResponse> {
        let view = self.view.as_ref().context("View not initialized")?;

        unsafe {
            // Calculate dimensions
            let width = (self.window_width_pts * self.backing_scale) as usize;
            let height = (self.window_height_pts * self.backing_scale) as usize;

            debug!(
                "draw_framebuffer_to_view: {} x {} physical pixels (scale={})",
                width, height, self.backing_scale
            );

            let bytes_per_row = width * BYTES_PER_PIXEL;

            // Create CGImage from buffer
            let data = Arc::new(Vec::from(buffer.as_ref()));
            let provider = CGDataProvider::from_buffer(data);
            let color_space = CGColorSpace::create_device_rgb();
            let bitmap_info = 1u32; // kCGImageAlphaPremultipliedLast

            let image = CGImage::new(
                width,
                height,
                BITS_PER_COMPONENT,
                BITS_PER_PIXEL,
                bytes_per_row,
                &color_space,
                bitmap_info,
                &provider,
                false,
                0,
            );

            // Get CALayer
            let layer: *mut AnyObject = msg_send![&**view, layer];
            if layer.is_null() {
                return Err(anyhow::anyhow!(
                    "View has no layer - did setWantsLayer: YES fail?"
                ));
            }

            // Extract CGImageRef pointer
            let image_ref: *mut core_graphics::sys::CGImage = std::mem::transmute_copy(&image);

            // Set layer contents
            let _: () = msg_send![layer, setContents: image_ref];
            let _: () = msg_send![layer, setContentsScale: self.backing_scale];

            debug!(
                "draw_framebuffer_to_view: CGImage set (scale={})",
                self.backing_scale
            );
        }

        // Return buffer ownership to caller for reuse
        Ok(DriverResponse::PresentComplete(buffer))
    }

    /// Handle SetTitle
    fn handle_set_title(&mut self, title: &str) -> Result<DriverResponse> {
        if let Some(window) = &self.window {
            let ns_title = NSString::from_str(title);
            window.setTitle(&ns_title);
        }
        Ok(DriverResponse::TitleSet)
    }

    /// Handle Bell
    fn handle_bell(&mut self) -> Result<DriverResponse> {
        unsafe {
            let app = NSApplication::sharedApplication(self.mtm);
            let _: () = msg_send![&app, beep];
        }
        Ok(DriverResponse::BellRung)
    }

    /// Handle SetCursorVisibility
    fn handle_set_cursor_visibility(&mut self, _visible: bool) -> Result<DriverResponse> {
        // TODO: Implement with NSCursor
        Ok(DriverResponse::CursorVisibilitySet)
    }

    /// Handle CopyToClipboard
    fn handle_copy_to_clipboard(&mut self, text: &str) -> Result<DriverResponse> {
        unsafe {
            let pasteboard = NSPasteboard::generalPasteboard();
            pasteboard.clearContents();
            let ns_string = NSString::from_str(text);
            let _: bool =
                msg_send![&pasteboard, setString: &ns_string forType: NSPasteboardTypeString];
        }
        Ok(DriverResponse::ClipboardCopied)
    }

    /// Handle RequestPaste
    fn handle_request_paste(&mut self) -> Result<DriverResponse> {
        // Paste data will arrive via PasteData event (not implemented yet)
        Ok(DriverResponse::PasteRequested)
    }
}

impl Drop for CocoaDisplayDriver {
    fn drop(&mut self) {
        info!("CocoaDisplayDriver::drop() - Closing window");
        if let Some(window) = &self.window {
            unsafe {
                let _: () = msg_send![&**window, close];
            }
        }
        info!("CocoaDisplayDriver::drop() - Drop complete");
    }
}

// NSPasteboardTypeString constant
unsafe fn NSPasteboardTypeString() -> &'static NSString {
    extern "C" {
        static NSPasteboardTypeString: &'static NSString;
    }
    NSPasteboardTypeString
}
