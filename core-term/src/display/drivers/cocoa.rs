#![cfg(target_os = "macos")]

//! Cocoa DisplayDriver implementation using objc2.

use crate::display::driver::DisplayDriver;
use crate::display::messages::{
    DisplayError, DisplayEvent, DriverConfig, DriverRequest, DriverResponse, RenderSnapshot,
};
use crate::platform::waker::{CocoaWaker, EventLoopWaker};
use anyhow::{Context, Result};
use core_graphics::base::CGFloat;
use core_graphics::color_space::CGColorSpace;
use core_graphics::data_provider::CGDataProvider;
use core_graphics::image::CGImage;
use log::{debug, info, trace};
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyObject, Bool, Sel};
use objc2::{class, msg_send, sel, MainThreadOnly};
//use core_foundation::base::TCFType;
use foreign_types_shared::ForeignType;

// Explicit imports from objc2-app-kit
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSEvent, NSEventMask,
    NSEventModifierFlags, NSEventType, NSPasteboard, NSWindow, NSWindowStyleMask,
};

// Explicit imports from objc2-foundation
use objc2_foundation::{
    MainThreadMarker, NSDate, NSDefaultRunLoopMode, NSObject, NSPoint, NSRect, NSSize, NSString,
};
use std::ffi::{c_void, CStr};
use std::sync::Arc;

// Default dimensions
const DEFAULT_WINDOW_X: f64 = 100.0;
const DEFAULT_WINDOW_Y: f64 = 100.0;
const BYTES_PER_PIXEL: usize = 4;
const BITS_PER_COMPONENT: usize = 8;
const BITS_PER_PIXEL: usize = 32;
// Aggressive timeout to test wake events - should still feel instant if wake mechanism works
const MAX_DRAW_LATENCY_SECONDS: f64 = 1.0; // 1 second - relies entirely on wake events

const VIEW_CLASS_NAME: &str = "CoreTermView";
const DELEGATE_CLASS_NAME: &str = "CoreTermWindowDelegate";

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
    fn new(config: &DriverConfig) -> Result<Self> {
        let mtm =
            MainThreadMarker::new().context("CocoaDisplayDriver must be created on main thread")?;
        info!("CocoaDisplayDriver::new() - Full initialization with config");

        Self::register_view_class();
        Self::register_delegate_class();
        Self::init_app(mtm)?;

        // Calculate window size from config
        let cols = config.initial_cols;
        let rows = config.initial_rows;
        let window_width_pts = (cols * config.cell_width_px) as f64;
        let window_height_pts = (rows * config.cell_height_px) as f64;

        let mut driver = Self {
            mtm,
            window: None,
            view: None,
            delegate: None,
            window_width_pts,
            window_height_pts,
            backing_scale: 1.0,
            framebuffer: None,
        };

        // Create window and view
        let window = driver.create_window(window_width_pts as usize, window_height_pts as usize)?;
        let view = driver.create_view(window_width_pts, window_height_pts)?;

        unsafe {
            let _: () = msg_send![&window, setContentView: &*view];
        }

        let backing_scale: CGFloat = unsafe { msg_send![&window, backingScaleFactor] };
        info!(
            "CocoaDisplayDriver: Created window {}x{} pts, backing scale {}",
            window_width_pts, window_height_pts, backing_scale
        );

        driver.window = Some(window);
        driver.view = Some(view);
        driver.backing_scale = backing_scale;
        driver.window_width_pts = window_width_pts;
        driver.window_height_pts = window_height_pts;

        Ok(driver)
    }

    fn create_waker(&self) -> Box<dyn EventLoopWaker> {
        Box::new(CocoaWaker::new())
    }

    fn handle_request(
        &mut self,
        request: DriverRequest,
    ) -> std::result::Result<DriverResponse, DisplayError> {
        match request {
            DriverRequest::Init => Ok(self.handle_init()?),
            DriverRequest::PollEvents => Ok(self.handle_poll_events()?),
            DriverRequest::RequestFramebuffer => Ok(self.handle_request_framebuffer()?),
            // Present returns DisplayError directly, so no wrapping needed
            DriverRequest::Present(snapshot) => self.handle_present(snapshot),
            DriverRequest::SetTitle(title) => Ok(self.handle_set_title(&title)?),
            DriverRequest::Bell => Ok(self.handle_bell()?),
            DriverRequest::SetCursorVisibility(visible) => {
                Ok(self.handle_set_cursor_visibility(visible)?)
            }
            DriverRequest::CopyToClipboard(text) => Ok(self.handle_copy_to_clipboard(&text)?),
            DriverRequest::RequestPaste => Ok(self.handle_request_paste()?),
            DriverRequest::SubmitClipboardData(text) => Ok(self.handle_copy_to_clipboard(&text)?),
        }
    }
}

impl CocoaDisplayDriver {
    fn register_view_class() {
        use objc2::declare::ClassBuilder;
        use std::sync::Once;
        static REGISTER_ONCE: Once = Once::new();
        REGISTER_ONCE.call_once(|| {
            let name = CStr::from_bytes_with_nul(b"CoreTermView\0").unwrap();
            let mut builder = ClassBuilder::new(name, class!(NSView))
                .expect("Failed to create CoreTermView class");

            // FIX: Restore explicit cast to satisfy HRTB for *mut AnyObject
            unsafe extern "C" fn is_flipped(_this: *mut AnyObject, _cmd: Sel) -> Bool {
                trace!("CoreTermView::isFlipped called - returning YES");
                Bool::YES
            }
            unsafe {
                builder.add_method(
                    sel!(isFlipped),
                    is_flipped as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool,
                );
            }

            unsafe extern "C" fn accepts_first_responder(_this: *mut AnyObject, _cmd: Sel) -> Bool {
                debug!("CoreTermView::acceptsFirstResponder called - returning YES");
                Bool::YES
            }
            // FIX: Restore explicit cast
            unsafe {
                builder.add_method(
                    sel!(acceptsFirstResponder),
                    accepts_first_responder as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool,
                );
            }

            unsafe extern "C" fn become_first_responder(_this: *mut AnyObject, _cmd: Sel) -> Bool {
                info!("CoreTermView::becomeFirstResponder called - view gained focus");
                Bool::YES
            }
            // FIX: Restore explicit cast
            unsafe {
                builder.add_method(
                    sel!(becomeFirstResponder),
                    become_first_responder as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool,
                );
            }

            unsafe extern "C" fn key_down(_this: *mut AnyObject, _cmd: Sel, _event: *mut NSEvent) {
                trace!("CoreTermView::keyDown: called - event handled, NOT calling super");
            }
            // FIX: Restore explicit cast
            unsafe {
                builder.add_method(
                    sel!(keyDown:),
                    key_down as unsafe extern "C" fn(*mut AnyObject, Sel, *mut NSEvent),
                );
            }

            builder.register();
            debug!("Registered custom NSView subclass: {}", VIEW_CLASS_NAME);
        });
    }

    fn register_delegate_class() {
        use objc2::declare::ClassBuilder;
        use std::sync::Once;
        static REGISTER_ONCE: Once = Once::new();
        REGISTER_ONCE.call_once(|| {
            let name = CStr::from_bytes_with_nul(b"CoreTermWindowDelegate\0").unwrap();
            let mut builder = ClassBuilder::new(name, class!(NSObject))
                .expect("Failed to create CoreTermWindowDelegate class");

            unsafe extern "C" fn window_should_close(
                _this: *mut AnyObject,
                _cmd: Sel,
                _sender: *mut NSWindow,
            ) -> Bool {
                info!("CoreTermWindowDelegate::windowShouldClose: - allowing window to close");
                Bool::YES
            }
            // FIX: Restore explicit cast
            unsafe {
                builder.add_method(
                    sel!(windowShouldClose:),
                    window_should_close
                        as unsafe extern "C" fn(*mut AnyObject, Sel, *mut NSWindow) -> Bool,
                );
            }

            unsafe extern "C" fn window_will_close(
                _this: *mut AnyObject,
                _cmd: Sel,
                _notification: *mut NSObject,
            ) {
                info!("CoreTermWindowDelegate::windowWillClose: - window closing");
            }
            // FIX: Restore explicit cast
            unsafe {
                builder.add_method(
                    sel!(windowWillClose:),
                    window_will_close as unsafe extern "C" fn(*mut AnyObject, Sel, *mut NSObject),
                );
            }

            builder.register();
            debug!(
                "Registered NSWindowDelegate subclass: {}",
                DELEGATE_CLASS_NAME
            );
        });
    }

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

    fn handle_init(&mut self) -> Result<DriverResponse> {
        info!("CocoaDisplayDriver: Init - showing window and returning metrics");

        let width_px = (self.window_width_pts * self.backing_scale) as u32;
        let height_px = (self.window_height_pts * self.backing_scale) as u32;

        info!(
            "CocoaDisplayDriver: {} x {} pts ({} x {} px)",
            self.window_width_pts, self.window_height_pts, width_px, height_px
        );

        let buffer_size = (width_px as usize) * (height_px as usize) * 4; // RGBA
        let framebuffer = vec![0u8; buffer_size].into_boxed_slice();
        self.framebuffer = Some(framebuffer);

        // Show window
        if let Some(window) = &self.window {
            if let Some(view) = &self.view {
                unsafe {
                    let _: () = msg_send![window, makeKeyAndOrderFront: None::<&NSObject>];
                    let _: Bool = msg_send![window, makeFirstResponder: &**view];
                }
            }
        }

        Ok(DriverResponse::InitComplete {
            width_px,
            height_px,
            scale_factor: self.backing_scale,
        })
    }

    fn create_window(&mut self, width: usize, height: usize) -> Result<Retained<NSWindow>> {
        unsafe {
            let content_rect = NSRect::new(
                NSPoint::new(DEFAULT_WINDOW_X, DEFAULT_WINDOW_Y),
                NSSize::new(width as f64, height as f64),
            );
            let style = NSWindowStyleMask::Titled
                | NSWindowStyleMask::Closable
                | NSWindowStyleMask::Miniaturizable
                | NSWindowStyleMask::Resizable;

            let window = NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(self.mtm),
                content_rect,
                style,
                NSBackingStoreType::Buffered,
                false,
            );

            let title = NSString::from_str("CoreTerm");
            window.setTitle(&title);
            window.center();
            let _: () = msg_send![&window, setOpaque: Bool::YES];
            let black_color: *mut AnyObject = msg_send![class!(NSColor), blackColor];
            let _: () = msg_send![&window, setBackgroundColor: black_color];

            let delegate_class = class!(CoreTermWindowDelegate);
            let delegate: Retained<NSObject> = msg_send![delegate_class, new];

            let _: () = msg_send![&window, setDelegate: &*delegate];
            self.delegate = Some(delegate);

            info!("NSWindow created successfully");
            Ok(window)
        }
    }

    fn create_view(&self, width: f64, height: f64) -> Result<Retained<NSObject>> {
        unsafe {
            let view_class = class!(CoreTermView);

            // Explicitly use Allocated<AnyObject> to guide msg_send!
            let view_alloc: Allocated<AnyObject> = msg_send![view_class, alloc];

            let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(width, height));

            // Init consumes Allocated and returns Retained<AnyObject>
            let view_ptr: Retained<AnyObject> = msg_send![view_alloc, initWithFrame: frame];

            // Safe transmute to specific type
            let view: Retained<NSObject> = std::mem::transmute(view_ptr);

            let _: () = msg_send![&*view, setWantsLayer: Bool::YES];
            let mask: u64 = 18;
            let _: () = msg_send![&*view, setAutoresizingMask: mask];

            info!("CoreTermView created successfully");
            Ok(view)
        }
    }

    fn handle_poll_events(&mut self) -> Result<DriverResponse> {
        let mut events = Vec::new();
        unsafe {
            let app = NSApplication::sharedApplication(self.mtm);
            let timeout = NSDate::dateWithTimeIntervalSinceNow(MAX_DRAW_LATENCY_SECONDS);
            let immediate = NSDate::distantPast();
            let mut first_event = true;
            loop {
                // Use timeout for first event, then poll immediately (distantPast) for remaining events
                let event_timeout = if first_event { &timeout } else { &immediate };
                let event = app.nextEventMatchingMask_untilDate_inMode_dequeue(
                    NSEventMask::Any,
                    Some(event_timeout),
                    &NSDefaultRunLoopMode,
                    true,
                );
                if let Some(event) = event {
                    first_event = false; // We got an event, switch to non-blocking mode for draining
                    if event.r#type() == NSEventType::ApplicationDefined {
                        // Wake event - exit immediately to process pending display actions
                        debug!("cocoa: Received wake event (ApplicationDefined)");
                        let _: () = msg_send![&app, sendEvent: &*event];
                        break;
                    }
                    debug!("cocoa: Received NSEvent type={:?}", event.r#type());
                    if let Some(evt) = self.convert_event(&event) {
                        events.push(evt);
                        // Don't send events we handle to the app - skip sendEvent for our events
                    } else {
                        // Only send events we don't handle to the app
                        let _: () = msg_send![&app, sendEvent: &*event];
                    }
                } else {
                    // No more events available - exit and return what we have
                    break;
                }
            }
        }
        Ok(DriverResponse::Events(events))
    }

    fn convert_event(&self, event: &NSEvent) -> Option<DisplayEvent> {
        let event_type = event.r#type();
        debug!("cocoa: convert_event type={:?}", event_type);
        match event_type {
            NSEventType::KeyDown => {
                let chars = event.characters();
                let text = chars.map(|s| s.to_string());
                let key_code = event.keyCode();
                let symbol = Self::map_keycode_to_symbol(key_code);
                let modifiers = Self::extract_modifiers(event);
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
                    scale_factor: self.backing_scale,
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
                    scale_factor: self.backing_scale,
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
                    scale_factor: self.backing_scale,
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
                    scale_factor: self.backing_scale,
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
                    scale_factor: self.backing_scale,
                    modifiers,
                })
            }
            _ => None,
        }
    }

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

    fn handle_request_framebuffer(&mut self) -> Result<DriverResponse> {
        let buffer = self
            .framebuffer
            .take()
            .context("Framebuffer already transferred")?;
        Ok(DriverResponse::Framebuffer(buffer))
    }

    fn handle_present(
        &mut self,
        snapshot: RenderSnapshot,
    ) -> std::result::Result<DriverResponse, DisplayError> {
        let view = match &self.view {
            Some(v) => v,
            None => {
                return Err(DisplayError::PresentationFailed(
                    snapshot,
                    "View is None".to_string(),
                ))
            }
        };

        unsafe {
            // Extract dimensions from the RenderSnapshot instead of driver state
            let width = snapshot.width_px as usize;
            let height = snapshot.height_px as usize;
            let bytes_per_row = width * BYTES_PER_PIXEL;

            let data = Arc::new(snapshot.framebuffer.as_ref());
            let provider = CGDataProvider::from_buffer(data);
            let color_space = CGColorSpace::create_device_rgb();
            let image = CGImage::new(
                width,
                height,
                BITS_PER_COMPONENT,
                BITS_PER_PIXEL,
                bytes_per_row,
                &color_space,
                1,
                &provider,
                false,
                0,
            );

            let layer: *mut AnyObject = msg_send![&**view, layer];
            if layer.is_null() {
                return Err(DisplayError::PresentationFailed(
                    snapshot,
                    "View has no layer".to_string(),
                ));
            }
            let image_ref = image.as_ptr();  // Returns CGImageRef (raw pointer)
            let _: () = msg_send![layer, setContents: image_ref as *mut c_void];
            let _: () = msg_send![layer, setContentsScale: self.backing_scale];
        }
        Ok(DriverResponse::PresentComplete(snapshot))
    }

    fn handle_set_title(&mut self, title: &str) -> Result<DriverResponse> {
        if let Some(window) = &self.window {
            let ns_title = NSString::from_str(title);
            window.setTitle(&ns_title);
        }
        Ok(DriverResponse::TitleSet)
    }

    fn handle_bell(&mut self) -> Result<DriverResponse> {
        unsafe {
            let app = NSApplication::sharedApplication(self.mtm);
            let _: () = msg_send![&app, beep];
        }
        Ok(DriverResponse::BellRung)
    }

    fn handle_set_cursor_visibility(&mut self, _visible: bool) -> Result<DriverResponse> {
        Ok(DriverResponse::CursorVisibilitySet)
    }

    fn handle_copy_to_clipboard(&mut self, text: &str) -> Result<DriverResponse> {
        unsafe {
            let pasteboard = NSPasteboard::generalPasteboard();
            pasteboard.clearContents();
            let ns_string = NSString::from_str(text);
            let _: bool =
                msg_send![&pasteboard, setString: &*ns_string, forType: NSPasteboardTypeString()];
        }
        Ok(DriverResponse::ClipboardCopied)
    }

    fn handle_request_paste(&mut self) -> Result<DriverResponse> {
        Ok(DriverResponse::PasteRequested)
    }
}

impl Drop for CocoaDisplayDriver {
    fn drop(&mut self) {
        unsafe {
            if let Some(window) = &self.window {
                let _: () = msg_send![&**window, close];
            }

            // Terminate the application when driver is dropped
            let app = NSApplication::sharedApplication(self.mtm);
            app.terminate(None);
        }
        info!("CocoaDisplayDriver::drop() - Drop complete");
    }
}

#[allow(non_snake_case)]
unsafe fn NSPasteboardTypeString() -> &'static NSString {
    extern "C" {
        static NSPasteboardTypeString: &'static NSString;
    }
    NSPasteboardTypeString
}
