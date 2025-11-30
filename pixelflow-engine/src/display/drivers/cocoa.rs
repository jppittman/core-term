#![cfg(use_cocoa_display)]

//! Cocoa DisplayDriver implementation using objc2.
//!
//! Driver struct is cmd_tx + waker - trivially Clone.
//! run() reads Configure, creates Cocoa resources, runs event loop.

use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, RenderSnapshot};
use crate::input::{KeySymbol, Modifiers};
use crate::platform::waker::CocoaWaker;
use anyhow::{anyhow, Context, Result};
use core_graphics::base::CGFloat;
use core_graphics::color_space::CGColorSpace;
use core_graphics::data_provider::CGDataProvider;
use core_graphics::image::CGImage;
use foreign_types_shared::ForeignType;
use log::{debug, info, trace};
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyObject, Bool, Sel};
use objc2::{class, msg_send, sel};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSEvent, NSEventMask,
    NSEventModifierFlags, NSEventType, NSPasteboard, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{
    MainThreadMarker, NSDate, NSDefaultRunLoopMode, NSObject, NSPoint, NSRect, NSSize, NSString,
};
use std::ffi::{c_void, CStr};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;

const BYTES_PER_PIXEL: usize = 4;
const BITS_PER_COMPONENT: usize = 8;
const BITS_PER_PIXEL: usize = 32;
const DEFAULT_WINDOW_X: f64 = 100.0;
const DEFAULT_WINDOW_Y: f64 = 100.0;
// Brief timeout - waker posts events to interrupt blocking
const EVENT_TIMEOUT_SECONDS: f64 = 1.0;

// --- Run State (only original driver has this) ---
struct RunState {
    cmd_rx: Receiver<DriverCommand>,
    engine_tx: EngineSender,
}

// --- Display Driver ---

/// Cocoa display driver.
///
/// Clone to get a handle for sending commands. Only the original can run().
pub struct CocoaDisplayDriver {
    cmd_tx: SyncSender<DriverCommand>,
    waker: CocoaWaker,
    /// Only present on original, None on clones
    run_state: Option<RunState>,
}

impl Clone for CocoaDisplayDriver {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            waker: self.waker.clone(),
            run_state: None, // Clones can't run
        }
    }
}

impl DisplayDriver for CocoaDisplayDriver {
    fn new(engine_tx: EngineSender) -> Result<Self> {
        let (cmd_tx, cmd_rx) = sync_channel(16);

        Ok(Self {
            cmd_tx,
            waker: CocoaWaker::new(),
            run_state: Some(RunState { cmd_rx, engine_tx }),
        })
    }

    fn send(&self, cmd: DriverCommand) -> Result<()> {
        self.cmd_tx.send(cmd)?;
        // Wake the main thread's event loop so it checks cmd_rx
        let _ = self.waker.wake();
        Ok(())
    }

    fn run(&self) -> Result<()> {
        let run_state = self
            .run_state
            .as_ref()
            .ok_or_else(|| anyhow!("Only original driver can run (this is a clone)"))?;

        run_event_loop(&run_state.cmd_rx, &run_state.engine_tx)
    }
}

// --- Event Loop ---

fn run_event_loop(cmd_rx: &Receiver<DriverCommand>, engine_tx: &EngineSender) -> Result<()> {
    // Must be on main thread for Cocoa
    let mtm = MainThreadMarker::new().context("Cocoa driver must run on main thread")?;

    // 1. Read Configure command first
    let config = match cmd_rx.recv()? {
        DriverCommand::Configure(c) => c,
        other => return Err(anyhow!("Expected Configure, got {:?}", other)),
    };

    info!("Cocoa: Creating resources with config");

    // Register classes and init app
    register_view_class();
    register_delegate_class();
    init_app(mtm)?;

    // Calculate window size from config
    let window_width_pts = (config.initial_cols * config.cell_width_px) as f64;
    let window_height_pts = (config.initial_rows * config.cell_height_px) as f64;

    // Create window and view
    let (window, view, _delegate) =
        create_window_and_view(mtm, window_width_pts, window_height_pts)?;

    let backing_scale: CGFloat = unsafe { msg_send![&window, backingScaleFactor] };
    let width_px = (window_width_pts * backing_scale) as u32;
    let height_px = (window_height_pts * backing_scale) as u32;

    info!(
        "Cocoa: Window created {}x{} pts ({}x{} px), scale {}",
        window_width_pts, window_height_pts, width_px, height_px, backing_scale
    );

    // Show window and make view first responder
    unsafe {
        let _: () = msg_send![&window, makeKeyAndOrderFront: None::<&NSObject>];
        let _: Bool = msg_send![&window, makeFirstResponder: &*view];
    }

    // Send initial resize
    let _ = engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::Resize {
        width_px,
        height_px,
    }));

    // 2. Create state and run event loop
    let mut state = CocoaState {
        mtm,
        window,
        view,
        window_width_pts,
        window_height_pts,
        backing_scale,
    };

    state.event_loop(cmd_rx, engine_tx)
}

// --- Cocoa State (only exists during run) ---

struct CocoaState {
    mtm: MainThreadMarker,
    window: Retained<NSWindow>,
    view: Retained<NSObject>,
    window_width_pts: f64,
    window_height_pts: f64,
    backing_scale: f64,
}

impl CocoaState {
    fn event_loop(
        &mut self,
        cmd_rx: &Receiver<DriverCommand>,
        engine_tx: &EngineSender,
    ) -> Result<()> {
        loop {
            // 1. Poll Cocoa events
            unsafe {
                let app = NSApplication::sharedApplication(self.mtm);
                let timeout = NSDate::dateWithTimeIntervalSinceNow(EVENT_TIMEOUT_SECONDS);
                let immediate = NSDate::distantPast();
                let mut first_event = true;

                loop {
                    let event_timeout = if first_event { &timeout } else { &immediate };
                    let event = app.nextEventMatchingMask_untilDate_inMode_dequeue(
                        NSEventMask::Any,
                        Some(event_timeout),
                        &NSDefaultRunLoopMode,
                        true,
                    );

                    if let Some(event) = event {
                        first_event = false;

                        // Wake event - exit to check commands
                        if event.r#type() == NSEventType::ApplicationDefined {
                            debug!("Cocoa: Wake event received");
                            let _: () = msg_send![&app, sendEvent: &*event];
                            break;
                        }

                        // Convert and send display events
                        if let Some(display_event) = self.convert_event(&event) {
                            if matches!(display_event, DisplayEvent::CloseRequested) {
                                info!("Cocoa: CloseRequested, exiting event loop");
                                return Ok(());
                            }
                            let _ = engine_tx.send(EngineCommand::DisplayEvent(display_event));
                        } else {
                            // Events we don't handle go to the app
                            let _: () = msg_send![&app, sendEvent: &*event];
                        }
                    } else {
                        break;
                    }
                }
            }

            // 2. Process commands from engine
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    DriverCommand::Configure(_) => {
                        // Already configured, ignore
                    }
                    DriverCommand::Shutdown => {
                        info!("Cocoa: Shutdown command received");
                        return Ok(());
                    }
                    DriverCommand::Present(snapshot) => {
                        if let Ok(snapshot) = self.handle_present(snapshot) {
                            let _ = engine_tx.send(EngineCommand::PresentComplete(snapshot));
                        }
                    }
                    DriverCommand::SetTitle(title) => {
                        self.handle_set_title(&title);
                        let _ = engine_tx.send(EngineCommand::DriverAck);
                    }
                    DriverCommand::CopyToClipboard(text) => {
                        self.handle_copy_to_clipboard(&text);
                        let _ = engine_tx.send(EngineCommand::DriverAck);
                    }
                    DriverCommand::RequestPaste => {
                        if let Some(text) = self.handle_request_paste() {
                            let _ = engine_tx.send(EngineCommand::DisplayEvent(
                                DisplayEvent::PasteData { text },
                            ));
                        }
                        let _ = engine_tx.send(EngineCommand::DriverAck);
                    }
                    DriverCommand::Bell => {
                        self.handle_bell();
                        let _ = engine_tx.send(EngineCommand::DriverAck);
                    }
                }
            }
        }
    }

    fn convert_event(&self, event: &NSEvent) -> Option<DisplayEvent> {
        let event_type = event.r#type();
        trace!("Cocoa: convert_event type={:?}", event_type);

        match event_type {
            NSEventType::KeyDown => {
                let chars = event.characters();
                let text = chars.map(|s| s.to_string());
                let key_code = event.keyCode();
                let symbol = map_keycode_to_symbol(key_code);
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::Key {
                    symbol,
                    modifiers,
                    text,
                })
            }
            NSEventType::LeftMouseDown => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseButtonPress {
                    button: 1,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    scale_factor: self.backing_scale,
                    modifiers,
                })
            }
            NSEventType::RightMouseDown => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseButtonPress {
                    button: 3,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    scale_factor: self.backing_scale,
                    modifiers,
                })
            }
            NSEventType::LeftMouseUp => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseButtonRelease {
                    button: 1,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    scale_factor: self.backing_scale,
                    modifiers,
                })
            }
            NSEventType::RightMouseUp => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseButtonRelease {
                    button: 3,
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
                let modifiers = extract_modifiers(event);
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

    fn handle_present(&mut self, snapshot: RenderSnapshot) -> Result<RenderSnapshot> {
        unsafe {
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

            let layer: *mut AnyObject = msg_send![&*self.view, layer];
            if layer.is_null() {
                return Err(anyhow!("View has no layer"));
            }
            let image_ref = image.as_ptr();
            let _: () = msg_send![layer, setContents: image_ref as *mut c_void];
            let _: () = msg_send![layer, setContentsScale: self.backing_scale];
        }
        Ok(snapshot)
    }

    fn handle_set_title(&self, title: &str) {
        let ns_title = NSString::from_str(title);
        self.window.setTitle(&ns_title);
    }

    fn handle_bell(&self) {
        unsafe {
            let app = NSApplication::sharedApplication(self.mtm);
            let _: () = msg_send![&app, beep];
        }
    }

    fn handle_copy_to_clipboard(&self, text: &str) {
        unsafe {
            let pasteboard = NSPasteboard::generalPasteboard();
            pasteboard.clearContents();
            let ns_string = NSString::from_str(text);
            let _: bool = msg_send![&pasteboard, setString: &*ns_string, forType: ns_pasteboard_type_string()];
        }
    }

    fn handle_request_paste(&self) -> Option<String> {
        unsafe {
            let pasteboard = NSPasteboard::generalPasteboard();
            let string: Option<Retained<NSString>> =
                msg_send![&pasteboard, stringForType: ns_pasteboard_type_string()];
            string.map(|s| s.to_string())
        }
    }
}

impl Drop for CocoaState {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![&*self.window, close];
            let app = NSApplication::sharedApplication(self.mtm);
            app.stop(None);
        }
        info!("CocoaState dropped - resources cleaned up");
    }
}

// --- Helper functions ---

fn register_view_class() {
    use objc2::declare::ClassBuilder;
    use std::sync::Once;
    static REGISTER_ONCE: Once = Once::new();
    REGISTER_ONCE.call_once(|| {
        let name = CStr::from_bytes_with_nul(b"CoreTermView\0").unwrap();
        let mut builder =
            ClassBuilder::new(name, class!(NSView)).expect("Failed to create CoreTermView class");

        unsafe extern "C" fn is_flipped(_this: *mut AnyObject, _cmd: Sel) -> Bool {
            Bool::YES
        }
        unsafe {
            builder.add_method(
                sel!(isFlipped),
                is_flipped as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool,
            );
        }

        unsafe extern "C" fn accepts_first_responder(_this: *mut AnyObject, _cmd: Sel) -> Bool {
            Bool::YES
        }
        unsafe {
            builder.add_method(
                sel!(acceptsFirstResponder),
                accepts_first_responder as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool,
            );
        }

        unsafe extern "C" fn become_first_responder(_this: *mut AnyObject, _cmd: Sel) -> Bool {
            Bool::YES
        }
        unsafe {
            builder.add_method(
                sel!(becomeFirstResponder),
                become_first_responder as unsafe extern "C" fn(*mut AnyObject, Sel) -> Bool,
            );
        }

        unsafe extern "C" fn key_down(_this: *mut AnyObject, _cmd: Sel, _event: *mut NSEvent) {
            // Don't call super - we handle key events ourselves
        }
        unsafe {
            builder.add_method(
                sel!(keyDown:),
                key_down as unsafe extern "C" fn(*mut AnyObject, Sel, *mut NSEvent),
            );
        }

        builder.register();
        debug!("Registered CoreTermView class");
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
            info!("CoreTermWindowDelegate: windowShouldClose");
            Bool::YES
        }
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
            info!("CoreTermWindowDelegate: windowWillClose");
        }
        unsafe {
            builder.add_method(
                sel!(windowWillClose:),
                window_will_close as unsafe extern "C" fn(*mut AnyObject, Sel, *mut NSObject),
            );
        }

        builder.register();
        debug!("Registered CoreTermWindowDelegate class");
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

fn create_window_and_view(
    mtm: MainThreadMarker,
    width: f64,
    height: f64,
) -> Result<(Retained<NSWindow>, Retained<NSObject>, Retained<NSObject>)> {
    unsafe {
        // Create window
        let content_rect = NSRect::new(
            NSPoint::new(DEFAULT_WINDOW_X, DEFAULT_WINDOW_Y),
            NSSize::new(width, height),
        );
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Miniaturizable
            | NSWindowStyleMask::Resizable;

        let window = NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            content_rect,
            style,
            NSBackingStoreType::Buffered,
            false,
        );

        let title = NSString::from_str("PixelFlow");
        window.setTitle(&title);
        window.center();
        let _: () = msg_send![&window, setOpaque: Bool::YES];
        let black_color: *mut AnyObject = msg_send![class!(NSColor), blackColor];
        let _: () = msg_send![&window, setBackgroundColor: black_color];

        // Create delegate
        let delegate_class = class!(CoreTermWindowDelegate);
        let delegate: Retained<NSObject> = msg_send![delegate_class, new];
        let _: () = msg_send![&window, setDelegate: &*delegate];

        // Create view
        let view_class = class!(CoreTermView);
        let view_alloc: Allocated<AnyObject> = msg_send![view_class, alloc];
        let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(width, height));
        let view_ptr: Retained<AnyObject> = msg_send![view_alloc, initWithFrame: frame];
        let view: Retained<NSObject> = std::mem::transmute(view_ptr);

        let _: () = msg_send![&*view, setWantsLayer: Bool::YES];
        let mask: u64 = 18; // NSViewWidthSizable | NSViewHeightSizable
        let _: () = msg_send![&*view, setAutoresizingMask: mask];

        // Set view as content view
        let _: () = msg_send![&window, setContentView: &*view];

        info!("Cocoa window and view created");
        Ok((window, view, delegate))
    }
}

fn extract_modifiers(event: &NSEvent) -> Modifiers {
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

fn map_keycode_to_symbol(keycode: u16) -> KeySymbol {
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

#[allow(non_snake_case)]
unsafe fn ns_pasteboard_type_string() -> &'static NSString {
    extern "C" {
        static NSPasteboardTypeString: &'static NSString;
    }
    NSPasteboardTypeString
}
