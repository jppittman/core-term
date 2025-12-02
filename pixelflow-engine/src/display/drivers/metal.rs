#![cfg(use_cocoa_display)]

//! Metal DisplayDriver implementation using raw FFI.
//!
//! Zero-copy pipeline:
//! 1. CPU renders into MTLBuffer (shared memory)
//! 2. Blit to CAMetalLayer drawable
//! 3. Present
//!
//! No CGImage, no copies, proper vsync.

use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, WindowId};
use crate::input::{KeySymbol, Modifiers};
use crate::platform::waker::{CocoaWaker, EventLoopWaker};
use anyhow::{anyhow, Context, Result};
use log::{debug, info, trace};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, Sel};
use objc2::{class, msg_send, sel, MainThreadOnly};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSEvent, NSEventMask,
    NSEventModifierFlags, NSEventType, NSPasteboard, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{
    MainThreadMarker, NSDate, NSDefaultRunLoopMode, NSObject, NSPoint, NSRect, NSSize, NSString,
};
use pixelflow_render::color::Rgba;
use pixelflow_render::Frame;
use std::ffi::CStr;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

#[link(name = "Metal", kind = "framework")]
#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut AnyObject;
}

// Metal pixel format: BGRA8Unorm = 80
const MTL_PIXEL_FORMAT_BGRA8_UNORM: u64 = 80;
// MTLResourceStorageModeShared = 0
const MTL_RESOURCE_STORAGE_MODE_SHARED: u64 = 0;

const DEFAULT_WINDOW_X: f64 = 100.0;
const DEFAULT_WINDOW_Y: f64 = 100.0;
const EVENT_TIMEOUT_SECONDS: f64 = 1.0;

// --- Window ---
struct Window {
    id: WindowId,
    width: u32,
    height: u32,
    title: String,
}

// --- Run State ---
struct RunState {
    cmd_rx: Receiver<DriverCommand<Rgba>>,
    engine_tx: EngineSender<Rgba>,
}

// --- Display Driver ---

pub struct MetalDisplayDriver {
    cmd_tx: SyncSender<DriverCommand<Rgba>>,
    waker: CocoaWaker,
    run_state: Option<RunState>,
}

impl Clone for MetalDisplayDriver {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            waker: self.waker.clone(),
            run_state: None,
        }
    }
}

impl DisplayDriver for MetalDisplayDriver {
    type Pixel = Rgba;

    fn new(engine_tx: EngineSender<Rgba>) -> Result<Self> {
        let (cmd_tx, cmd_rx) = sync_channel(16);

        Ok(Self {
            cmd_tx,
            waker: CocoaWaker::new(),
            run_state: Some(RunState { cmd_rx, engine_tx }),
        })
    }

    fn send(&self, cmd: DriverCommand<Rgba>) -> Result<()> {
        self.cmd_tx.send(cmd)?;
        let _ = self.waker.wake();
        Ok(())
    }

    fn run(&self) -> Result<()> {
        let run_state = self
            .run_state
            .as_ref()
            .ok_or_else(|| anyhow!("Only original driver can run"))?;

        run_event_loop(&run_state.cmd_rx, &run_state.engine_tx)
    }
}

// --- Metal State ---

struct MetalState {
    device: *mut AnyObject,
    command_queue: *mut AnyObject,
    layer: *mut AnyObject,
    buffer: *mut AnyObject,
    buffer_ptr: *mut u8,
    buffer_size: usize,
    width: usize,
    height: usize,
}

impl MetalState {
    unsafe fn new(layer: *mut AnyObject, width: usize, height: usize) -> Result<Self> {
        // Create device
        let device = MTLCreateSystemDefaultDevice();
        if device.is_null() {
            return Err(anyhow!("Failed to create Metal device"));
        }
        info!("Metal: Device created");

        // Configure layer
        let _: () = msg_send![layer, setDevice: device];
        let _: () = msg_send![layer, setPixelFormat: MTL_PIXEL_FORMAT_BGRA8_UNORM];
        let _: () = msg_send![layer, setFramebufferOnly: Bool::NO];

        // Create command queue
        let command_queue: *mut AnyObject = msg_send![device, newCommandQueue];
        if command_queue.is_null() {
            return Err(anyhow!("Failed to create command queue"));
        }

        // Create shared buffer
        let buffer_size = width * height * 4;
        let buffer: *mut AnyObject = msg_send![
            device,
            newBufferWithLength: buffer_size as u64
            options: MTL_RESOURCE_STORAGE_MODE_SHARED
        ];
        if buffer.is_null() {
            return Err(anyhow!("Failed to create Metal buffer"));
        }

        let buffer_ptr: *mut u8 = msg_send![buffer, contents];
        info!("Metal: Buffer created {}x{} ({} bytes)", width, height, buffer_size);

        Ok(Self {
            device,
            command_queue,
            layer,
            buffer,
            buffer_ptr,
            buffer_size,
            width,
            height,
        })
    }

    unsafe fn resize(&mut self, width: usize, height: usize) -> Result<()> {
        if width == self.width && height == self.height {
            return Ok(());
        }

        let buffer_size = width * height * 4;
        let buffer: *mut AnyObject = msg_send![
            self.device,
            newBufferWithLength: buffer_size as u64
            options: MTL_RESOURCE_STORAGE_MODE_SHARED
        ];
        if buffer.is_null() {
            return Err(anyhow!("Failed to create resized Metal buffer"));
        }

        // Release old buffer
        let _: () = msg_send![self.buffer, release];

        self.buffer = buffer;
        self.buffer_ptr = msg_send![buffer, contents];
        self.buffer_size = buffer_size;
        self.width = width;
        self.height = height;

        info!("Metal: Resized to {}x{}", width, height);
        Ok(())
    }

    unsafe fn present(&mut self, frame: &Frame<Rgba>) -> Result<()> {
        let frame_size = frame.width as usize * frame.height as usize * 4;

        // Resize if needed
        if frame_size != self.buffer_size {
            self.resize(frame.width as usize, frame.height as usize)?;
        }

        // Copy frame data to Metal buffer (this is the only copy - CPU to shared memory)
        // On Apple Silicon this is effectively zero-copy since it's unified memory
        let src = frame.as_bytes();
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.buffer_ptr, src.len());

        // Get next drawable
        let drawable: *mut AnyObject = msg_send![self.layer, nextDrawable];
        if drawable.is_null() {
            return Err(anyhow!("Failed to get drawable"));
        }

        let texture: *mut AnyObject = msg_send![drawable, texture];

        // Create command buffer
        let command_buffer: *mut AnyObject = msg_send![self.command_queue, commandBuffer];

        // Copy from buffer to texture using replaceRegion (CPU -> texture, no encoder needed)
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MTLOrigin { x: u64, y: u64, z: u64 }
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MTLSize { width: u64, height: u64, depth: u64 }
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct MTLRegion { origin: MTLOrigin, size: MTLSize }

        unsafe impl objc2::Encode for MTLOrigin {
            const ENCODING: objc2::Encoding = objc2::Encoding::Struct(
                "MTLOrigin",
                &[objc2::Encoding::ULongLong, objc2::Encoding::ULongLong, objc2::Encoding::ULongLong]
            );
        }
        unsafe impl objc2::Encode for MTLSize {
            const ENCODING: objc2::Encoding = objc2::Encoding::Struct(
                "MTLSize",
                &[objc2::Encoding::ULongLong, objc2::Encoding::ULongLong, objc2::Encoding::ULongLong]
            );
        }
        unsafe impl objc2::Encode for MTLRegion {
            const ENCODING: objc2::Encoding = objc2::Encoding::Struct(
                "MTLRegion",
                &[MTLOrigin::ENCODING, MTLSize::ENCODING]
            );
        }

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: self.width as u64,
                height: self.height as u64,
                depth: 1
            }
        };
        let bytes_per_row = (self.width * 4) as u64;

        // replaceRegion copies CPU data directly to texture (synchronous)
        let _: () = msg_send![
            texture,
            replaceRegion: region
            mipmapLevel: 0u64
            withBytes: self.buffer_ptr
            bytesPerRow: bytes_per_row
        ];

        // Present and commit
        let _: () = msg_send![command_buffer, presentDrawable: drawable];
        let _: () = msg_send![command_buffer, commit];

        Ok(())
    }
}

impl Drop for MetalState {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.buffer, release];
            let _: () = msg_send![self.command_queue, release];
            // Device is autoreleased
        }
        info!("MetalState dropped");
    }
}

// --- Event Loop ---

fn run_event_loop(
    cmd_rx: &Receiver<DriverCommand<Rgba>>,
    engine_tx: &EngineSender<Rgba>,
) -> Result<()> {
    let mtm = MainThreadMarker::new().context("Must run on main thread")?;

    // Wait for CreateWindow command
    #[allow(deprecated)]
    let win = match cmd_rx.recv()? {
        DriverCommand::CreateWindow { id, width, height, title } => Window { id, width, height, title },
        DriverCommand::Configure(c) => {
            // Legacy fallback
            Window {
                id: WindowId::PRIMARY,
                width: (c.initial_cols * c.cell_width_px) as u32,
                height: (c.initial_rows * c.cell_height_px) as u32,
                title: "PixelFlow Metal".to_string(),
            }
        }
        other => return Err(anyhow!("Expected CreateWindow, got {:?}", other)),
    };

    info!("Metal: Creating resources for window {:?}", win.id);

    // Init app
    register_view_class();
    register_delegate_class();
    init_app(mtm)?;

    let (window, view, _delegate) = create_window_and_view(mtm, &win)?;

    // Get layer from view and set up Metal
    let layer: *mut AnyObject = unsafe { msg_send![&*view, layer] };

    let backing_scale: f64 = unsafe { msg_send![&window, backingScaleFactor] };
    let window_width_pts = win.width as f64;
    let window_height_pts = win.height as f64;
    let width_px = (window_width_pts * backing_scale) as u32;
    let height_px = (window_height_pts * backing_scale) as u32;

    // Set layer size
    unsafe {
        let _: () = msg_send![layer, setContentsScale: backing_scale];
        let bounds = NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(window_width_pts, window_height_pts),
        );
        let _: () = msg_send![layer, setBounds: bounds];
    }

    let mut metal = unsafe { MetalState::new(layer, width_px as usize, height_px as usize)? };

    info!("Metal: Window {}x{} pts ({}x{} px), scale {}",
          window_width_pts, window_height_pts, width_px, height_px, backing_scale);

    // Show window
    unsafe {
        let _: () = msg_send![&window, makeKeyAndOrderFront: std::ptr::null::<AnyObject>()];
        let _: Bool = msg_send![&window, makeFirstResponder: &*view];
    }

    // Send WindowCreated event
    let _ = engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::WindowCreated {
        id: win.id,
        width_px,
        height_px,
        scale: backing_scale,
    }));

    // Event loop state
    let mut state = CocoaEventState {
        mtm,
        window,
        view,
        window_id: win.id,
        window_width_pts,
        window_height_pts,
        backing_scale,
    };

    // Main loop
    loop {
        // Poll Cocoa events
        unsafe {
            let app = NSApplication::sharedApplication(mtm);
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

                    if event.r#type() == NSEventType::ApplicationDefined {
                        let _: () = msg_send![&app, sendEvent: &*event];
                        break;
                    }

                    if let Some(display_event) = state.convert_event(&event) {
                        if matches!(display_event, DisplayEvent::CloseRequested { .. }) {
                            info!("Metal: CloseRequested");
                            return Ok(());
                        }
                        let _ = engine_tx.send(EngineCommand::DisplayEvent(display_event));
                    } else {
                        let _: () = msg_send![&app, sendEvent: &*event];
                    }
                } else {
                    break;
                }
            }
        }

        // Process commands
        #[allow(deprecated)]
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DriverCommand::CreateWindow { .. } => {
                    // Already handled at startup
                }
                DriverCommand::DestroyWindow { id } => {
                    info!("Metal: DestroyWindow {:?}", id);
                    // TODO: multi-window support
                }
                DriverCommand::Configure(_) => {
                    // Legacy, ignored
                }
                DriverCommand::Shutdown => {
                    info!("Metal: Shutdown");
                    return Ok(());
                }
                DriverCommand::Present { id: _, frame } => {
                    if let Err(e) = unsafe { metal.present(&frame) } {
                        log::error!("Metal present error: {}", e);
                    }
                    // Return frame for reuse
                    let _ = engine_tx.send(EngineCommand::PresentComplete(frame));
                }
                DriverCommand::SetTitle { id: _, title } => {
                    let ns_title = NSString::from_str(&title);
                    state.window.setTitle(&ns_title);
                    let _ = engine_tx.send(EngineCommand::DriverAck);
                }
                DriverCommand::SetSize { id: _, width, height } => {
                    let size = NSSize::new(width as f64, height as f64);
                    state.window.setContentSize(size);
                    let _ = engine_tx.send(EngineCommand::DriverAck);
                }
                DriverCommand::CopyToClipboard(text) => {
                    state.copy_to_clipboard(&text);
                    let _ = engine_tx.send(EngineCommand::DriverAck);
                }
                DriverCommand::RequestPaste => {
                    if let Some(text) = state.request_paste() {
                        let _ = engine_tx.send(EngineCommand::DisplayEvent(
                            DisplayEvent::PasteData { text },
                        ));
                    }
                    let _ = engine_tx.send(EngineCommand::DriverAck);
                }
                DriverCommand::Bell => {
                    state.bell();
                    let _ = engine_tx.send(EngineCommand::DriverAck);
                }
            }
        }
    }
}

// --- Cocoa Event State ---

struct CocoaEventState {
    mtm: MainThreadMarker,
    window: Retained<NSWindow>,
    view: Retained<NSObject>,
    window_id: WindowId,
    window_width_pts: f64,
    window_height_pts: f64,
    backing_scale: f64,
}

impl CocoaEventState {
    fn convert_event(&self, event: &NSEvent) -> Option<DisplayEvent> {
        let event_type = event.r#type();
        trace!("Metal: event type={:?}", event_type);
        let id = self.window_id;

        match event_type {
            NSEventType::KeyDown => {
                let chars = event.characters();
                let text = chars.map(|s| s.to_string());
                let key_code = event.keyCode();
                let symbol = map_keycode_to_symbol(key_code);
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::Key { id, symbol, modifiers, text })
            }
            NSEventType::LeftMouseDown => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseButtonPress {
                    id,
                    button: 1,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    modifiers,
                })
            }
            NSEventType::RightMouseDown => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseButtonPress {
                    id,
                    button: 3,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    modifiers,
                })
            }
            NSEventType::LeftMouseUp => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseButtonRelease {
                    id,
                    button: 1,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    modifiers,
                })
            }
            NSEventType::RightMouseUp => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseButtonRelease {
                    id,
                    button: 3,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    modifiers,
                })
            }
            NSEventType::MouseMoved
            | NSEventType::LeftMouseDragged
            | NSEventType::RightMouseDragged => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                Some(DisplayEvent::MouseMove {
                    id,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    modifiers,
                })
            }
            NSEventType::ScrollWheel => {
                let location = event.locationInWindow();
                let modifiers = extract_modifiers(event);
                let dx = event.scrollingDeltaX() as f32;
                let dy = event.scrollingDeltaY() as f32;
                Some(DisplayEvent::MouseScroll {
                    id,
                    dx,
                    dy,
                    x: location.x as i32,
                    y: (self.window_height_pts - location.y) as i32,
                    modifiers,
                })
            }
            _ => None,
        }
    }

    fn copy_to_clipboard(&self, text: &str) {
        unsafe {
            let pasteboard = NSPasteboard::generalPasteboard();
            pasteboard.clearContents();
            let ns_string = NSString::from_str(text);
            let _: bool = msg_send![&pasteboard, setString: &*ns_string, forType: ns_pasteboard_type_string()];
        }
    }

    fn request_paste(&self) -> Option<String> {
        unsafe {
            let pasteboard = NSPasteboard::generalPasteboard();
            let string: Option<Retained<NSString>> =
                msg_send![&pasteboard, stringForType: ns_pasteboard_type_string()];
            string.map(|s| s.to_string())
        }
    }

    fn bell(&self) {
        unsafe {
            let app = NSApplication::sharedApplication(self.mtm);
            let _: () = msg_send![&app, beep];
        }
    }
}

// --- Helper functions (shared with cocoa.rs) ---

fn register_view_class() {
    use objc2::declare::ClassBuilder;
    use std::sync::Once;
    static REGISTER_ONCE: Once = Once::new();
    REGISTER_ONCE.call_once(|| {
        let name = CStr::from_bytes_with_nul(b"MetalTermView\0").unwrap();
        let mut builder = ClassBuilder::new(name, class!(NSView))
            .expect("Failed to create MetalTermView class");

        unsafe extern "C" fn is_flipped(_this: *mut AnyObject, _cmd: Sel) -> Bool {
            Bool::YES
        }
        unsafe extern "C" fn accepts_first_responder(_this: *mut AnyObject, _cmd: Sel) -> Bool {
            Bool::YES
        }
        unsafe extern "C" fn become_first_responder(_this: *mut AnyObject, _cmd: Sel) -> Bool {
            Bool::YES
        }
        unsafe extern "C" fn key_down(_this: *mut AnyObject, _cmd: Sel, _event: *mut NSEvent) {}
        unsafe extern "C" fn wants_update_layer(_this: *mut AnyObject, _cmd: Sel) -> Bool {
            Bool::YES
        }

        unsafe {
            builder.add_method(sel!(isFlipped), is_flipped as unsafe extern "C" fn(_, _) -> Bool);
            builder.add_method(sel!(acceptsFirstResponder), accepts_first_responder as unsafe extern "C" fn(_, _) -> Bool);
            builder.add_method(sel!(becomeFirstResponder), become_first_responder as unsafe extern "C" fn(_, _) -> Bool);
            builder.add_method(sel!(keyDown:), key_down as unsafe extern "C" fn(_, _, _));
            builder.add_method(sel!(wantsUpdateLayer), wants_update_layer as unsafe extern "C" fn(_, _) -> Bool);
        }

        builder.register();
        debug!("Registered MetalTermView class");
    });
}

fn register_delegate_class() {
    use objc2::declare::ClassBuilder;
    use std::sync::Once;
    static REGISTER_ONCE: Once = Once::new();
    REGISTER_ONCE.call_once(|| {
        let name = CStr::from_bytes_with_nul(b"MetalTermWindowDelegate\0").unwrap();
        let mut builder = ClassBuilder::new(name, class!(NSObject))
            .expect("Failed to create delegate class");

        unsafe extern "C" fn window_should_close(_: *mut AnyObject, _: Sel, _: *mut NSWindow) -> Bool {
            info!("MetalTermWindowDelegate: windowShouldClose");
            Bool::YES
        }
        unsafe extern "C" fn window_will_close(_: *mut AnyObject, _: Sel, _: *mut NSObject) {
            info!("MetalTermWindowDelegate: windowWillClose");
        }

        unsafe {
            builder.add_method(
                sel!(windowShouldClose:),
                window_should_close as unsafe extern "C" fn(_, _, _) -> Bool,
            );
            builder.add_method(
                sel!(windowWillClose:),
                window_will_close as unsafe extern "C" fn(_, _, _),
            );
        }

        builder.register();
        debug!("Registered MetalTermWindowDelegate class");
    });
}

fn init_app(mtm: MainThreadMarker) -> Result<()> {
    unsafe {
        let app = NSApplication::sharedApplication(mtm);
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
        let _: () = msg_send![&app, finishLaunching];
        let _: () = msg_send![&app, activateIgnoringOtherApps: Bool::YES];
        Ok(())
    }
}

fn create_window_and_view(
    mtm: MainThreadMarker,
    win: &Window,
) -> Result<(Retained<NSWindow>, Retained<NSObject>, Retained<NSObject>)> {
    let width = win.width as f64;
    let height = win.height as f64;

    unsafe {
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

        let title = NSString::from_str(&win.title);
        window.setTitle(&title);
        window.center();
        let _: () = msg_send![&window, setOpaque: Bool::YES];
        let black_color: *mut AnyObject = msg_send![class!(NSColor), blackColor];
        let _: () = msg_send![&window, setBackgroundColor: black_color];

        // Delegate
        let delegate_class = class!(MetalTermWindowDelegate);
        let delegate: Retained<NSObject> = msg_send![delegate_class, new];
        let _: () = msg_send![&window, setDelegate: &*delegate];

        // View with CAMetalLayer
        let view_class = class!(MetalTermView);
        let view_alloc: objc2::rc::Allocated<AnyObject> = msg_send![view_class, alloc];
        let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(width, height));
        let view_ptr: Retained<AnyObject> = msg_send![view_alloc, initWithFrame: frame];
        let view: Retained<NSObject> = std::mem::transmute(view_ptr);

        // Set up layer-backed view with CAMetalLayer
        let _: () = msg_send![&*view, setWantsLayer: Bool::YES];

        // Create and set CAMetalLayer
        let metal_layer: *mut AnyObject = msg_send![class!(CAMetalLayer), new];
        let _: () = msg_send![&*view, setLayer: metal_layer];

        let mask: u64 = 18; // NSViewWidthSizable | NSViewHeightSizable
        let _: () = msg_send![&*view, setAutoresizingMask: mask];

        let _: () = msg_send![&window, setContentView: &*view];

        info!("Metal: Window and view created");
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
