// src/platform/waker.rs
//! EventLoopWaker - Cross-thread signaling to wake the platform event loop.
//!
//! When background threads (orchestrator, PTY) receive data that requires
//! the platform to process events, they can call wake() to interrupt the
//! platform's event loop and force it to check for pending work.

use anyhow::Result;

/// Trait for waking the platform event loop from background threads.
///
/// Platform-specific implementations post events to the main thread's
/// event queue to interrupt blocking event polls.
pub trait EventLoopWaker: Send + Sync {
    /// Wake the event loop, causing it to return from blocking poll.
    fn wake(&self) -> Result<()>;
}

pub struct NoOpWaker;

impl EventLoopWaker for NoOpWaker {
    fn wake(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(use_cocoa_display)]
pub use cocoa_waker::CocoaWaker;

#[cfg(use_x11_display)]
pub use x11_waker::X11Waker;

#[cfg(use_x11_display)]
mod x11_waker {
    use super::*;
    use std::mem;
    use std::sync::{Arc, Mutex};
    use x11::xlib;

    /// Inner state for X11Waker, populated once window exists.
    struct WakerInner {
        display: *mut xlib::Display,
        window: xlib::Window,
        wake_atom: xlib::Atom,
    }

    // SAFETY: XInitThreads is called before any X11 operations, making xlib thread-safe.
    // The display pointer and window ID can be safely shared across threads.
    unsafe impl Send for WakerInner {}
    unsafe impl Sync for WakerInner {}

    /// X11 implementation of EventLoopWaker using XSendEvent.
    ///
    /// Posts a ClientMessage event to the window's event queue, waking the
    /// event loop from XNextEvent blocking.
    ///
    /// The waker starts empty and is initialized by `set_target()` once the
    /// X11 window is created. Before initialization, `wake()` is a no-op.
    #[derive(Clone)]
    pub struct X11Waker {
        inner: Arc<Mutex<Option<WakerInner>>>,
    }

    impl X11Waker {
        /// Create a new uninitialized waker.
        ///
        /// Call `set_target()` once the X11 window is created.
        pub fn new() -> Self {
            Self {
                inner: Arc::new(Mutex::new(None)),
            }
        }

        /// Initialize the waker with the X11 display and window.
        ///
        /// Call this from `run()` after creating the window.
        pub fn set_target(&self, display: *mut xlib::Display, window: xlib::Window) {
            unsafe {
                let wake_atom = xlib::XInternAtom(
                    display,
                    b"PIXELFLOW_WAKE\0".as_ptr() as *const i8,
                    xlib::False,
                );

                let mut guard = self.inner.lock().unwrap();
                *guard = Some(WakerInner {
                    display,
                    window,
                    wake_atom,
                });
            }
        }

        /// Get the wake atom for filtering in the event loop.
        pub fn wake_atom(&self) -> Option<xlib::Atom> {
            self.inner.lock().unwrap().as_ref().map(|i| i.wake_atom)
        }
    }

    impl EventLoopWaker for X11Waker {
        fn wake(&self) -> Result<()> {
            let guard = self.inner.lock().unwrap();
            if let Some(inner) = guard.as_ref() {
                unsafe {
                    let mut event: xlib::XClientMessageEvent = mem::zeroed();
                    event.type_ = xlib::ClientMessage;
                    event.window = inner.window;
                    event.message_type = inner.wake_atom;
                    event.format = 32;

                    xlib::XSendEvent(
                        inner.display,
                        inner.window,
                        xlib::False,
                        xlib::NoEventMask,
                        &mut event as *mut _ as *mut xlib::XEvent,
                    );
                    xlib::XFlush(inner.display);
                }
            }
            // No-op if window not created yet (before run())
            Ok(())
        }
    }
}

#[cfg(use_cocoa_display)]
mod cocoa_waker {
    use super::*;
    use objc2::ffi::NSUInteger;
    use objc2::runtime::{AnyObject, Bool};
    use objc2::{class, msg_send};
    use objc2_foundation::NSPoint;
    use std::ffi::c_void;

    /// macOS implementation of EventLoopWaker using NSEvent posting.
    ///
    /// Posts a dummy NSEventTypeApplicationDefined event to NSApplication's
    /// event queue, which wakes the runloop from [NSApp nextEventMatchingMask...].
    #[derive(Clone)]
    pub struct CocoaWaker;

    impl CocoaWaker {
        pub fn new() -> Self {
            Self
        }
    }

    impl EventLoopWaker for CocoaWaker {
        fn wake(&self) -> Result<()> {
            // This method is called from background threads (Orchestrator, PTY).
            // We use unsafe raw objc calls to post an event to the main thread's runloop.
            unsafe {
                let app_class = class!(NSApplication);
                let app: *mut AnyObject = msg_send![app_class, sharedApplication];

                let ns_event_class = class!(NSEvent);
                // NSEventTypeApplicationDefined = 15
                let event_type: NSUInteger = 15;

                // Create a lightweight dummy event
                // Note: allocated raw pointer, autoreleased by factory method
                let event: *mut AnyObject = msg_send![
                    ns_event_class,
                    otherEventWithType: event_type,
                    location: NSPoint::new(0.0, 0.0),
                    modifierFlags: 0 as NSUInteger,
                    timestamp: 0.0,
                    windowNumber: 0 as NSUInteger,
                    context: std::ptr::null_mut::<c_void>(),
                    subtype: 0 as i16,
                    data1: 0 as isize,
                    data2: 0 as isize
                ];

                // Post to front of queue (atStart: YES) for immediate processing
                let _: () = msg_send![app, postEvent: event, atStart: Bool::YES];
            }
            Ok(())
        }
    }
}
