// src/platform/waker.rs
//! EventLoopWaker - Cross-thread signaling to wake the platform event loop.
//!
//! When background threads (orchestrator, PTY) receive data that requires
//! the platform to process events, they can call wake() to interrupt the
//! platform's event loop and force it to check for pending work.

use crate::error::RuntimeError;

/// Trait for waking the platform event loop from background threads.
///
/// Platform-specific implementations post events to the main thread's
/// event queue to interrupt blocking event polls.
pub trait EventLoopWaker: Send + Sync {
    /// Wake the event loop, causing it to return from blocking poll.
    fn wake(&self) -> Result<(), RuntimeError>;
}

pub struct NoOpWaker;

impl EventLoopWaker for NoOpWaker {
    fn wake(&self) -> Result<(), RuntimeError> {
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

    impl Default for X11Waker {
        fn default() -> Self {
            Self::new()
        }
    }

    impl actor_scheduler::WakeHandler for X11Waker {
        fn wake(&self) {
            <Self as EventLoopWaker>::wake(self).expect("Failed to wake X11 event loop");
        }
    }

    impl X11Waker {
        /// Create a new uninitialized waker.
        ///
        /// Call `set_target()` once the X11 window is created.
        #[must_use]
        pub fn new() -> Self {
            Self {
                inner: Arc::new(Mutex::new(None)),
            }
        }

        /// Initialize the waker with the X11 display and window.
        ///
        /// Call this from `run()` after creating the window.
        ///
        /// # Safety
        /// The display pointer must be valid.
        pub unsafe fn set_target(&self, display: *mut xlib::Display, window: xlib::Window) {
            unsafe {
                let wake_atom = xlib::XInternAtom(display, c"PIXELFLOW_WAKE".as_ptr(), xlib::False);

                let mut guard = self.inner.lock().unwrap();
                *guard = Some(WakerInner {
                    display,
                    window,
                    wake_atom,
                });
            }
        }

        /// Get the wake atom for filtering in the event loop.
        #[must_use]
        pub fn wake_atom(&self) -> Option<xlib::Atom> {
            self.inner.lock().unwrap().as_ref().map(|i| i.wake_atom)
        }

        /// Clear the waker target, making subsequent `wake()` calls no-ops.
        ///
        /// Call this before destroying the X11 window to prevent BadWindow errors.
        pub fn clear_target(&self) {
            let mut guard = self.inner.lock().unwrap();
            *guard = None;
        }
    }

    impl EventLoopWaker for X11Waker {
        fn wake(&self) -> Result<(), RuntimeError> {
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
    // Note: We use fully qualified paths to avoid import cycle or ambiguity
    use super::super::macos::cocoa::{self, NSApplication};
    use super::super::macos::sys::{self, Id, BOOL, NO};
    use super::*;
    use std::ptr;

    /// macOS implementation of EventLoopWaker using NSEvent posting.
    ///
    /// Posts a dummy NSEventTypeApplicationDefined event to NSApplication's
    /// event queue, which wakes the runloop from [NSApp nextEventMatchingMask...].
    #[derive(Clone)]
    pub struct CocoaWaker;

    impl actor_scheduler::WakeHandler for CocoaWaker {
        fn wake(&self) {
            <Self as EventLoopWaker>::wake(self).expect("Failed to wake Cocoa event loop");
        }
    }

    impl CocoaWaker {
        pub fn new() -> Self {
            Self
        }
    }

    impl EventLoopWaker for CocoaWaker {
        fn wake(&self) -> Result<(), RuntimeError> {
            unsafe {
                log::trace!("CocoaWaker: Waking up NSApp");
                let app = NSApplication::shared();

                let event_cls = sys::class(b"NSEvent\0");
                let sel_other = sys::sel(b"otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:\0");

                // Signature: (Class, SEL, NSUInteger, NSPoint, NSUInteger, double, NSInteger, id, short, long, long) -> id
                let sig: unsafe extern "C" fn(
                    Id,
                    sys::Sel,
                    u64,
                    cocoa::NSPoint,
                    u64,
                    f64,
                    i64,
                    Id,
                    i16,
                    i64,
                    i64,
                ) -> Id = std::mem::transmute(sys::objc_msgSend as *const std::ffi::c_void);

                let event = sig(
                    event_cls,
                    sel_other,
                    sys::NS_APP_DEFINED,
                    cocoa::NSPoint::new(0.0, 0.0),
                    0,
                    0.0,
                    0,
                    ptr::null_mut(),
                    0,
                    0,
                    0,
                );

                if !event.is_null() {
                    let sel_post = sys::sel(b"postEvent:atStart:\0");
                    sys::send_2::<(), Id, BOOL>(app.0, sel_post, event, NO);
                }
            }
            Ok(())
        }
    }
}
