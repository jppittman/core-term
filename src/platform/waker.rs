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

#[cfg(target_os = "macos")]
pub use cocoa_waker::CocoaWaker;

#[cfg(target_os = "macos")]
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
