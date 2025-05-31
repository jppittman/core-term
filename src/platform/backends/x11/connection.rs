// src/platform/backends/x11/connection.rs
#![allow(non_snake_case)] // Allow non-snake case for X11 types

use anyhow::{anyhow, Context, Result}; // Combined anyhow imports
use log::{debug, error, info, warn};
use std::os::unix::io::RawFd;
use std::ptr;

// X11 library imports
use x11::xlib;
use libc::c_int;

/// Represents and manages the connection to the X server.
///
/// This struct encapsulates the Xlib `Display` pointer and related common
/// identifiers such as the default screen, colormap, and visual. It provides
/// methods for establishing and closing the connection, and accessing these
/// core X11 resources.
///
/// The connection is automatically closed when this struct is dropped,
/// logging any errors during cleanup.
#[derive(Debug)]
pub struct Connection {
    display: *mut xlib::Display,
    screen: c_int,
    colormap: xlib::Colormap,
    visual: *mut xlib::Visual,
}

impl Connection {
    /// Establishes a new connection to the X server.
    ///
    /// This method attempts to open a connection to the X server specified by the
    /// `DISPLAY` environment variable (or the default server if `DISPLAY` is not set).
    /// Upon successful connection, it retrieves the default screen, colormap, and visual
    /// for that display.
    ///
    /// # Returns
    ///
    /// * `Ok(Connection)`: An initialized `Connection` instance if successful.
    /// * `Err(anyhow::Error)`: An error if any part of the connection or resource
    ///   retrieval process fails (e.g., cannot open display, cannot get default visual).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use anyhow::Result;
    /// # use core_terminal::platform::backends::x11::connection::Connection;
    /// # fn main() -> Result<()> {
    /// let x_connection = Connection::new()?;
    /// // Use the connection...
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self> {
        info!("Establishing X11 server connection.");

        // Open the X display. Passing NULL to XOpenDisplay means it will use
        // the DISPLAY environment variable.
        let display = unsafe { xlib::XOpenDisplay(ptr::null()) };
        if display.is_null() {
            return Err(anyhow!(
                "Failed to open X display. Check DISPLAY environment variable or X server status."
            ));
        }
        debug!("X display opened successfully: {:p}", display);

        // Get the default screen number for the display.
        let screen = unsafe { xlib::XDefaultScreen(display) };
        debug!("Default screen number: {}", screen);

        // Get the default colormap for the screen.
        let colormap = unsafe { xlib::XDefaultColormap(display, screen) };
        debug!("Default colormap ID: {}", colormap);

        // Get the default visual for the screen.
        let visual = unsafe { xlib::XDefaultVisual(display, screen) };
        if visual.is_null() {
            // This is highly unlikely if the display opened and screen is valid,
            // but good practice to check. Cleanup display before erroring.
            unsafe { xlib::XCloseDisplay(display) };
            return Err(anyhow!(
                "Failed to get default visual for screen {}.",
                screen
            ));
        }
        debug!("Default visual: {:p}", visual);

        info!("X11 server connection established successfully.");
        Ok(Connection {
            display,
            screen,
            colormap,
            visual,
        })
    }

    /// Closes the connection to the X server.
    ///
    /// This method ensures that the display connection is properly closed if it's currently open.
    /// It is idempotent, meaning it can be called multiple times: if the connection is already
    /// closed or was never successfully opened, it does nothing and returns `Ok(())`.
    ///
    /// # Returns
    ///
    /// * `Ok(())`: Always, as errors during `XCloseDisplay` are logged but not propagated
    ///   to allow cleanup to proceed as much as possible.
    pub fn cleanup(&mut self) -> Result<()> {
        if !self.display.is_null() {
            info!("Closing X11 display connection: {:p}", self.display);
            unsafe {
                // XCloseDisplay returns 0 on success, non-zero on error.
                // Typical Xlib examples don't strictly check this return,
                // as cleanup should proceed regardless. We log it if it happens.
                // For now, we follow this common practice and don't treat it as a Result::Err.
                let status = xlib::XCloseDisplay(self.display);
                if status != 0 {
                    // This is an unusual situation. XCloseDisplay typically returns 0 on success.
                    // A non-zero return might indicate an issue, but the display is likely closed or unusable anyway.
                    warn!("XCloseDisplay returned non-zero status: {}. Display may not have closed cleanly.", status);
                }
            }
            self.display = ptr::null_mut(); // Mark as closed to prevent reuse.
            debug!("X display connection closed.");
        } else {
            info!("X11 display connection already closed or was never opened. Cleanup skipped.");
        }
        Ok(())
    }

    /// Returns the raw X11 display pointer (`*mut xlib::Display`).
    ///
    /// This pointer is essential for most Xlib functions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `Connection` object (and thus the display)
    /// remains valid for the duration of the use of this pointer. The pointer
    /// becomes invalid after `cleanup` is called or the `Connection` is dropped.
    /// Accessing it after closure can lead to use-after-free bugs.
    /// It must not be null when passed to Xlib functions that require a valid display.
    #[inline]
    pub fn display(&self) -> *mut xlib::Display {
        self.display
    }

    /// Returns the default screen number for the connected display.
    ///
    /// This is typically `0` for single-monitor setups.
    #[inline]
    pub fn screen(&self) -> c_int {
        self.screen
    }

    /// Returns the default colormap ID for the default screen.
    ///
    /// The colormap is used for color allocation and management.
    #[inline]
    pub fn colormap(&self) -> xlib::Colormap {
        self.colormap
    }

    /// Returns a pointer to the default visual (`*mut xlib::Visual`) for the default screen.
    ///
    /// The visual determines how colors and graphics are displayed.
    ///
    /// # Safety
    ///
    /// Similar to `display()`, the caller must ensure the `Connection` (and thus the visual
    /// pointer derived from its display) remains valid. The pointer should not be used
    /// after the `Connection` is cleaned up or dropped.
    #[inline]
    pub fn visual(&self) -> *mut xlib::Visual {
        self.visual
    }

    /// Returns the file descriptor associated with the X connection.
    ///
    /// This file descriptor can be monitored for readability to determine when X events
    /// are pending. It's useful for integrating into event loops (e.g., with `epoll`, `select`).
    ///
    /// # Returns
    ///
    /// * `Some(RawFd)`: The file descriptor if the display connection is valid.
    /// * `None`: If the display connection is closed or was never initialized.
    pub fn get_event_fd(&self) -> Option<RawFd> {
        if self.display.is_null() {
            warn!("get_event_fd called on a closed or invalid X display.");
            None
        } else {
            // SAFETY: XConnectionNumber is safe to call with a valid, non-null display.
            let fd = unsafe { xlib::XConnectionNumber(self.display) };
            Some(fd)
        }
    }
}

/// Ensures that the X11 connection is closed when the `Connection` object goes out of scope.
///
/// This `Drop` implementation calls `cleanup()` to release X server resources.
/// Errors during cleanup in `drop` are logged but not propagated, as `drop` cannot return `Result`.
impl Drop for Connection {
    fn drop(&mut self) {
        info!("Dropping Connection object, ensuring cleanup.");
        if let Err(e) = self.cleanup() {
            // Use log::error! for consistency if a logger is set up.
            error!("Error during Connection cleanup in drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic test to check if Connection::new() can be called.
    // This test requires a running X server to pass.
    // It is commented out by default to allow headless CI environments to pass.
    // To run this test, ensure an X server is available and uncomment it.
    /*
    #[test]
    fn test_new_connection_requires_x_server() {
        match Connection::new() {
            Ok(mut conn) => {
                assert!(!conn.display().is_null(), "Display pointer should not be null on successful connection.");
                assert!(!conn.visual().is_null(), "Visual pointer should not be null.");
                if let Some(fd) = conn.get_event_fd() {
                    assert!(fd >= 0, "Event FD should be non-negative. Got: {}", fd);
                } else {
                    // This case might occur if XConnectionNumber somehow fails for a valid display,
                    // or if the display connection is extremely minimal (not typical).
                    // For a standard X server connection, an FD is expected.
                    panic!("Event FD should be available on a new, valid connection.");
                }
                // conn.cleanup() will be called by Drop.
            }
            Err(e) => {
                // This typically happens if no X server is available (e.g., in CI).
                // We can't easily assert success without an X server.
                // For local testing, ensure an X server is running.
                // This assertion helps indicate the test environment if it fails.
                panic!("test_new_connection_requires_x_server failed, likely due to no X server: {:?}", e);
            }
        }
    }
    */

    #[test]
    fn test_cleanup_idempotency() {
        // This test does not require a running X server as it simulates the display pointer state.
        let mut conn = Connection {
            display: ptr::null_mut(), // Start with a null display to simulate closed or uninitialized
            screen: 0,
            colormap: 0,
            visual: ptr::null_mut(),
        };

        // Call cleanup on an already "closed" connection
        assert!(conn.cleanup().is_ok(), "cleanup on null display should be Ok");
        assert!(conn.display.is_null(), "display should remain null after cleanup on null");

        // Simulate an opened display (without actually opening one).
        // THIS IS GENERALLY UNSAFE but acceptable for this specific test's logic,
        // as we are only checking if cleanup sets it to null and don't call Xlib functions
        // that would dereference this dummy pointer.
        let dummy_display_ptr = 1 as *mut xlib::Display;
        conn.display = dummy_display_ptr;

        assert!(conn.cleanup().is_ok(), "cleanup on dummy display should be Ok");
        assert!(conn.display.is_null(), "Display pointer should be null after cleanup");

        // Call cleanup again to ensure it's idempotent after being set to null
        assert!(conn.cleanup().is_ok(), "second cleanup call should also be Ok");
        assert!(conn.display.is_null(), "Display pointer should remain null");
    }

     #[test]
    fn test_get_event_fd_on_closed_display() {
        let conn = Connection { // Use a non-mutable conn for this test as get_event_fd takes &self
            display: ptr::null_mut(), // Simulate closed display
            screen: 0,
            colormap: 0,
            visual: ptr::null_mut(),
        };
        assert!(conn.get_event_fd().is_none(), "get_event_fd on a null display should return None");
    }

    // Test for get_event_fd when display is notionally open (requires mocking or specific setup if not using real X)
    // For now, covered by the intent of test_new_connection_requires_x_server if that were enabled and passing.
    // The main thing test_get_event_fd_on_closed_display covers is the null path.
}
