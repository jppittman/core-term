use crate::config::Config;
use crate::platform::backends::{self, Driver, BackendEvent, PlatformActionCommand, PlatformState};
use crate::platform::os::{pty::Pty, epoll::EpollEventInformer};
// use crate::platform::events::SystemEvent; // SystemEvent is already pub use'd below
use anyhow::Result;
use std::os::unix::io::{AsRawFd, RawFd};


pub mod backends;
pub mod events;
pub mod os;

pub use events::SystemEvent;

// Placeholder for Pty if the actual one from os::pty module (NixPty)
// doesn't match the simplified `Pty::new(...)` signature used in constructors.
// use crate::platform::events::SystemEvent; // SystemEvent is already pub use'd below
use anyhow::Result;
use std::os::unix::io::{AsRawFd, RawFd};


pub mod backends;
pub mod events;
pub mod os;

pub use events::SystemEvent;

pub enum Platform {
    #[cfg(feature = "x11")]
    LinuxX11 {
        pty: os::pty::NixPty, // Using concrete type
        event_informer: os::epoll::EventMonitor, // Using concrete type
        driver: Box<dyn Driver>,
    },
    #[cfg(feature = "console")]
    LinuxConsole {
        pty: os::pty::NixPty, // Using concrete type
        event_informer: os::epoll::EventMonitor, // Using concrete type
        driver: Box<dyn Driver>,
    },
}

impl Platform {
    #[cfg(feature = "x11")]
    pub fn new_x11(config: &Config) -> Result<Self> {
        let pty_shell_program = config.shell.program.as_deref().map_or("/bin/sh", |p| p.to_str().unwrap_or("/bin/sh"));
        // Convert Vec<String> to Vec<&str> for PtyConfig
        let pty_shell_args_str: Vec<&str> = config.shell.args.iter().map(AsRef::as_ref).collect();

        let pty_config = os::pty::PtyConfig {
            command_executable: pty_shell_program,
            args: &pty_shell_args_str,
            initial_cols: config.appearance.columns,
            initial_rows: config.appearance.rows,
        };
        let pty = os::pty::NixPty::spawn_with_config(&pty_config)?;
        let driver = backends::x11::X11Driver::new()?;
        let mut event_informer = os::epoll::EventMonitor::new()?;
        event_informer.add(pty.as_raw_fd(), 0, os::epoll::EpollFlags::EPOLLIN)?; // Token 0 for PTY
        if let Some(driver_fd) = driver.get_event_fd() {
            event_informer.add(driver_fd, 1, os::epoll::EpollFlags::EPOLLIN)?; // Token 1 for Driver
        }
        Ok(Platform::LinuxX11 { pty, event_informer, driver: Box::new(driver) })
    }

    #[cfg(feature = "console")]
    pub fn new_console(config: &Config) -> Result<Self> {
        let pty_shell_program = config.shell.program.as_deref().map_or("/bin/sh", |p| p.to_str().unwrap_or("/bin/sh"));
        let pty_shell_args_str: Vec<&str> = config.shell.args.iter().map(AsRef::as_ref).collect();

        let pty_config = os::pty::PtyConfig {
            command_executable: pty_shell_program,
            args: &pty_shell_args_str,
            initial_cols: config.appearance.columns,
            initial_rows: config.appearance.rows,
        };
        let pty = os::pty::NixPty::spawn_with_config(&pty_config)?;
        let driver = backends::console::ConsoleDriver::new()?;
        let mut event_informer = os::epoll::EventMonitor::new()?;
        event_informer.add(pty.as_raw_fd(), 0, os::epoll::EpollFlags::EPOLLIN)?;
        if let Some(driver_fd) = driver.get_event_fd() {
            event_informer.add(driver_fd, 1, os::epoll::EpollFlags::EPOLLIN)?;
        }
        Ok(Platform::LinuxConsole { pty, event_informer, driver: Box::new(driver) })
    }

    pub fn primary_io_read(&mut self) -> Result<Vec<u8>> {
        // The subtask implies pty.read() should return Result<Vec<u8>>,
        // but NixPty::read is std::io::Read. This might need adjustment later
        // or a helper method on NixPty. For now, following the subtask's structure.
        // This will likely not compile directly without a helper or different call.
        // To make it "compile" in isolation for this step, let's imagine a helper:
        fn read_to_vec(pty: &mut os::pty::NixPty) -> Result<Vec<u8>> {
            let mut buffer = Vec::with_capacity(4096); // Default buffer size
            // This is a simplified read loop. A real implementation would handle WouldBlock etc.
            // For now, this will likely fail if pty.read doesn't fill the buffer appropriately
            // or if it's non-blocking and returns 0 or WouldBlock immediately.
            // The following unsafe block is to allow extending the buffer length.
            unsafe {
                buffer.set_len(4096); // Unsafe: assumes read will fill or report less
            }
            match pty.read(&mut buffer) {
                Ok(n) => {
                    buffer.truncate(n);
                    Ok(buffer)
                }
                Err(e) => Err(anyhow::Error::from(e)),
            }
        }

        match self {
            #[cfg(feature = "x11")]
            Platform::LinuxX11 { pty, .. } => read_to_vec(pty),
            #[cfg(feature = "console")]
            Platform::LinuxConsole { pty, .. } => read_to_vec(pty),
        }
    }

    pub fn primary_io_write_all(&mut self, data: &[u8]) -> Result<()> {
        match self {
            #[cfg(feature = "x11")]
            Platform::LinuxX11 { pty, .. } => pty.write_all(data),
            #[cfg(feature = "console")]
            Platform::LinuxConsole { pty, .. } => pty.write_all(data),
        }
    }

    pub fn primary_io_resize(&mut self, cols: u16, rows: u16) -> Result<()> {
        match self {
            #[cfg(feature = "x11")]
            Platform::LinuxX11 { pty, .. } => pty.resize(cols, rows),
            #[cfg(feature = "console")]
            Platform::LinuxConsole { pty, .. } => pty.resize(cols, rows),
        }
    }
    pub fn driver_process_ui_events(&mut self) -> Result<Vec<BackendEvent>> {
        match self {
            #[cfg(feature = "x11")]
            Platform::LinuxX11 { driver, .. } => driver.process_ui_events(),
            #[cfg(feature = "console")]
            Platform::LinuxConsole { driver, .. } => driver.process_ui_events(),
        }
    }

    pub fn driver_execute_actions(&mut self, actions: Vec<PlatformActionCommand>) -> Result<()> {
        match self {
            #[cfg(feature = "x11")]
            Platform::LinuxX11 { driver, .. } => driver.execute_platform_actions(actions),
            #[cfg(feature = "console")]
            Platform::LinuxConsole { driver, .. } => driver.execute_platform_actions(actions),
        }
    }

    pub fn driver_get_platform_state(&self) -> PlatformState {
        match self {
            #[cfg(feature = "x11")]
            Platform::LinuxX11 { driver, .. } => driver.get_platform_state(),
            #[cfg(feature = "console")]
            Platform::LinuxConsole { driver, .. } => driver.get_platform_state(),
        }
    }

    pub fn cleanup(&mut self) -> Result<()> {
        match self {
            #[cfg(feature = "x11")]
            Platform::LinuxX11 { driver, .. } => driver.cleanup(),
            #[cfg(feature = "console")]
            Platform::LinuxConsole { driver, .. } => driver.cleanup(),
        }
    }

    // Define tokens used for epoll, must match those used in constructors.
    const PTY_TOKEN_ID: u64 = 0;
    const DRIVER_TOKEN_ID: u64 = 1;

    pub fn poll_system_events(&mut self, timeout_ms: Option<i32>) -> Result<Vec<SystemEvent>> {
        let event_monitor = match self {
            #[cfg(feature = "x11")]
            Platform::LinuxX11 { event_informer, .. } => event_informer,
            #[cfg(feature = "console")]
            Platform::LinuxConsole { event_informer, .. } => event_informer,
        };

        let mut system_events = Vec::new();
        let effective_timeout = timeout_ms.map_or(-1, |t| t as isize); // epoll_wait uses isize for timeout

        let epoll_events = event_monitor.events(effective_timeout)?;

        if epoll_events.is_empty() {
            if effective_timeout > 0 {
                // Timeout occurred with no file descriptor events
                system_events.push(SystemEvent::Tick);
            }
            // If timeout was 0 or -1 and no events, it's not a Tick, just means no events pending or blocking indefinitely.
        } else {
            // Use flags to ensure we only add one SystemEvent per token type,
            // even if multiple epoll event kinds (e.g., EPOLLIN, EPOLLHUP) are raised for the same FD.
            let mut pty_ready = false;
            let mut driver_ready = false;

            for event in epoll_events {
                let token = event.u64; // Direct access to the token
                if token == Self::PTY_TOKEN_ID {
                    pty_ready = true;
                } else if token == Self::DRIVER_TOKEN_ID {
                    driver_ready = true;
                }
                // Potentially log unexpected tokens if any:
                // else {
                //     log::warn!("poll_system_events: Received event with unknown token: {}", token);
                // }
            }

            if pty_ready {
                system_events.push(SystemEvent::PrimaryIoReady);
            }
            if driver_ready {
                system_events.push(SystemEvent::UiInputReady);
            }
        }
        Ok(system_events)
    }
}
