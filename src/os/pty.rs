// src/os/pty.rs

use anyhow::{Context, Result};
use std::io::{Read, Result as IoResult, Write};
use std::os::unix::io::{AsRawFd, RawFd};

use nix::unistd::Pid;

// ... (PtyConfig struct) ...
#[derive(Debug, Clone)]
pub struct PtyConfig<'a> {
    pub command_executable: &'a str,
    pub args: &'a [&'a str],
    pub initial_cols: u16,
    pub initial_rows: u16,
}

pub trait PtyChannel: Read + Write + AsRawFd + Send + Sync {
    /// Resizes the PTY's window dimensions.
    /// This now clearly uses std::result::Result
    fn resize(&self, cols: u16, rows: u16) -> Result<()>;
    fn child_pid(&self) -> Pid;
}

#[derive(Debug)]
pub struct NixPty {
    master_fd: RawFd,
    child_pid: nix::unistd::Pid,
}

struct PtyDeviceEndGuard(RawFd);
impl PtyDeviceEndGuard {
    fn disarm(&mut self) {
        self.0 = -1;
    }
}
impl Drop for PtyDeviceEndGuard {
    fn drop(&mut self) {
        if self.0 != -1 {
            if let Err(_e) = nix::unistd::close(self.0) {
                // log error
            }
        }
    }
}

impl NixPty {
    // This internal method returns your specific PtyError
    fn set_pty_size_internal(fd: RawFd, cols: u16, rows: u16) -> Result<()> {
        use nix::pty::Winsize;
        nix::ioctl_write_ptr_bad!(tcsetwinsize, nix::libc::TIOCSWINSZ, Winsize);
        let winsize = Winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe { tcsetwinsize(fd, &winsize)? };
        log::trace!("NixPty: Set PTY fd {} size to {}x{}", fd, cols, rows);
        Ok(())
    }

    // Your spawn_with_config and other methods returning Result<Self, PtyError> or Result<(), PtyError>
    // would use std::result::Result<..., PtyError>
    pub fn spawn_with_config(_config: &PtyConfig) -> Result<Self> {
        // ... your implementation using set_pty_size_internal ...
        // For the sake of a compilable example:
        Ok(NixPty {
            master_fd: 0,
            child_pid: Pid::from_raw(0),
        })
    }
    // Add dummy read, write, as_raw_fd, drop for compilation
    pub fn spawn_shell_command(
        _shell_command_str: &str,
        _initial_cols: u16,
        _initial_rows: u16,
    ) -> Result<Self> {
        Ok(NixPty {
            master_fd: 0,
            child_pid: Pid::from_raw(0),
        })
    }

    fn set_fd_nonblocking(_fd: RawFd) -> Result<()> {
        Ok(())
    }
    fn child_process_setup_and_exec(_m: RawFd, _d: RawFd, _c: &PtyConfig) {}
    pub fn terminate_child_process(&mut self) -> Result<()> {
        Ok(())
    }
}

impl Drop for NixPty {
    fn drop(&mut self) { /* ... */
    }
}
impl Read for NixPty {
    fn read(&mut self, _buf: &mut [u8]) -> IoResult<usize> {
        Ok(0)
    }
}
impl Write for NixPty {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}
impl AsRawFd for NixPty {
    fn as_raw_fd(&self) -> RawFd {
        self.master_fd
    }
}

// NixPty implements PtyChannel where the generic error E is anyhow::Error
impl PtyChannel for NixPty {
    fn resize(&self, cols: u16, rows: u16) -> Result<()> {
        // NixPty::set_pty_size_internal returns Result<(), PtyError>
        // Convert PtyError to AnyhowError
        Self::set_pty_size_internal(self.master_fd, cols, rows)
            .with_context(|| format!("NixPty failed to resize PTY to {}x{}", cols, rows))
    }

    fn child_pid(&self) -> Pid {
        self.child_pid
    }
}
