// src/os/pty.rs

use anyhow::{Context, Result};
use std::io::{Read, Result as IoResult, Write};
use std::os::unix::io::{AsFd, AsRawFd, OwnedFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::sync::Arc;

use nix::fcntl::{fcntl, FcntlArg, FdFlag, OFlag};
use nix::pty::openpty;
use nix::sys::signal::{kill, Signal};
use nix::sys::termios;
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::Pid;
use std::io::{Error as IoError, ErrorKind as IoErrorKind};

/// Configuration for spawning a PTY.
#[derive(Debug, Clone)]
pub struct PtyConfig<'a> {
    /// The executable to run (e.g., "/bin/bash").
    pub command_executable: &'a str,
    /// Arguments to the executable.
    pub args: &'a [&'a str],
    /// Initial columns.
    pub initial_cols: u16,
    /// Initial rows.
    pub initial_rows: u16,
}

/// Trait abstracting a PTY channel.
///
/// Allows reading/writing data and managing the PTY session (resizing, PID access).
pub trait PtyChannel: Read + Write + AsRawFd + Send + Sync {
    /// Resizes the PTY window.
    ///
    /// # Parameters
    /// * `cols` - New width in columns.
    /// * `rows` - New height in rows.
    fn resize(&self, cols: u16, rows: u16) -> Result<()>;

    /// Returns the process ID of the child process attached to the PTY.
    fn child_pid(&self) -> Pid;
}

/// Implementation of `PtyChannel` using `nix` for POSIX systems.
#[derive(Debug)]
pub struct NixPty {
    master_fd: Arc<OwnedFd>,
    child_pid: Option<Pid>,
}

impl NixPty {
    fn set_pty_size_internal<Fd: AsFd>(fd: Fd, cols: u16, rows: u16) -> anyhow::Result<()> {
        use nix::pty::Winsize;
        let raw_fd = fd.as_fd().as_raw_fd();
        nix::ioctl_write_ptr_bad!(tcsetwinsize, nix::libc::TIOCSWINSZ, Winsize);
        let winsize = Winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe { tcsetwinsize(raw_fd, &winsize) }
            .map_err(|e| anyhow::anyhow!("ioctl TIOCSWINSZ failed for fd {}: {}", raw_fd, e))?;
        log::trace!("NixPty: Set PTY fd {} size to {}x{}", raw_fd, cols, rows);
        Ok(())
    }

    /// Spawns a new process connected to a PTY using the given configuration.
    ///
    /// This uses `std::process::Command` to handle forking safely, avoiding potential
    /// deadlocks in multi-threaded environments (like `cargo test`) that can occur with
    /// manual `fork` and non-async-signal-safe code (e.g. `malloc` in `eprintln!`).
    ///
    /// # Parameters
    /// * `config` - Configuration for the PTY and command.
    ///
    /// # Returns
    /// * A new `NixPty` instance in the parent process.
    pub fn spawn_with_config(config: &PtyConfig) -> Result<Self> {
        let pty_results =
            openpty(None, None).with_context(|| "Failed to open PTY (nix::pty::openpty call)")?;
        let master_fd = pty_results.master;
        let slave_fd = pty_results.slave;

        // Set FD_CLOEXEC on both master and slave so they are closed in the child upon exec.
        // We will dup2 the slave to 0,1,2 in pre_exec, so the duplicated FDs will remain open,
        // while the original slave_fd (and master_fd) will be closed.
        Self::set_cloexec(&master_fd)?;
        Self::set_cloexec(&slave_fd)?;

        // Configure slave PTY attributes (in the parent, operating on the slave FD).
        // This is safe because slave_fd refers to the same underlying PTY.
        let mut termios_attrs =
            termios::tcgetattr(&slave_fd).with_context(|| "Failed to get terminal attributes")?;
        termios::cfmakeraw(&mut termios_attrs);
        termios_attrs.local_flags |= termios::LocalFlags::ISIG;
        termios_attrs.input_flags |= termios::InputFlags::ICRNL;
        termios::tcsetattr(&slave_fd, termios::SetArg::TCSANOW, &termios_attrs)
            .with_context(|| "Failed to set terminal attributes to raw mode")?;

        let mut cmd = Command::new(config.command_executable);
        cmd.args(config.args);

        // Pre-exec setup in the child.
        // Must use only async-signal-safe functions (no malloc, no locks).
        let slave_raw_fd = slave_fd.as_raw_fd();
        unsafe {
            cmd.pre_exec(move || {
                // Create new session
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }

                // Set controlling terminal
                // On macOS/BSD, TIOCSCTTY is unsigned long, but ioctl expects unsigned long.
                // However, constants might be inferred as u32.
                // We cast both arguments to ensure compatibility across platforms.
                if libc::ioctl(slave_raw_fd, libc::TIOCSCTTY as _, 0 as libc::c_int) == -1 {
                    return Err(std::io::Error::last_os_error());
                }

                // Dup2 slave to stdin, stdout, stderr
                if libc::dup2(slave_raw_fd, libc::STDIN_FILENO) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::dup2(slave_raw_fd, libc::STDOUT_FILENO) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::dup2(slave_raw_fd, libc::STDERR_FILENO) == -1 {
                    return Err(std::io::Error::last_os_error());
                }

                // We don't need to manually close slave_raw_fd or master_fd here
                // because we set FD_CLOEXEC on them in the parent.
                // dup2 clears FD_CLOEXEC on the new FDs (0, 1, 2), so they will persist.

                Ok(())
            });
        }

        log::debug!(
            "Spawning command: {} with args {:?}",
            config.command_executable,
            config.args
        );

        // Spawn the process
        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn command '{}'", config.command_executable))?;

        let child_pid = Pid::from_raw(child.id() as i32);
        log::debug!(
            "Parent: Spawned child with PID {}, PTY master FD {}",
            child_pid,
            master_fd.as_raw_fd()
        );

        // In parent: configure master PTY size and non-blocking I/O
        Self::set_pty_size_internal(&master_fd, config.initial_cols, config.initial_rows)
            .with_context(|| "Parent: Failed to set initial PTY size")?;

        Self::set_fd_nonblocking(&master_fd)
            .with_context(|| "Parent: Failed to set master PTY to non-blocking")?;

        // slave_fd is dropped here, closing the parent's handle to the slave PTY.
        // The child has its own copies (0, 1, 2).

        Ok(NixPty {
            master_fd: Arc::new(master_fd),
            child_pid: Some(child_pid),
        })
    }

    /// Creates a clone of the PTY handle for reading.
    /// The clone shares the file descriptor but does not own the child process.
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            master_fd: self.master_fd.clone(),
            child_pid: None,
        })
    }

    /// Helper to spawn a shell command (deprecated/unimplemented convenience method).
    pub fn spawn_shell_command(
        _shell_command_str: &str,
        _initial_cols: u16,
        _initial_rows: u16,
    ) -> Result<Self> {
        unimplemented!("spawn_shell_command is not fully implemented with OwnedFd yet.");
    }

    fn set_cloexec<Fd: AsFd>(fd: Fd) -> Result<()> {
        let raw_fd = fd.as_fd().as_raw_fd();
        fcntl(fd.as_fd(), FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))
            .with_context(|| format!("Failed to set FD_CLOEXEC for fd {}", raw_fd))?;
        log::trace!("NixPty: Set FD_CLOEXEC on fd {}", raw_fd);
        Ok(())
    }

    fn set_fd_nonblocking<Fd: AsFd>(fd: Fd) -> Result<()> {
        let raw_fd = fd.as_fd().as_raw_fd();
        let flags = fcntl(fd.as_fd(), FcntlArg::F_GETFL)
            .with_context(|| format!("Failed to get FD flags for fd {}", raw_fd))?;
        let mut non_blocking_flags = OFlag::from_bits_truncate(flags);
        non_blocking_flags.insert(OFlag::O_NONBLOCK);
        fcntl(fd.as_fd(), FcntlArg::F_SETFL(non_blocking_flags))
            .with_context(|| format!("Failed to set FD {} to non-blocking", raw_fd))?;
        log::trace!("NixPty: Set FD {} to non-blocking", raw_fd);
        Ok(())
    }

    /// Terminates the child process.
    #[allow(dead_code)]
    pub fn terminate_child_process(&mut self) -> Result<()> {
        if let Some(pid) = self.child_pid {
            log::info!("Terminating child process {}", pid);
            kill(pid, Some(Signal::SIGKILL))
                .with_context(|| format!("Failed to send SIGKILL to child process {}", pid))
        } else {
            Ok(())
        }
    }
}

impl Drop for NixPty {
    fn drop(&mut self) {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::debug!(
            "NixPty drop: Cleaning up PTY master_fd: {} (child_pid: {:?})",
            master_raw_fd,
            self.child_pid
        );
        // self.master_fd (Arc<OwnedFd>) is dropped automatically.
        // The underlying FD closes only when strong_count hits 0.

        let pid = match self.child_pid {
            Some(p) => p,
            None => {
                // This is a clone (Read Thread), so we don't manage the child process.
                return;
            }
        };

        if pid.as_raw() <= 0 {
            log::debug!(
                "NixPty drop: Invalid child PID ({}), skipping child process handling.",
                pid
            );
            return;
        }

        match waitpid(pid, Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::StillAlive) => {
                log::debug!(
                    "NixPty drop: Child process {} is still alive. Sending SIGHUP.",
                    pid
                );
                if let Err(e) = kill(pid, Some(Signal::SIGHUP)) {
                    log::warn!(
                        "NixPty drop: Failed to send SIGHUP to child process {}: {}",
                        pid,
                        e
                    );
                } else {
                    log::debug!(
                        "NixPty drop: Successfully sent SIGHUP to child process {}.",
                        pid
                    );
                }
            }
            Ok(status) => {
                log::debug!(
                    "NixPty drop: Child process {} already exited or changed state: {:?}",
                    pid,
                    status
                );
            }
            Err(e) => {
                // nix::Error
                if matches!(e, nix::Error::ECHILD) || matches!(e, nix::Error::ESRCH) {
                    log::debug!(
                        "NixPty drop: Child process {} does not exist or is not a child (waitpid error: {}). Already reaped?",
                        pid,
                        e
                    );
                } else {
                    log::warn!(
                        "NixPty drop: Error checking child process {} status with waitpid: {}",
                        pid,
                        e
                    );
                }
            }
        }
    }
}

impl Read for NixPty {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::trace!("NixPty::read attempting to read from fd {}", master_raw_fd);
        match nix::unistd::read(&self.master_fd, buf) {
            // Pass &OwnedFd
            Ok(bytes_read) => {
                log::trace!(
                    "NixPty::read successfully read {} bytes from fd {}",
                    bytes_read,
                    master_raw_fd
                );
                Ok(bytes_read)
            }
            Err(nix::Error::EIO) => Ok(0),
            Err(nix_err) => {
                if matches!(nix_err, nix::Error::EAGAIN)
                    || matches!(nix_err, nix::Error::EWOULDBLOCK)
                {
                    log::debug!(
                        "NixPty::read on fd {}: Got {}, mapping to WouldBlock",
                        master_raw_fd,
                        nix_err
                    );
                    Err(IoError::new(IoErrorKind::WouldBlock, nix_err))
                } else {
                    log::warn!(
                        "NixPty::read on fd {}: Got unhandled nix::Error {}, mapping to Other",
                        master_raw_fd,
                        nix_err
                    );
                    Err(IoError::other(nix_err))
                }
            }
        }
    }
}

impl Write for NixPty {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::trace!(
            "NixPty::write attempting to write {} bytes to fd {}",
            buf.len(),
            master_raw_fd
        );
        match nix::unistd::write(&self.master_fd, buf) {
            // Pass &OwnedFd
            Ok(bytes_written) => {
                log::trace!(
                    "NixPty::write successfully wrote {} bytes to fd {}",
                    bytes_written,
                    master_raw_fd
                );
                Ok(bytes_written)
            }
            Err(nix_err) => {
                if matches!(nix_err, nix::Error::EAGAIN)
                    || matches!(nix_err, nix::Error::EWOULDBLOCK)
                {
                    log::debug!(
                        "NixPty::write on fd {}: Got {}, mapping to WouldBlock",
                        master_raw_fd,
                        nix_err
                    );
                    Err(IoError::new(IoErrorKind::WouldBlock, nix_err))
                } else {
                    log::warn!(
                        "NixPty::write on fd {}: Got unhandled nix::Error {}, mapping to Other",
                        master_raw_fd,
                        nix_err
                    );
                    Err(IoError::other(nix_err))
                }
            }
        }
    }

    fn flush(&mut self) -> IoResult<()> {
        log::trace!("NixPty::flush called for fd {}", self.master_fd.as_raw_fd());
        Ok(())
    }
}

impl AsRawFd for NixPty {
    fn as_raw_fd(&self) -> RawFd {
        self.master_fd.as_raw_fd()
    }
}

impl PtyChannel for NixPty {
    fn resize(&self, cols: u16, rows: u16) -> anyhow::Result<()> {
        Self::set_pty_size_internal(&*self.master_fd, cols, rows).with_context(|| {
            format!(
                "NixPty: Failed to set PTY size to {}x{} for fd {}",
                cols,
                rows,
                self.master_fd.as_raw_fd()
            )
        })?;

        if let Some(pid) = self.child_pid {
            kill(pid, Some(Signal::SIGWINCH)).with_context(|| {
                format!("NixPty: Failed to send SIGWINCH to child process {}", pid)
            })?;

            log::debug!(
                "NixPty: Resized PTY to {}x{} and sent SIGWINCH to PID {}",
                cols,
                rows,
                pid
            );
        } else {
            log::warn!(
                "NixPty: Resize called on a clone (no PID). PTY size set, but SIGWINCH not sent."
            );
        }

        Ok(())
    }

    fn child_pid(&self) -> Pid {
        self.child_pid.unwrap_or_else(|| Pid::from_raw(0))
    }
}
