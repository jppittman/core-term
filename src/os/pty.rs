// src/os/pty.rs

use anyhow::{Context, Result};
use std::ffi::CString;
use std::io::{Read, Result as IoResult, Write};
use std::os::unix::io::{AsFd, AsRawFd, OwnedFd, RawFd};

use nix::fcntl::{fcntl, FcntlArg, OFlag};
use nix::pty::openpty;
use nix::sys::signal::{kill, Signal};
use nix::sys::termios;
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{fork, execvp, setsid, Pid, ForkResult}; // Added ForkResult
use std::io::{Error as IoError, ErrorKind as IoErrorKind};

#[derive(Debug, Clone)]
pub struct PtyConfig<'a> {
    pub command_executable: &'a str,
    pub args: &'a [&'a str],
    pub initial_cols: u16,
    pub initial_rows: u16,
}

pub trait PtyChannel: Read + Write + AsRawFd + Send + Sync {
    fn resize(&self, cols: u16, rows: u16) -> Result<()>;
    fn child_pid(&self) -> Pid;
}

#[derive(Debug)]
pub struct NixPty {
    master_fd: OwnedFd,
    child_pid: Pid,
}

// PtyDeviceEndGuard might be re-evaluated; OwnedFd handles its own drop.
#[derive(Debug)]
struct PtyDeviceEndGuard(OwnedFd);
impl PtyDeviceEndGuard {
    #[allow(dead_code)]
    fn disarm(&mut self) {
        // To disarm, one would typically use std::mem::forget or std::mem::take.
        // This is a placeholder if specific disarm logic is needed.
    }
}
// Drop for PtyDeviceEndGuard is implicitly handled by OwnedFd.

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

    pub fn spawn_with_config(config: &PtyConfig) -> Result<Self> {
        let pty_results = openpty(None, None)
            .with_context(|| "Failed to open PTY (nix::pty::openpty call)")?;
        let master_fd = pty_results.master;
        let slave_fd = pty_results.slave;
        
        let master_raw_fd_for_log = master_fd.as_raw_fd();

        let child_pid = match unsafe { fork() }.with_context(|| "Failed to fork process")? {
            ForkResult::Parent { child, .. } => {
                drop(slave_fd); // Parent closes its copy of slave PTY
                log::debug!(
                    "Parent: Forked child with PID {}, PTY master FD {}",
                    child,
                    master_raw_fd_for_log
                );

                Self::set_pty_size_internal(&master_fd, config.initial_cols, config.initial_rows)
                    .with_context(|| "Parent: Failed to set initial PTY size")?;

                Self::set_fd_nonblocking(&master_fd)
                    .with_context(|| "Parent: Failed to set master PTY to non-blocking")?;
                child
            }
            ForkResult::Child => {
                drop(master_fd); // Child closes its copy of master PTY
                setsid().with_context(|| "Child: Failed to create new session")?;

                let slave_raw_fd = slave_fd.as_raw_fd();

                let tiocsctty_res = unsafe { libc::ioctl(slave_raw_fd, libc::TIOCSCTTY as _, 0) };
                if tiocsctty_res == -1 {
                    return Err(anyhow::Error::from(nix::Error::last())
                        .context("Child: Failed to set PTY slave as controlling terminal (ioctl TIOCSCTTY)"));
                }
                
                let mut termios_attrs = termios::tcgetattr(&slave_fd)
                    .with_context(|| "Child: Failed to get terminal attributes")?;
                termios::cfmakeraw(&mut termios_attrs);
                termios::tcsetattr(&slave_fd, termios::SetArg::TCSANOW, &termios_attrs)
                    .with_context(|| "Child: Failed to set terminal attributes to raw mode")?;

                // Using libc::dup2 directly as nix::unistd::dup2 signature caused issues.
                if unsafe { libc::dup2(slave_raw_fd, libc::STDIN_FILENO) } == -1 {
                    return Err(anyhow::Error::from(nix::Error::last())
                        .context("Child: Failed to dup slave PTY to stdin using libc::dup2"));
                }
                if unsafe { libc::dup2(slave_raw_fd, libc::STDOUT_FILENO) } == -1 {
                    return Err(anyhow::Error::from(nix::Error::last())
                        .context("Child: Failed to dup slave PTY to stdout using libc::dup2"));
                }
                if unsafe { libc::dup2(slave_raw_fd, libc::STDERR_FILENO) } == -1 {
                    return Err(anyhow::Error::from(nix::Error::last())
                        .context("Child: Failed to dup slave PTY to stderr using libc::dup2"));
                }
                
                // After dup2, the original slave_fd (wrapping slave_raw_fd) should be closed,
                // as 0, 1, and 2 now refer to the same underlying file description.
                // Letting slave_fd (OwnedFd) go out of scope will achieve this.
                // No std::mem::forget is needed because openpty() will not return an FD in the 0,1,2 range.
                // So slave_raw_fd will be a distinct number (e.g. 5), and that FD 5 needs to be closed.
                // If slave_fd were to be forgotten, FD 5 would leak.
                drop(slave_fd);

                let command_cst = CString::new(config.command_executable)
                    .with_context(|| format!("Child: Failed to create CString for command: {}", config.command_executable))?;
                
                let mut args_cst_vec = Vec::new();
                args_cst_vec.push(CString::new(config.command_executable.split('/').last().unwrap_or(config.command_executable))
                    .with_context(|| "Child: Failed to create CString for command name (arg0)")?);
                for arg in config.args {
                    args_cst_vec.push(CString::new(*arg)
                        .with_context(|| format!("Child: Failed to create CString for argument: {}", arg))?);
                }
                
                log::debug!("Child: Executing command: {:?} with args {:?}", command_cst, args_cst_vec);
                let exec_err = execvp(&command_cst, &args_cst_vec).unwrap_err();
                eprintln!("Child: Failed to execute command '{:?}': {}", command_cst, exec_err);
                std::process::exit(1);
            }
        };

        Ok(NixPty { master_fd, child_pid })
    }

    pub fn spawn_shell_command(
        _shell_command_str: &str, _initial_cols: u16, _initial_rows: u16
    ) -> Result<Self> {
        unimplemented!("spawn_shell_command is not fully implemented with OwnedFd yet.");
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

    #[allow(dead_code)]
    pub fn terminate_child_process(&mut self) -> Result<()> {
        log::info!("Terminating child process {}", self.child_pid);
        kill(self.child_pid, Some(Signal::SIGKILL))
            .with_context(|| format!("Failed to send SIGKILL to child process {}", self.child_pid))
    }
}

impl Drop for NixPty {
    fn drop(&mut self) {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::debug!(
            "NixPty drop: Cleaning up PTY master_fd: {} (child_pid: {})",
            master_raw_fd, self.child_pid
        );
        // self.master_fd (OwnedFd) is dropped automatically, closing the FD.

        if self.child_pid.as_raw() <= 0 {
            log::debug!("NixPty drop: Invalid or no child PID ({}), skipping child process handling.", self.child_pid);
            return;
        }
        
        match waitpid(self.child_pid, Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::StillAlive) => {
                log::debug!("NixPty drop: Child process {} is still alive. Sending SIGHUP.", self.child_pid);
                if let Err(e) = kill(self.child_pid, Some(Signal::SIGHUP)) {
                    log::warn!("NixPty drop: Failed to send SIGHUP to child process {}: {}", self.child_pid, e);
                } else {
                    log::debug!("NixPty drop: Successfully sent SIGHUP to child process {}.", self.child_pid);
                }
            }
            Ok(status) => {
                log::debug!("NixPty drop: Child process {} already exited or changed state: {:?}", self.child_pid, status);
            }
            Err(e) => { // nix::Error
                if matches!(e, nix::Error::ECHILD) || matches!(e, nix::Error::ESRCH) {
                    log::debug!("NixPty drop: Child process {} does not exist or is not a child (waitpid error: {}). Already reaped?", self.child_pid, e);
                } else {
                    log::warn!("NixPty drop: Error checking child process {} status with waitpid: {}", self.child_pid, e);
                }
            }
        }
    }
}

impl Read for NixPty {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::trace!("NixPty::read attempting to read from fd {}", master_raw_fd);
        match nix::unistd::read(&self.master_fd, buf) { // Pass &OwnedFd
            Ok(bytes_read) => {
                log::trace!("NixPty::read successfully read {} bytes from fd {}", bytes_read, master_raw_fd);
                Ok(bytes_read)
            }
            Err(nix_err) => {
                if matches!(nix_err, nix::Error::EAGAIN) || matches!(nix_err, nix::Error::EWOULDBLOCK) {
                    log::debug!("NixPty::read on fd {}: Got {}, mapping to WouldBlock", master_raw_fd, nix_err);
                    Err(IoError::new(IoErrorKind::WouldBlock, nix_err))
                } else {
                    log::warn!("NixPty::read on fd {}: Got unhandled nix::Error {}, mapping to Other", master_raw_fd, nix_err);
                    Err(IoError::new(IoErrorKind::Other, nix_err))
                }
            }
        }
    }
}

impl Write for NixPty {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::trace!("NixPty::write attempting to write {} bytes to fd {}", buf.len(), master_raw_fd);
        match nix::unistd::write(&self.master_fd, buf) { // Pass &OwnedFd
            Ok(bytes_written) => {
                log::trace!("NixPty::write successfully wrote {} bytes to fd {}", bytes_written, master_raw_fd);
                Ok(bytes_written)
            }
            Err(nix_err) => {
                if matches!(nix_err, nix::Error::EAGAIN) || matches!(nix_err, nix::Error::EWOULDBLOCK) {
                    log::debug!("NixPty::write on fd {}: Got {}, mapping to WouldBlock", master_raw_fd, nix_err);
                    Err(IoError::new(IoErrorKind::WouldBlock, nix_err))
                } else {
                    log::warn!("NixPty::write on fd {}: Got unhandled nix::Error {}, mapping to Other", master_raw_fd, nix_err);
                    Err(IoError::new(IoErrorKind::Other, nix_err))
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
        Self::set_pty_size_internal(&self.master_fd, cols, rows)
            .with_context(|| format!("NixPty: PtyChannel::resize failed to set PTY size to {}x{} for fd {}", cols, rows, self.master_fd.as_raw_fd()))
    }

    fn child_pid(&self) -> Pid {
        self.child_pid
    }
}
