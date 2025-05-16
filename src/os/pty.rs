// src/pty.rs (or src/os/pty.rs)
// STYLE GUIDE: Rustdoc for module explaining its purpose.
//! Handles pseudo-terminal (PTY) creation and interaction.
//! This module provides an abstraction layer over OS-specific PTY mechanisms,
//! using RAII for resource management.

use std::ffi::CString;
use std::io::{Read, Write, Result as IoResult};
use std::os::unix::io::{AsRawFd, RawFd};
use thiserror::Error; // For structured error handling

// STYLE GUIDE: Define Constants for magic numbers or common literals.
const DEFAULT_SHELL_EXECUTABLE: &str = "/bin/sh";
const SHELL_COMMAND_FLAG: &str = "-c";
const DEFAULT_TERM_ENV: &str = "myterm-256color"; // Standard TERM value for capable terminals

/// Defines errors specific to PTY operations.
// STYLE GUIDE: Rustdoc for public enums.
#[derive(Debug, Error)]
pub enum PtyError {
    /// Error during PTY device opening.
    #[error("Failed to open PTY device: {original_err}")]
    Open {
        #[source]
        original_err: nix::Error,
    },
    /// Error during child process spawning (fork).
    #[error("Failed to spawn child process: {original_err}")]
    Spawn {
        #[source]
        original_err: nix::Error,
    },
    /// Error during child process execution (execvp).
    #[error("Child process exec failed for command '{command}': {original_err}")]
    Exec {
        command: String,
        #[source]
        original_err: nix::Error,
    },
    /// Generic I/O error with the PTY.
    #[error("PTY I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Error from a `nix` system call not covered by other variants.
    #[error("Nix system call failed: {original_err}")]
    Nix {
        #[source]
        original_err: nix::Error,
    },
    /// Command string or arguments were invalid (e.g., empty, contained null bytes).
    #[error("Invalid shell command or argument: {reason}")]
    InvalidCommand { reason: String },
    /// Error related to managing the child process after it has been spawned.
    #[error("Child process management error: {0}")]
    ChildProcessManagement(String),
}

/// Configuration for spawning a PTY.
// STYLE GUIDE: Rustdoc for public structs. Group related arguments - OK.
#[derive(Debug, Clone)]
pub struct PtyConfig<'a> {
    /// The executable command to run in the PTY (e.g., "/bin/bash").
    pub command_executable: &'a str,
    /// Arguments for the command. Conventionally, `args[0]` is the command itself.
    /// Example: `args: &["/bin/bash", "-l"]` for a login shell.
    /// Example: `args: &["sh", "-c", "ls -la"]` where `command_executable` is "sh".
    pub args: &'a [&'a str],
    /// Initial number of columns for the PTY.
    pub initial_cols: u16,
    /// Initial number of rows for the PTY.
    pub initial_rows: u16,
    // pub working_directory: Option<&'a std::path::Path>, // Future extension
}

/// Defines the interface for interacting with a pseudo-terminal (PTY).
// STYLE GUIDE: Rustdoc for public traits.
pub trait PtyChannel: Read + Write + AsRawFd + Send + Sync {
    /// Resizes the PTY's window dimensions.
    fn resize(&self, cols: u16, rows: u16) -> Result<(), PtyError>;
    // fn child_pid(&self) -> nix::unistd::Pid; // Expose child PID if needed
}

/// A concrete PTY implementation for Linux/Unix-like systems using the `nix` crate.
/// Manages the PTY master file descriptor and the spawned child process.
/// Utilizes RAII for cleanup of the master FD and attempts graceful child termination.
// STYLE GUIDE: Rustdoc for public structs.
#[derive(Debug)]
pub struct NixPty {
    master_fd: RawFd,
    child_pid: nix::unistd::Pid,
}

/// RAII guard for the PTY "device" end file descriptor (formerly slave_fd).
/// Ensures it's closed if not explicitly disarmed (e.g., after child setup).
struct PtyDeviceEndGuard(RawFd);
impl PtyDeviceEndGuard {
    fn disarm(&mut self) {
        self.0 = -1; // Mark as disarmed; Drop will do nothing.
    }
}
impl Drop for PtyDeviceEndGuard {
    fn drop(&mut self) {
        if self.0 != -1 { // Only close if not disarmed
            log::trace!("PtyDeviceEndGuard: Closing PTY device end fd {} in drop", self.0);
            if let Err(e) = nix::unistd::close(self.0) {
                // Errors in drop are logged but not propagated.
                log::warn!("PtyDeviceEndGuard: Error closing PTY device end fd {} in drop: {}", self.0, e);
            }
        }
    }
}

impl NixPty {
    /// Spawns a new PTY running a shell command string (e.g., "ls -l" or "bash").
    /// The command string will be executed via "sh -c <command_string>".
    // STYLE GUIDE: Function argument count - OK.
    pub fn spawn_shell_command(
        shell_command_str: &str,
        initial_cols: u16,
        initial_rows: u16,
    ) -> Result<Self, PtyError> {
        // STYLE GUIDE: Use guard clauses for early exit.
        if shell_command_str.is_empty() {
            return Err(PtyError::InvalidCommand {
                reason: "Shell command string cannot be empty.".to_string(),
            });
        }
        let config = PtyConfig {
            command_executable: DEFAULT_SHELL_EXECUTABLE,
            // args[0] is the program name for execvp
            args: &[DEFAULT_SHELL_EXECUTABLE, SHELL_COMMAND_FLAG, shell_command_str],
            initial_cols,
            initial_rows,
        };
        Self::spawn_with_config(&config)
    }

    /// Spawns a new PTY with detailed configuration.
    // STYLE GUIDE: Function argument count (1 for config) - OK.
    pub fn spawn_with_config(config: &PtyConfig) -> Result<Self, PtyError> {
        use nix::fcntl::{fcntl, FcntlArg, OFlag};
        use nix::pty::openpty;
        use nix::unistd::{fork, ForkResult, Pid};

        // STYLE GUIDE: Guard clauses for invalid config before OS calls.
        if config.command_executable.is_empty() {
            return Err(PtyError::InvalidCommand {
                reason: "Command executable cannot be empty.".to_string(),
            });
        }
        if config.args.is_empty() {
            return Err(PtyError::InvalidCommand {
                reason: "Arguments list (args) cannot be empty (must include command name).".to_string(),
            });
        }
        // Typically, config.args[0] should match config.command_executable or be the name of the program being run.
        // This check is a bit nuanced depending on how `execvp` is being used (e.g. `sh -c "command"` vs `command arg1 arg2`)
        // For now, we assume `PtyConfig` is constructed correctly by the caller.

        let pty_pair = openpty(None, None).map_err(|e| PtyError::Open { original_err: e })?;
        let master_fd = pty_pair.master;
        let device_end_fd = pty_pair.slave; // Renamed from slave_fd for clarity
        log::debug!(
            "NixPty: Opened PTY master_fd={}, device_end_fd={}",
            master_fd, device_end_fd
        );

        // RAII guard for device_end_fd.
        let mut device_fd_guard = PtyDeviceEndGuard(device_end_fd);

        // Attempt to set initial size and non-blocking mode.
        // If these fail, master_fd won't be part of a NixPty struct yet (if fork fails later),
        // and device_fd_guard will close device_end_fd.
        Self::set_pty_size(device_end_fd, config.initial_cols, config.initial_rows)?;
        Self::set_fd_nonblocking(master_fd)?;

        match unsafe { fork() } {
            Ok(ForkResult::Parent { child, .. }) => {
                // Parent process successfully forked.
                // device_fd_guard's Drop will handle closing device_end_fd in the parent.
                log::info!("NixPty (Parent): Spawned child process with PID: {}", child);
                Ok(NixPty { master_fd, child_pid: child })
            }
            Ok(ForkResult::Child) => {
                // Child process.
                device_fd_guard.disarm(); // Child now owns device_end_fd.
                Self::child_process_setup_and_exec(master_fd, device_end_fd, config);
                // child_process_setup_and_exec calls execvp and should not return.
                unreachable!("execvp in child_process_setup_and_exec should not return.");
            }
            Err(e) => {
                // Fork failed. master_fd needs to be closed manually here as NixPty struct
                // (and its Drop impl) isn't created. device_fd_guard handles device_end_fd.
                log::error!("NixPty: Fork failed: {}", e);
                let _ = nix::unistd::close(master_fd); // Attempt cleanup
                Err(PtyError::Spawn{ original_err: e })
            }
        }
    }

    /// Sets the terminal window size for the given PTY file descriptor.
    fn set_pty_size(fd: RawFd, cols: u16, rows: u16) -> Result<(), PtyError> {
        use nix::pty::Winsize;
        nix::ioctl_write_ptr_bad!(tcsetwinsize, nix::libc::TIOCSWINSZ, Winsize);
        let winsize = Winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 };
        unsafe { tcsetwinsize(fd, &winsize) }.map_err(|e| PtyError::Nix { original_err: e })?;
        log::trace!("NixPty: Set PTY fd {} size to {}x{}", fd, cols, rows);
        Ok(())
    }

    /// Sets the given file descriptor to non-blocking mode.
    fn set_fd_nonblocking(fd: RawFd) -> Result<(), PtyError> {
        use nix::fcntl::{fcntl, FcntlArg, OFlag};
        let flags = fcntl(fd, FcntlArg::F_GETFL).map_err(|e| PtyError::Nix { original_err: e })?;
        let nonblock_flags = OFlag::from_bits_truncate(flags) | OFlag::O_NONBLOCK;
        fcntl(fd, FcntlArg::F_SETFL(nonblock_flags)).map_err(|e| PtyError::Nix { original_err: e })?;
        log::trace!("NixPty: Set fd {} to non-blocking", fd);
        Ok(())
    }
    
    /// Prepares and executes the command in the child process. Does not return on success.
    /// Exits child process on unrecoverable errors.
    fn child_process_setup_and_exec(master_fd_to_close: RawFd, device_end_fd: RawFd, config: &PtyConfig) {
        use nix::unistd::{setsid, dup2, execvp};
        
        // Helper for critical child operations. On error, prints to stderr and exits child.
        // STYLE GUIDE: Avoid deep nesting by handling errors immediately.
        fn child_critical_op<T, F: FnOnce() -> Result<T, nix::Error>>(op: F, err_msg_prefix: &str) -> T {
            match op() {
                Ok(val) => val,
                Err(e) => {
                    eprintln!("myterm (child): {} - Error: {}. Exiting child.", err_msg_prefix, e);
                    std.process::exit(1);
                }
            }
        }

        child_critical_op(|| nix::unistd::close(master_fd_to_close), "Failed to close master PTY fd in child");
        child_critical_op(setsid, "setsid failed in child");

        nix::ioctl_write_int_bad!(tiocsctty, nix::libc::TIOCSCTTY);
        child_critical_op(|| unsafe { tiocsctty(device_end_fd, 0) }, "TIOCSCTTY failed for PTY device end");

        child_critical_op(|| dup2(device_end_fd, nix::libc::STDIN_FILENO).map(|_| ()), "dup2 to STDIN failed");
        child_critical_op(|| dup2(device_end_fd, nix::libc::STDOUT_FILENO).map(|_| ()), "dup2 to STDOUT failed");
        child_critical_op(|| dup2(device_end_fd, nix::libc::STDERR_FILENO).map(|_| ()), "dup2 to STDERR failed");

        if device_end_fd > nix::libc::STDERR_FILENO { // Avoid closing stdio if device_end_fd was 0,1,2
            child_critical_op(|| nix::unistd::close(device_end_fd), "Failed to close original PTY device end fd in child");
        }

        if std::env::var_os("TERM").is_none() {
            std.env::set_var("TERM", DEFAULT_TERM_ENV);
        }

        let c_command_executable = CString::new(config.command_executable).unwrap_or_else(|e| {
            eprintln!("myterm (child): Command executable ('{}') contains null byte: {}. Exiting child.", config.command_executable, e);
            std::process::exit(126); // Invalid argument
        });
        
        let c_args: Vec<CString> = config.args.iter()
            .map(|s| CString::new(*s).unwrap_or_else(|e| {
                eprintln!("myterm (child): Argument ('{}') contains null byte: {}. Exiting child.", s, e);
                std.process::exit(126); // Invalid argument
            }))
            .collect();

        log::debug!("NixPty (Child): Executing command: {:?} with args {:?}", config.command_executable, config.args);
        
        // execvp replaces the current process. It only returns if an error occurs.
        // The error from execvp is reported via the panic hook if not handled by exit.
        let exec_result = execvp(&c_command_executable, &c_args);
        
        // If execvp returns, it's always an error.
        let nix_error = exec_result.err().expect("execvp returned Ok, which is impossible; should only return on error");
        eprintln!(
            "myterm (child): execvp failed for command '{} {:?}': {}. Exiting child.",
            config.command_executable, config.args, nix_error
        );
        std::process::exit(127); // Standard exit code for "command not found" or exec failure.
    }

    /// Attempts to gracefully terminate the child process.
    pub fn terminate_child_process(&mut self) -> Result<(), PtyError> {
        // STYLE GUIDE: Early return for already terminated/invalid PID.
        if self.child_pid.as_raw() <= 0 {
            log::trace!("NixPty: Child process (PID {}) not active or already handled.", self.child_pid);
            return Ok(());
        }

        log::debug!("NixPty: Attempting to terminate child process PID: {}", self.child_pid);
        use nix::sys::signal::{kill, Signal};
        use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};

        let signals_to_try = [Signal::SIGHUP, Signal::SIGTERM, Signal::SIGKILL];
        let mut last_signal_error: Option<nix::Error> = None;

        for (index, &signal) in signals_to_try.iter().enumerate() {
            log::trace!("NixPty: Sending {} to child PID {}", signal, self.child_pid);
            
            // Attempt to send the signal
            match kill(self.child_pid, Some(signal)) {
                Ok(_) => { /* Signal sent successfully */ }
                Err(nix::Error::ESRCH) => { // ESRCH: No such process
                    log::info!("NixPty: Child process {} already exited before {}.", self.child_pid, signal);
                    self.child_pid = Pid::from_raw(0); // Mark as gone
                    return Ok(());
                }
                Err(e) => { // Other error sending signal
                    log::warn!("NixPty: Error sending {} to child PID {}: {}", signal, self.child_pid, e);
                    last_signal_error = Some(e);
                    // If a less severe signal fails (not ESRCH), maybe don't escalate to SIGKILL immediately,
                    // but for this loop structure, we'll record and continue to check status or try next.
                    // If SIGKILL itself fails to send, that's critical.
                    if signal == Signal::SIGKILL {
                        return Err(PtyError::Nix { original_err: e });
                    }
                    continue; // Try next signal only if sending failed for non-ESRCH, non-SIGKILL.
                              // Actually, if send fails (and not ESRCH), we should probably try waitpid anyway.
                }
            }

            // Wait briefly for the process to react, especially for SIGHUP/SIGTERM.
            let wait_duration = if signal == Signal::SIGKILL {
                std::time::Duration::from_millis(100) // Longer for SIGKILL
            } else {
                std::time::Duration::from_millis(20)  // Shorter for graceful signals
            };
            std::thread::sleep(wait_duration);

            // Check status non-blockingly
            match waitpid(self.child_pid, Some(WaitPidFlag::WNOHANG)) {
                Ok(WaitStatus::Exited(_, _) | WaitStatus::Signaled(_, _, _)) => {
                    log::info!("NixPty: Child process {} successfully terminated/reaped after {}.", self.child_pid, signal);
                    self.child_pid = Pid::from_raw(0); // Mark as gone
                    return Ok(());
                }
                Ok(WaitStatus::StillAlive) | Ok(WaitStatus::Stopped(_, _)) => {
                    log::trace!("NixPty: Child {} still present after {} and wait.", self.child_pid, signal);
                    // Continue to the next signal if this wasn't the last one.
                    if index == signals_to_try.len() - 1 {
                        log::warn!("NixPty: Child {} still alive after all termination attempts.", self.child_pid);
                    }
                }
                Err(nix::Error::ECHILD) => { // ECHILD: No child processes, or already reaped
                    log::info!("NixPty: Child process {} reaped by other means or does not exist (ECHILD after {}).", self.child_pid, signal);
                    self.child_pid = Pid::from_raw(0); // Mark as gone
                    return Ok(());
                }
                Err(e) => {
                    log::warn!("NixPty: waitpid error for child {} after {}: {}", self.child_pid, signal, e);
                    last_signal_error = Some(e);
                    // If waitpid itself fails badly, it's hard to know the state.
                    // We might break and return an error. For now, let it try the next signal if any.
                }
            }
        }
        
        // If loop completes and child_pid is still considered active
        if self.child_pid.as_raw() > 0 {
            let final_err_msg = format!("Child process {} could not be confirmed as terminated after all attempts.", self.child_pid);
            log::error!("{}", final_err_msg);
            // Return the last significant error encountered, or a generic ChildProcessManagement error.
            if let Some(nix_err) = last_signal_error {
                return Err(PtyError::Nix { original_err: nix_err });
            } else {
                return Err(PtyError::ChildProcessManagement(final_err_msg));
            }
        }
        Ok(()) // Marked as gone during the loop
    }
}

impl Drop for NixPty {
    fn drop(&mut self) {
        log::debug!("NixPty: Drop called for master_fd: {}, child_pid: {}", self.master_fd, self.child_pid);
        
        // Best-effort attempt to terminate the child process if it's still considered active.
        // Errors during this phase in Drop are logged but not propagated.
        if self.child_pid.as_raw() > 0 {
            if let Err(e) = self.terminate_child_process() {
                 log::warn!("NixPty (Drop): Error during terminate_child_process (ignoring in drop): {}", e);
            }
        }

        // Close the master file descriptor.
        if self.master_fd != -1 { // Check if fd is valid (e.g., not already closed)
            if let Err(e) = nix::unistd::close(self.master_fd) {
                log::error!("NixPty (Drop): Error closing master_fd {}: {}", self.master_fd, e);
            } else {
                log::trace!("NixPty (Drop): Successfully closed master_fd {}", self.master_fd);
            }
            // self.master_fd = -1; // Optionally mark as closed to prevent double-close attempts
                                 // if drop could somehow be called more than once (not typical for owned values).
        }
    }
}

// Implement Read, Write, AsRawFd for NixPty
impl Read for NixPty {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        match nix::unistd::read(self.master_fd, buf) {
            Ok(count) => Ok(count),
            Err(nix::errno::Errno::EAGAIN) | Err(nix::errno::Errno::EWOULDBLOCK) => {
                Err(std::io::Error::new(std::io::ErrorKind::WouldBlock, "PTY read would block"))
            }
            Err(e) => {
                log::error!("NixPty: Read error on master_fd {}: {}", self.master_fd, e);
                Err(std::io::Error::new(std::io::ErrorKind::Other, e))
            },
        }
    }
}

impl Write for NixPty {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        match nix::unistd::write(self.master_fd, buf) {
            Ok(count) => Ok(count),
            Err(e) => {
                 log::error!("NixPty: Write error on master_fd {}: {}", self.master_fd, e);
                Err(std::io::Error::new(std::io::ErrorKind::Other, e))
            },
        }
    }
    fn flush(&mut self) -> IoResult<()> { Ok(()) }
}

impl AsRawFd for NixPty {
    fn as_raw_fd(&self) -> RawFd { self.master_fd }
}

impl PtyChannel for NixPty {
    fn resize(&self, cols: u16, rows: u16) -> Result<(), PtyError> {
        Self::set_pty_size(self.master_fd, cols, rows)
    }
    // fn child_pid(&self) -> nix::unistd::Pid { self.child_pid }
}
