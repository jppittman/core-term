//! Main application entry point and PTY setup for myterm.

// Declare modules
mod glyph;
mod term;
mod backends;

use backends::{TerminalBackend, XBackend}; // Using X11 backend
use term::Term;

use anyhow::{Context, Result};
use std::ffi::CStr; // Removed CString
use std::fs::File;
// Removed std::io
use std::os::unix::io::{RawFd, FromRawFd};
use std::os::unix::process::CommandExt; // for Command::pre_exec
use std::process::{Command, Stdio};

// --- Constants ---
const DEFAULT_SHELL: &str = "/bin/sh";
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;
const TERM_ENV_VAR: &str = "TERM";
const TERM_VALUE: &str = "xterm-256color"; // Advertise 256 color support

/// Creates a new pseudo-terminal (PTY) and spawns a child process (shell).
///
/// Sets up the PTY master/slave pair, forks, and executes the specified shell
/// in the child process with its standard streams connected to the PTY slave.
///
/// # Arguments
/// * `shell_path` - Path to the shell executable.
/// * `cols` - Initial width of the terminal in columns.
/// * `rows` - Initial height of the terminal in rows.
///
/// # Returns
/// A tuple containing:
/// * `Ok((File, RawFd, i32))` - The `File` object owning the PTY master fd,
///   the raw PTY master fd (for passing to backend), and the child process ID (PID).
/// * `Err` - If any step in PTY creation or process spawning fails.
fn create_pty_and_spawn_shell(shell_path: &str, cols: u16, rows: u16) -> Result<(File, RawFd, i32)> {
    // Open the PTY master device
    let master_fd = unsafe { libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY) };
    if master_fd < 0 {
        return Err(std::io::Error::last_os_error()).context("Failed to open PTY master");
    }
    // Wrap fd in File for automatic closing on drop (RAII)
    let pty_master_file = unsafe { File::from_raw_fd(master_fd) };

    // Grant access to the slave PTY device
    if unsafe { libc::grantpt(master_fd) } != 0 {
        return Err(std::io::Error::last_os_error()).context("Failed to grant PTY slave access");
    }

    // Unlock the slave PTY device
    if unsafe { libc::unlockpt(master_fd) } != 0 {
        return Err(std::io::Error::last_os_error()).context("Failed to unlock PTY slave");
    }

    // Get the path to the corresponding slave PTY device
    let slave_name_ptr = unsafe { libc::ptsname(master_fd) };
    if slave_name_ptr.is_null() {
        return Err(std::io::Error::last_os_error()).context("Failed to get PTY slave name");
    }
    // Clone the name as CString because the pointer might become invalid
    let slave_name = unsafe { CStr::from_ptr(slave_name_ptr) }.to_owned();

    // --- Fork and Spawn Shell ---
    let mut command = Command::new(shell_path);
    command.env(TERM_ENV_VAR, TERM_VALUE);

    // Safety: This closure runs *after* fork but *before* exec in the child process.
    // It's critical to only call async-signal-safe functions here.
    // See `man 7 signal-safety` for details.
    unsafe {
        command.pre_exec(move || {
            // --- Child Process PTY Setup ---

            // Close the master PTY fd in the child; it only needs the slave.
            // The parent retains the master fd via pty_master_file.
            libc::close(master_fd);

            // Create a new session and become the session leader.
            if libc::setsid() < 0 {
                // Use perror or similar for logging in pre_exec if needed, but avoid complex logic.
                return Err(std::io::Error::last_os_error());
            }

            // Open the slave PTY device.
            let slave_fd = libc::open(slave_name.as_ptr(), libc::O_RDWR);
            if slave_fd < 0 { return Err(std::io::Error::last_os_error()); }

            // Set the slave PTY as the controlling terminal for the session.
            // TIOCSCTTY requires the caller to be the session leader and have no controlling terminal.
            if libc::ioctl(slave_fd, libc::TIOCSCTTY, 1) < 0 {
                 return Err(std::io::Error::last_os_error());
            }

            // Redirect child's stdin, stdout, stderr to the slave PTY.
            libc::dup2(slave_fd, libc::STDIN_FILENO);
            libc::dup2(slave_fd, libc::STDOUT_FILENO);
            libc::dup2(slave_fd, libc::STDERR_FILENO);

            // Close the extra slave_fd reference now that stdio points to it.
            libc::close(slave_fd);

            // Set the initial window size on the PTY slave.
            let winsz = libc::winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 };
            if libc::ioctl(libc::STDIN_FILENO, libc::TIOCSWINSZ, &winsz) < 0 {
                 // Non-fatal error if setting initial size fails.
                 // Use write for simple logging in async-signal-safe context if absolutely needed.
                 // eprintln!("Child: Warning: ioctl(TIOCSWINSZ) failed"); // eprintln might not be safe
                 let msg = b"Child: Warning: ioctl(TIOCSWINSZ) failed\n";
                 libc::write(libc::STDERR_FILENO, msg.as_ptr() as *const libc::c_void, msg.len());
            }

            Ok(()) // Indicate successful pre_exec setup
        });
    }

    // Configure stdio for the parent process side of the Command.
    // We don't want the parent to interact directly with the child's stdio pipes.
    command.stdin(Stdio::null());
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());

    // Spawn the child process (executes the shell).
    let child = command.spawn().context("Failed to spawn shell")?;
    let child_pid = child.id() as i32; // Cast PID to i32

    // Return the master file wrapper, the raw master fd, and the child PID.
    Ok((pty_master_file, master_fd, child_pid))
}

/// Sets up the terminal state, backend, and runs the main event loop.
fn run_terminal() -> Result<()> {
    // TODO: Add command line argument parsing (e.g., using clap) for shell, geometry, etc.
    let initial_cols = DEFAULT_COLS;
    let initial_rows = DEFAULT_ROWS;

    // Determine the shell to execute.
    let shell = std::env::var("SHELL").unwrap_or_else(|_| DEFAULT_SHELL.to_string());

    // Create the PTY and spawn the shell process.
    let (_pty_master_file, pty_fd, _child_pid) = // Keep File in scope for RAII cleanup
        create_pty_and_spawn_shell(&shell, initial_cols as u16, initial_rows as u16)
            .context("Failed to set up PTY")?;
    // Note: _pty_master_file owns the fd; it will be closed when it goes out of scope.
    // pty_fd is the raw descriptor passed to the backend.

    // Create the terminal emulator state.
    let mut term = Term::new(initial_cols, initial_rows);

    // Create the chosen UI backend.
    // Use a Box<dyn Trait> for potential future flexibility in choosing backends.
    let mut backend: Box<dyn TerminalBackend> = Box::new(XBackend::new(initial_cols, initial_rows)?);

    // --- Main Loop ---
    // The backend's `run` method is responsible for the event loop,
    // handling UI events and ideally multiplexing PTY I/O.
    println!("Starting backend run loop...");

    let exit_requested = backend.run(&mut term, pty_fd)
                                .context("Backend event loop failed")?;

    if exit_requested {
        println!("Exit requested by backend.");
    } else {
         // This case implies `run` returned Ok(false), which shouldn't happen
         // if it only exits upon request or error.
         println!("Backend loop exited unexpectedly.");
    }

    // --- Cleanup ---
    // Backend cleanup (closing display, freeing resources) should happen in its Drop impl.
    // PTY master file descriptor is closed automatically when `_pty_master_file` goes out of scope.
    // TODO: Add signal handling (SIGCHLD) to detect shell exit more reliably.
    // TODO: Add signal handling (SIGTERM, SIGINT) for graceful shutdown.

    println!("Terminal exiting.");
    Ok(())
}

/// Application entry point. Parses arguments and runs the terminal.
fn main() {
    env_logger::init();
    // Execute the main terminal logic.
    if let Err(e) = run_terminal() {
        // Print detailed error information using anyhow's multi-line format.
        eprintln!("Error: {:?}", e);
        std::process::exit(1); // Exit with a non-zero code to indicate failure.
    }
}

