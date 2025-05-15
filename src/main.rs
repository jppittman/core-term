//! Main application entry point and PTY setup for myterm.

mod glyph;
mod term;
mod backends;
mod ansi;

use backends::{TerminalBackend, XBackend};
use term::TerminalEmulator;
// LocaleController is now managed by a lazy static in term::unicode,
// so main.rs doesn't need to create or pass it directly.
// The initialization is handled by the first call to `get_char_display_width`.

use anyhow::{Context, Result};
use std::ffi::CStr;
use std::fs::File;
use std::os::unix::io::{RawFd, FromRawFd};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};

const DEFAULT_SHELL: &str = "/bin/sh";
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;
const DEFAULT_SCROLLBACK_LIMIT: usize = 1000;
const TERM_ENV_VAR: &str = "TERM";
const TERM_VALUE: &str = "xterm-256color";

fn create_pty_and_spawn_shell(shell_path: &str, cols: u16, rows: u16) -> Result<(File, RawFd, i32)> {
    // ... (function content remains the same) ...
    let master_fd = unsafe { libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY) };
    if master_fd < 0 { return Err(std::io::Error::last_os_error()).context("Failed to open PTY master"); }
    let pty_master_file = unsafe { File::from_raw_fd(master_fd) };
    if unsafe { libc::grantpt(master_fd) } != 0 { return Err(std::io::Error::last_os_error()).context("Failed to grant PTY slave access"); }
    if unsafe { libc::unlockpt(master_fd) } != 0 { return Err(std::io::Error::last_os_error()).context("Failed to unlock PTY slave"); }
    let slave_name_ptr = unsafe { libc::ptsname(master_fd) };
    if slave_name_ptr.is_null() { return Err(std::io::Error::last_os_error()).context("Failed to get PTY slave name"); }
    let slave_name = unsafe { CStr::from_ptr(slave_name_ptr) }.to_owned();
    let mut command = Command::new(shell_path);
    command.env(TERM_ENV_VAR, TERM_VALUE);
    unsafe {
        command.pre_exec(move || {
            libc::close(master_fd);
            if libc::setsid() < 0 { return Err(std::io::Error::last_os_error()); }
            let slave_fd = libc::open(slave_name.as_ptr(), libc::O_RDWR);
            if slave_fd < 0 { return Err(std::io::Error::last_os_error()); }
            if libc::ioctl(slave_fd, libc::TIOCSCTTY, 1) < 0 { return Err(std::io::Error::last_os_error()); }
            libc::dup2(slave_fd, libc::STDIN_FILENO);
            libc::dup2(slave_fd, libc::STDOUT_FILENO);
            libc::dup2(slave_fd, libc::STDERR_FILENO);
            if slave_fd > 2 { libc::close(slave_fd); }
            let winsz = libc::winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 };
            if libc::ioctl(libc::STDIN_FILENO, libc::TIOCSWINSZ, &winsz) < 0 {
                 let msg = b"Child: Warning: ioctl(TIOCSWINSZ) failed\n";
                 libc::write(libc::STDERR_FILENO, msg.as_ptr() as *const libc::c_void, msg.len());
            }
            Ok(())
        });
    }
    command.stdin(Stdio::null()); command.stdout(Stdio::null()); command.stderr(Stdio::null());
    let child = command.spawn().context("Failed to spawn shell")?;
    let child_pid = child.id() as i32;
    Ok((pty_master_file, master_fd, child_pid))
}

// run_terminal no longer needs to accept LocaleController
fn run_terminal() -> Result<()> {
    let initial_cols = DEFAULT_COLS;
    let initial_rows = DEFAULT_ROWS;
    let scrollback_limit = DEFAULT_SCROLLBACK_LIMIT;

    let shell = std::env::var("SHELL").unwrap_or_else(|_| DEFAULT_SHELL.to_string());

    let (_pty_master_file, pty_fd, _child_pid) =
        create_pty_and_spawn_shell(&shell, initial_cols as u16, initial_rows as u16)
            .context("Failed to set up PTY")?;

    let mut term_emulator = TerminalEmulator::new(initial_cols, initial_rows, scrollback_limit);
    let mut backend: Box<dyn TerminalBackend> = Box::new(XBackend::new(initial_cols, initial_rows)?);

    println!("Starting backend run loop...");
    // The backend's `run` method will call `term_emulator.interpret_input`,
    // which in turn will call `unicode::get_char_display_width`.
    // The lazy static `GLOBAL_LOCALE_CONTROLLER` in `unicode.rs` will handle
    // the one-time initialization of the locale when `get_char_display_width` is first called.
    let exit_requested = backend.run(&mut term_emulator, pty_fd)
                                .context("Backend event loop failed")?;

    if exit_requested {
        println!("Exit requested by backend.");
    } else {
         println!("Backend loop exited unexpectedly.");
    }

    println!("Terminal exiting.");
    Ok(())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Locale initialization is now handled lazily by the static LocaleController
    // when get_char_display_width is first called. No explicit call needed here.
    // The `LocaleController::new()` inside the `Lazy::new(...)` in unicode.rs
    // will perform the `setlocale` call.

    if let Err(e) = run_terminal() {
        eprintln!("Error: {:?}", e);
        std::process::exit(1);
    }
}

