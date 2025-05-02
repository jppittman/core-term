// Rust Terminal MVP (Proof of Concept)

// ** Declare modules **
mod glyph;
// ** FIX: Correct module declaration **
mod backends;
mod term;

// ** Use items from modules **
// ** FIX: Remove unused Glyph/Attributes here **
// use glyph::{Glyph, Attributes};
use backends::{TerminalBackend, ConsoleBackend, XBackend};
use term::Term;

// ** Keep necessary libc imports for main logic **
use libc::{
    // Basic types
    c_int, pid_t, /*termios, winsize,*/ // Removed unused termios types
    // PTY/Fork/Exec
    openpty, fork, execvp, setsid, ioctl, /*TIOCSWINSZ,*/ TIOCSCTTY, close, // Removed TIOCSWINSZ
    // Standard FDs
    STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO,
    // Termios control
    // tcsetattr, TCSAFLUSH, // Removed unused termios functions
    // Epoll
    epoll_create1, epoll_ctl, epoll_wait, epoll_event,
    EPOLLIN, EPOLLRDHUP, EPOLLERR, EPOLLHUP, // Removed EPOLL_CTL_ADD, EPOLL_CLOEXEC
    // fcntl for non-blocking
    fcntl, F_GETFL, F_SETFL, O_NONBLOCK,
    // For putenv in child
    putenv,
};
use std::env;
use std::ffi::{CString, CStr};
use std::path::Path;
use std::ptr;
use std::process;
use std::io::{self, Read};
// ** FIX: Add FromRawFd and AsRawFd, remove unused imports **
use std::os::unix::io::{FromRawFd, RawFd, AsRawFd};
use std::mem;
#[cfg(test)]
use std::fs::OpenOptions;
use anyhow::{Result, Context, Error as AnyhowError};


// --- Constants ---
const BUFSIZ: usize = 4096;
const MAX_EPOLL_EVENTS: c_int = 10;
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;
const DEFAULT_WIDTH_PX: usize = 640;
const DEFAULT_HEIGHT_PX: usize = 480;
const DEFAULT_FONT_NAME: &str = "fixed";
// ESC_ARG_SIZ is now internal to term.rs
const DEFAULT_SHELL: &str = "/bin/sh";

// --- Structs/Enums/Impls Moved to term.rs and glyph.rs ---


// Create PTY, fork, and exec the shell
// ** FIX: Restore implementation **
unsafe fn create_pty_and_fork(shell_path: &CStr, shell_args: &[*mut i8]) -> Result<(pid_t, std::fs::File)> {
    let mut pty_parent_fd: c_int = -1;
    let mut pty_child_fd: c_int = -1;
    let openpty_res = unsafe { openpty(&mut pty_parent_fd, &mut pty_child_fd, ptr::null_mut(), ptr::null_mut(), ptr::null_mut()) };
    if openpty_res < 0 { return Err(AnyhowError::from(io::Error::last_os_error()).context("openpty failed")); }
    let child_pid: pid_t = unsafe { fork() };
    if child_pid < 0 { unsafe { close(pty_parent_fd); close(pty_child_fd); } return Err(AnyhowError::from(io::Error::last_os_error()).context("fork failed")); }
    if child_pid != 0 { // Parent
        unsafe { close(pty_child_fd); }
        // ** FIX: Use FromRawFd trait **
        let pty_parent_file = unsafe { std::fs::File::from_raw_fd(pty_parent_fd) };
        return Ok((child_pid, pty_parent_file));
    }
    // Child process
    unsafe {
        close(pty_parent_fd);
        if setsid() < 0 { eprintln!("Child Error: setsid failed: {}", io::Error::last_os_error()); close(pty_child_fd); process::exit(1); }
        if libc::dup2(pty_child_fd, STDIN_FILENO) < 0 { eprintln!("Child Error: dup2 stdin failed: {}", io::Error::last_os_error()); close(pty_child_fd); process::exit(1); }
        if libc::dup2(pty_child_fd, STDOUT_FILENO) < 0 { eprintln!("Child Error: dup2 stdout failed: {}", io::Error::last_os_error()); close(pty_child_fd); process::exit(1); }
        if libc::dup2(pty_child_fd, STDERR_FILENO) < 0 { eprintln!("Child Error: dup2 stderr failed: {}", io::Error::last_os_error()); close(pty_child_fd); process::exit(1); }
        if ioctl(STDIN_FILENO, TIOCSCTTY, ptr::null_mut::<c_int>()) < 0 { eprintln!("Child Error: ioctl TIOCSCTTY failed: {}", io::Error::last_os_error()); if pty_child_fd > 2 { close(pty_child_fd); } process::exit(1); }
        if pty_child_fd > 2 { close(pty_child_fd); }
        let term_env_var = CString::new("TERM=xterm-256color").unwrap();
        putenv(term_env_var.into_raw() as *mut i8);
        execvp(shell_path.as_ptr(), shell_args.as_ptr() as *const *const i8);
        eprintln!("Child Error: execvp failed for '{}': {}", shell_path.to_string_lossy(), io::Error::last_os_error());
        process::exit(1);
    }
}

// Handles reading from the PTY parent and processing data.
// ** FIX: Restore implementation **
fn handle_pty_read(term: &mut Term, buf: &mut [u8]) -> Result<bool> {
    match term.pty_parent.read(buf) {
        Ok(0) => Ok(true), // PTY closed
        Ok(nread) => { term.process_pty_data(&buf[..nread])?; Ok(false) }
        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(false),
        Err(ref e) if e.kind() == io::ErrorKind::Interrupted => Ok(false),
        Err(e) => { eprintln!("Error reading from PTY: {} (kind: {:?})", e, e.kind()); Err(e.into()) }
    }
}

// Function to prepare shell command and arguments
// ** FIX: Restore implementation **
fn prepare_shell_command() -> Result<(CString, Vec<CString>)> {
    let shell_path_str = env::var("SHELL").unwrap_or_else(|_| DEFAULT_SHELL.to_string());
    let shell_path = CString::new(shell_path_str.clone())?;
    let shell_name = Path::new(&shell_path_str).file_name().and_then(|s| s.to_str()).unwrap_or("sh");
    let shell_arg0 = CString::new(shell_name)?;
    let shell_arg1 = CString::new("-l")?;
    Ok((shell_path, vec![shell_arg0, shell_arg1]))
}

// Main function implementing the epoll event loop
// ** FIX: Restore implementation **
fn main() -> Result<()> {
    let (shell_path, shell_args_c) = prepare_shell_command()?;
    let mut shell_args_ptr: Vec<*mut i8> = shell_args_c.iter().map(|cs| cs.as_ptr() as *mut i8).collect();
    shell_args_ptr.push(ptr::null_mut());

    // SAFETY: Main block contains many FFI calls.
    unsafe {
        let (_child_pid, pty_parent_file) = create_pty_and_fork(&shell_path, &shell_args_ptr)?;
        let pty_parent_fd = pty_parent_file.as_raw_fd();
        // Set non-blocking
        let flags = fcntl(pty_parent_fd, F_GETFL, 0);
        if flags < 0 || fcntl(pty_parent_fd, F_SETFL, flags | O_NONBLOCK) < 0 {
             eprintln!("Warning: Failed to set PTY master non-blocking: {}", io::Error::last_os_error());
              close(pty_parent_fd); return Err(io::Error::last_os_error().into());
        }

        // Use the desired backend from the backends module
        // let mut backend: Box<dyn TerminalBackend> = Box::new(XBackend::new());
        let mut backend: Box<dyn TerminalBackend> = Box::new(XBackend::new());

        backend.init().context("Backend init failed")?;
        let (initial_cols, initial_rows) = backend.get_dimensions();
        let mut term = Term::new(_child_pid, pty_parent_file, initial_cols, initial_rows);
        if let Err(e) = term.resize(initial_cols, initial_rows) { eprintln!("Warning: Failed to set initial PTY size: {}", e); }

        // Epoll setup
        let epoll_fd = epoll_create1(libc::EPOLL_CLOEXEC);
        if epoll_fd < 0 { return Err(AnyhowError::from(io::Error::last_os_error()).context("epoll_create1 failed")); }
        let mut pty_event = epoll_event { events: (EPOLLIN | EPOLLRDHUP) as u32, u64: pty_parent_fd as u64 };
        if epoll_ctl(epoll_fd, libc::EPOLL_CTL_ADD, pty_parent_fd, &mut pty_event) < 0 { close(epoll_fd); return Err(AnyhowError::from(io::Error::last_os_error()).context("epoll_ctl PTY failed")); }
        let backend_fds = backend.get_event_fds();
        for &fd in &backend_fds {
            let mut backend_event = epoll_event { events: (EPOLLIN | EPOLLRDHUP) as u32, u64: fd as u64 };
            if epoll_ctl(epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut backend_event) < 0 { close(epoll_fd); return Err(AnyhowError::from(io::Error::last_os_error()).context(format!("epoll_ctl backend fd {} failed", fd))); }
        }

        let mut pty_buf = vec![0u8; BUFSIZ];
        let mut events: Vec<epoll_event> = vec![mem::zeroed(); MAX_EPOLL_EVENTS as usize];
        println!("Terminal MVP running. Type commands or press Ctrl+D to exit.");
        backend.draw(&mut term)?; // Pass mutable backend

        // Event loop
        loop {
            let num_events = epoll_wait(epoll_fd, events.as_mut_ptr(), MAX_EPOLL_EVENTS, -1);
            if num_events < 0 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted { continue; }
                close(epoll_fd); return Err(AnyhowError::from(io::Error::last_os_error()).context("epoll_wait failed"));
            }
            let mut should_exit = false;
            for i in 0..num_events {
                let event = &events[i as usize];
                let event_fd = event.u64 as RawFd;
                let event_kind = event.events;
                 if event_kind & (EPOLLERR | EPOLLHUP | EPOLLRDHUP) as u32 != 0 {
                     if event_fd == pty_parent_fd { eprintln!("PTY hang-up/error (event 0x{:x}). Shell exited.", event_kind); should_exit = true; break; }
                     else if backend_fds.contains(&event_fd) { eprintln!("Backend hang-up/error on fd {} (event 0x{:x}).", event_fd, event_kind); should_exit = true; break; }
                     else { eprintln!("Error/hang-up on unexpected fd {} (event 0x{:x}).", event_fd, event_kind); should_exit = true; break; }
                 }
                 if event_kind & EPOLLIN as u32 != 0 {
                    if event_fd == pty_parent_fd {
                        match handle_pty_read(&mut term, &mut pty_buf) {
                            Ok(true) => { should_exit = true; break; } // PTY closed
                            Ok(false) => { if let Err(e) = backend.draw(&mut term) { eprintln!("Draw error: {}", e); } } // Pass mutable backend
                            Err(e) => { eprintln!("PTY read error: {}", e); should_exit = true; break; }
                        }
                    } else if backend_fds.contains(&event_fd) {
                         match backend.handle_event(&mut term, event_fd, event_kind) {
                             Ok(true) => { should_exit = true; break; } // Backend exit request
                             Ok(false) => {}
                             Err(e) => { eprintln!("Backend event error: {}", e); should_exit = true; break; }
                         }
                    }
                 }
            }
            if should_exit { break; }
        }
        close(epoll_fd);
        println!("Terminal MVP exiting.");
    } // End main unsafe block
    Ok(())
}
