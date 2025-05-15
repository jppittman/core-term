//! Main application entry point and orchestrator for myterm.
//!
//! This module is responsible for:
//! - Initializing all core components: TerminalEmulator, AnsiProcessor, Driver, Renderer.
//! - Setting up the PTY and spawning the shell.
//! - Running the main event loop (using epoll) to handle:
//!   - PTY output from the shell.
//!   - Platform events from the Driver (input, resize, etc.).
//! - Processing PTY data and Driver events, feeding them to the TerminalEmulator.
//! - Executing EmulatorActions returned by the TerminalEmulator.
//! - Calling the Renderer to draw the terminal state.
//! - Handling application shutdown.

// Declare modules
mod ansi;
mod backends;
mod glyph; // Used by TerminalEmulator, Renderer, Driver
mod renderer;
mod term; // New renderer module

// Crate-level imports
use crate::{
    ansi::AnsiProcessor,
    backends::{BackendEvent, Driver, x11::XDriver}, // Assuming XDriver will be refactored
    renderer::Renderer,
    term::{EmulatorAction, EmulatorInput, TerminalEmulator, TerminalInterface},
};

use log::trace;

// Standard library and OS-specific imports
use anyhow::{Context, Result};
use std::{
    env,
    ffi::CStr,
    fs::File,
    io::Read, // Added Write for PTY
    os::unix::{
        io::{FromRawFd, RawFd},
        process::CommandExt,
    },
    process::{Command, Stdio},
};

// Libc imports for PTY, epoll, and other low-level operations
use libc::{
    self,
    EPOLL_CTL_ADD,
    EPOLL_CTL_DEL,
    EPOLLIN,
    F_GETFL,
    F_SETFL,    // For setting non-blocking mode
    O_NONBLOCK, // Added O_NONBLOCK
    TIOCSWINSZ, // For PTY size
    c_int,
    c_void,
    epoll_create1,
    epoll_ctl,
    epoll_event,
    epoll_wait,
    winsize,
};

// Logging
use log::{debug, error, info, warn};

// --- Constants ---
const DEFAULT_SHELL: &str = "/bin/sh";
const DEFAULT_SCROLLBACK_LIMIT: usize = 1000;
const TERM_ENV_VAR: &str = "TERM";
const TERM_VALUE: &str = "xterm-256color"; // Common term value for good compatibility

const PTY_READ_BUF_SIZE: usize = 4096;
const MAX_EPOLL_EVENTS: usize = 2; // PTY and Driver
const EPOLL_TIMEOUT_MS: c_int = -1; // Block indefinitely

// Event loop FDs for epoll
const PTY_FD_EVENT_ID: u64 = 1;
const DRIVER_FD_EVENT_ID: u64 = 2;

/// Creates a pseudo-terminal (PTY) and spawns the specified shell within it.
///
/// Sets the initial terminal dimensions for the PTY.
///
/// # Arguments
/// * `shell_path`: Path to the shell executable.
/// * `cols`: Initial number of columns for the terminal.
/// * `rows`: Initial number of rows for the terminal.
///
/// # Returns
/// A tuple containing the PTY master file, its raw file descriptor, and the child shell's PID.
fn create_pty_and_spawn_shell(
    shell_path: &str,
    cols: u16,
    rows: u16,
    initial_width_px: u16,  // Added for TIOCSWINSZ
    initial_height_px: u16, // Added for TIOCSWINSZ
) -> Result<(File, RawFd, i32)> {
    info!(
        "Creating PTY and spawning shell '{}' with size {}x{} ({}x{}px)",
        shell_path, cols, rows, initial_width_px, initial_height_px
    );
    // Open PTY master
    let master_fd = unsafe { libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY) };
    if master_fd < 0 {
        return Err(std::io::Error::last_os_error())
            .context("Failed to open PTY master (posix_openpt)");
    }

    // Grant access to PTY slave
    if unsafe { libc::grantpt(master_fd) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(master_fd) };
        return Err(err).context("Failed to grant PTY slave access (grantpt)");
    }

    // Unlock PTY slave
    if unsafe { libc::unlockpt(master_fd) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(master_fd) };
        return Err(err).context("Failed to unlock PTY slave (unlockpt)");
    }

    // Get PTY slave name
    let slave_name_ptr = unsafe { libc::ptsname(master_fd) };
    if slave_name_ptr.is_null() {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(master_fd) };
        return Err(err).context("Failed to get PTY slave name (ptsname)");
    }
    let slave_name = unsafe { CStr::from_ptr(slave_name_ptr) }.to_owned();
    debug!("PTY slave device: {:?}", slave_name);

    // Fork and spawn shell
    let mut command = Command::new(shell_path);
    command.env(TERM_ENV_VAR, TERM_VALUE); // Set TERM environment variable

    // Safety: pre_exec is inherently unsafe as it runs in the child process after fork
    // but before exec. It must only call async-signal-safe functions.
    unsafe {
        command.pre_exec(move || {
            // In child process
            // Close master PTY fd (child doesn't need it)
            libc::close(master_fd);

            // Create new session and set controlling terminal
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error()); // Error in child
            }

            // Open PTY slave
            // slave_name needs to be CString for libc::open
            let slave_fd = libc::open(slave_name.as_ptr() as *const libc::c_char, libc::O_RDWR);
            if slave_fd < 0 {
                return Err(std::io::Error::last_os_error());
            }

            // Set PTY slave as controlling terminal
            if libc::ioctl(slave_fd, libc::TIOCSCTTY, 1) < 0 {
                return Err(std::io::Error::last_os_error());
            }

            // Duplicate slave fd to stdin, stdout, stderr
            libc::dup2(slave_fd, libc::STDIN_FILENO);
            libc::dup2(slave_fd, libc::STDOUT_FILENO);
            libc::dup2(slave_fd, libc::STDERR_FILENO);

            // Close original slave_fd if it's not one of 0,1,2
            if slave_fd > libc::STDERR_FILENO {
                libc::close(slave_fd);
            }

            // Set initial terminal window size for the PTY slave
            let winsz = libc::winsize {
                ws_row: rows,
                ws_col: cols,
                ws_xpixel: initial_width_px,
                ws_ypixel: initial_height_px,
            };
            if libc::ioctl(libc::STDIN_FILENO, libc::TIOCSWINSZ, &winsz) < 0 {
                // Non-fatal in child, but log to its stderr (which is the PTY)
                let msg = b"myterm child: Warning: ioctl(TIOCSWINSZ) failed in pre_exec\n";
                libc::write(
                    libc::STDERR_FILENO,
                    msg.as_ptr() as *const libc::c_void,
                    msg.len(),
                );
            }
            Ok(())
        });
    }

    // Configure stdio for the parent process's view of the child
    command.stdin(Stdio::null());
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());

    let child = command.spawn().context("Failed to spawn shell")?;
    let child_pid = child.id() as i32;
    info!("Shell spawned with PID: {}", child_pid);

    // Convert master_fd to File for easier handling (e.g., non-blocking reads)
    let pty_master_file = unsafe { File::from_raw_fd(master_fd) };

    // Set PTY master to non-blocking
    // Safety: FFI call
    let current_flags = unsafe { libc::fcntl(master_fd, F_GETFL, 0) };
    if current_flags < 0 {
        return Err(std::io::Error::last_os_error())
            .context("Failed to get PTY master flags (fcntl F_GETFL)");
    }
    // Safety: FFI call
    if unsafe { libc::fcntl(master_fd, F_SETFL, current_flags | O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error())
            .context("Failed to set PTY master to non-blocking (fcntl F_SETFL)");
    }
    debug!("PTY master fd {} set to non-blocking", master_fd);

    Ok((pty_master_file, master_fd, child_pid))
}

/// Selects and initializes the terminal driver.
///
/// TODO: Implement actual driver selection (e.g., based on args or config).
/// For now, defaults to XDriver.
fn select_driver() -> Result<Box<dyn Driver>> {
    info!("Selecting terminal driver (defaulting to XDriver)");
    // Assuming XDriver::new() is refactored to implement Driver and takes no args.
    // And that XDriver itself will determine initial pixel/font dimensions.
    let driver = XDriver::new().context("Failed to initialize XDriver")?;
    Ok(Box::new(driver))
}

/// Handles an EmulatorAction returned by the TerminalEmulator.
fn handle_emulator_action(
    action: EmulatorAction,
    pty_fd: RawFd,
    _driver: &mut dyn Driver, // Pass driver if actions need direct driver interaction
    _term_emulator: &mut TerminalEmulator, // Pass term if actions need to query it
) -> Result<()> {
    trace!("Handling EmulatorAction: {:?}", action);
    match action {
        EmulatorAction::WritePty(bytes) => {
            debug!("Writing {} bytes to PTY: {:?}", bytes.len(), bytes);
            // Safety: FFI call to write to PTY
            let mut total_written = 0;
            while total_written < bytes.len() {
                let written = unsafe {
                    libc::write(
                        pty_fd,
                        bytes.as_ptr().add(total_written) as *const c_void,
                        bytes.len() - total_written,
                    )
                };
                if written < 0 {
                    let err = std::io::Error::last_os_error();
                    // Handle EINTR and EAGAIN/EWOULDBLOCK if PTY were non-blocking for write
                    if err.kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(err).context("Failed to write to PTY");
                }
                total_written += written as usize;
            }
        }
        EmulatorAction::SetTitle(title) => {
            info!("EmulatorAction::SetTitle: {}", title);
            // TODO: Implement title setting. This might require a method on the Driver trait
            // if it's a common capability, or be handled by specific driver features.
            // For now, we log it. driver.set_title(&title)?;
        }
        EmulatorAction::RingBell => {
            info!("EmulatorAction::RingBell");
            // TODO: Implement bell. driver.ring_bell()?;
        }
        EmulatorAction::RequestRedraw => {
            // This action is a signal; the main loop sets needs_render based on it.
            trace!("EmulatorAction::RequestRedraw received (handled by main loop flag)");
        }
        EmulatorAction::SetCursorVisibility(visible) => {
            info!("EmulatorAction::SetCursorVisibility: {}", visible);
            // TODO: Implement cursor visibility. driver.set_cursor_visibility(visible)?;
        }
    }
    Ok(())
}

/// Handles a BackendEvent received from the Driver.
#[allow(clippy::too_many_arguments)] // This function coordinates many parts, arguments are justified
fn handle_backend_event(
    event: BackendEvent,
    term_emulator: &mut TerminalEmulator,
    driver: &mut dyn Driver, // Use `&mut dyn Driver`
    pty_fd: RawFd,
    needs_exit: &mut bool,
    needs_render: &mut bool,
) -> Result<()> {
    trace!("Handling BackendEvent: {:?}", event);
    match event {
        BackendEvent::Key { keysym, text } => {
            let input = EmulatorInput::User(BackendEvent::Key { keysym, text });
            if let Some(action) = term_emulator.interpret_input(input) {
                // Pass driver and term_emulator by reference
                handle_emulator_action(action, pty_fd, driver, term_emulator)?;
            }
            // Key events often change state that needs rendering (e.g., echo, cursor move).
            *needs_render = true;
        }
        BackendEvent::Resize {
            width_px,
            height_px,
        } => {
            info!(
                "BackendEvent::Resize: new pixel dimensions {}x{}",
                width_px, height_px
            );
            let (font_w, font_h) = driver.get_font_dimensions();
            if font_w == 0 || font_h == 0 {
                warn!(
                    "Driver returned zero font dimensions, cannot resize terminal character grid."
                );
                return Ok(());
            }

            let new_cols = (width_px as usize / font_w).max(1);
            let new_rows = (height_px as usize / font_h).max(1);
            info!(
                "Calculated new terminal dimensions: {}x{} cells",
                new_cols, new_rows
            );

            term_emulator.resize(new_cols, new_rows);

            // Update PTY window size
            let winsz = winsize {
                ws_row: new_rows as u16,
                ws_col: new_cols as u16,
                ws_xpixel: width_px,
                ws_ypixel: height_px,
            };
            // Safety: FFI call to ioctl
            if unsafe { libc::ioctl(pty_fd, TIOCSWINSZ, &winsz) } < 0 {
                warn!(
                    "ioctl(TIOCSWINSZ) failed for PTY fd {}: {}",
                    pty_fd,
                    std::io::Error::last_os_error()
                );
            } else {
                debug!("ioctl(TIOCSWINSZ) successful for PTY fd {}", pty_fd);
            }
            *needs_render = true;
        }
        BackendEvent::CloseRequested => {
            info!("BackendEvent::CloseRequested received.");
            *needs_exit = true;
        }
        BackendEvent::FocusGained => {
            info!("BackendEvent::FocusGained");
            // According to north_star.md, focus changes might require redraw.
            *needs_render = true;
        }
        BackendEvent::FocusLost => {
            info!("BackendEvent::FocusLost");
            *needs_render = true;
        }
    }
    Ok(())
}

/// Main terminal orchestration logic.
fn run_terminal_orchestrator() -> Result<()> {
    info!("Starting myterm orchestrator...");

    // 1. Initialize Driver
    // The driver is expected to initialize the display system and determine
    // initial font and pixel dimensions.
    let mut driver = select_driver().context("Failed to select/initialize terminal driver")?;

    // 2. Get initial dimensions from Driver
    let (font_width, font_height) = driver.get_font_dimensions();
    if font_width == 0 || font_height == 0 {
        anyhow::bail!(
            "Driver returned invalid font dimensions (width: {}, height: {}). Cannot proceed.",
            font_width,
            font_height
        );
    }
    let (display_width_px, display_height_px) = driver.get_display_dimensions_pixels();
    if display_width_px == 0 || display_height_px == 0 {
        anyhow::bail!(
            "Driver returned invalid display pixel dimensions (width: {}, height: {}). Cannot proceed.",
            display_width_px,
            display_height_px
        );
    }

    let initial_cols = (display_width_px as usize / font_width).max(1);
    let initial_rows = (display_height_px as usize / font_height).max(1);
    info!(
        "Initial derived terminal size: {}x{} cells (Font: {}x{}, Display: {}x{}px)",
        initial_cols, initial_rows, font_width, font_height, display_width_px, display_height_px
    );

    // 3. Setup PTY and Spawn Shell
    let shell = env::var("SHELL").unwrap_or_else(|_| DEFAULT_SHELL.to_string());
    let (mut pty_master_file, pty_fd, child_pid) = create_pty_and_spawn_shell(
        &shell,
        initial_cols as u16,
        initial_rows as u16,
        display_width_px,
        display_height_px,
    )
    .context("Failed to set up PTY and spawn shell")?;
    debug!("PTY master fd: {}", pty_fd);

    // 4. Initialize other core components
    let mut term_emulator =
        TerminalEmulator::new(initial_cols, initial_rows, DEFAULT_SCROLLBACK_LIMIT);
    let mut ansi_processor = AnsiProcessor::new();
    let mut renderer = Renderer::new(); // Renderer is stateless for now

    // 5. Setup epoll for event handling
    // Safety: FFI call
    let epoll_fd = unsafe { epoll_create1(libc::EPOLL_CLOEXEC) };
    if epoll_fd < 0 {
        return Err(std::io::Error::last_os_error()).context("Failed to create epoll instance");
    }
    debug!("Epoll instance created: fd={}", epoll_fd);

    // Add PTY fd to epoll
    let mut pty_ep_event = epoll_event {
        events: EPOLLIN as u32,
        u64: PTY_FD_EVENT_ID, // User data to identify this event source
    };
    // Safety: FFI call
    if unsafe { epoll_ctl(epoll_fd, EPOLL_CTL_ADD, pty_fd, &mut pty_ep_event) } < 0 {
        let err = std::io::Error::last_os_error();
        unsafe {
            libc::close(epoll_fd);
        }
        return Err(err).context(format!("Failed to add PTY fd {} to epoll", pty_fd));
    }
    debug!("Added PTY fd {} to epoll", pty_fd);

    // Add Driver's event fd to epoll (if provided)
    if let Some(driver_event_fd) = driver.get_event_fd() {
        debug!("Driver provides event fd: {}", driver_event_fd);
        let mut driver_ep_event = epoll_event {
            events: EPOLLIN as u32,
            u64: DRIVER_FD_EVENT_ID, // User data
        };
        // Safety: FFI call
        if unsafe {
            epoll_ctl(
                epoll_fd,
                EPOLL_CTL_ADD,
                driver_event_fd,
                &mut driver_ep_event,
            )
        } < 0
        {
            let err = std::io::Error::last_os_error();
            unsafe {
                libc::close(epoll_fd);
            } // Also remove PTY fd from epoll if robust
            return Err(err).context(format!(
                "Failed to add Driver event fd {} to epoll",
                driver_event_fd
            ));
        }
        debug!("Added Driver event fd {} to epoll", driver_event_fd);
    } else {
        info!(
            "Driver does not provide an event fd; events might be polled or handled differently."
        );
        // TODO: If no driver FD, the loop might need a timeout for driver.process_events()
        // or the driver handles its events on a separate thread.
        // For now, this example assumes drivers that integrate with epoll are preferred.
    }

    // 6. Main Event Loop
    let mut pty_buffer = [0u8; PTY_READ_BUF_SIZE];
    let mut ep_events: [epoll_event; MAX_EPOLL_EVENTS] = unsafe { std::mem::zeroed() };

    let mut needs_render = true; // Initial render
    let mut needs_exit = false;

    info!("Starting main event loop...");
    while !needs_exit {
        // Safety: FFI call to epoll_wait
        let num_events = unsafe {
            epoll_wait(
                epoll_fd,
                ep_events.as_mut_ptr(),
                MAX_EPOLL_EVENTS as c_int,
                EPOLL_TIMEOUT_MS,
            )
        };

        if num_events < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                trace!("epoll_wait interrupted, continuing.");
                continue; // EINTR
            }
            // Close FDs before returning
            unsafe {
                libc::close(epoll_fd);
            }
            return Err(err).context("epoll_wait failed");
        }

        for i in 0..num_events as usize {
            let event_id = ep_events[i].u64;
            match event_id {
                PTY_FD_EVENT_ID => {
                    trace!("Activity on PTY fd {}", pty_fd);
                    match pty_master_file.read(&mut pty_buffer) {
                        Ok(0) => {
                            info!("PTY EOF received (shell likely exited).");
                            needs_exit = true;
                        }
                        Ok(bytes_read) => {
                            debug!("Read {} bytes from PTY.", bytes_read);
                            let commands = ansi_processor.process_bytes(&pty_buffer[..bytes_read]);
                            for cmd in commands {
                                let input = EmulatorInput::Ansi(cmd);
                                if let Some(action) = term_emulator.interpret_input(input) {
                                    if matches!(&action, EmulatorAction::RequestRedraw) {
                                        needs_render = true;
                                    }
                                    handle_emulator_action(
                                        action,
                                        pty_fd,
                                        &mut *driver,
                                        &mut term_emulator,
                                    )?;
                                }
                            }
                            // If any ANSI commands were processed, assume a redraw is needed.
                            // More fine-grained checks could be based on term_emulator.is_dirty()
                            if term_emulator.take_dirty_lines().is_empty() && !needs_render {
                                // if no explicit redraw request and no dirty lines from interpret_input
                            } else {
                                needs_render = true;
                            }
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            trace!("PTY read would block (EWOULDBLOCK/EAGAIN).");
                            // This shouldn't happen often if epoll reported EPOLLIN,
                            // but non-blocking read can still return it.
                        }
                        Err(e) => {
                            error!("Error reading from PTY: {}", e);
                            needs_exit = true; // Treat PTY read errors as fatal for the session
                        }
                    }
                }
                DRIVER_FD_EVENT_ID => {
                    trace!("Activity on Driver event fd.");
                    let backend_events = driver
                        .process_events()
                        .context("Driver failed to process events")?;
                    for be_event in backend_events {
                        handle_backend_event(
                            be_event,
                            &mut term_emulator,
                            &mut *driver,
                            pty_fd,
                            &mut needs_exit,
                            &mut needs_render,
                        )?;
                        if needs_exit {
                            break;
                        }
                    }
                }
                _ => {
                    warn!("epoll_wait returned unknown event id: {}", event_id);
                }
            }
            if needs_exit {
                break;
            }
        }
        if needs_exit {
            break;
        }

        // If driver doesn't have an event FD, we might need to poll it periodically.
        // This is a simplified example; a real implementation might use a timeout in epoll_wait
        // and call process_events if driver.get_event_fd().is_none() and timeout occurs.
        if driver.get_event_fd().is_none() {
            let backend_events = driver
                .process_events()
                .context("Driver failed to process events (polling)")?;
            for be_event in backend_events {
                handle_backend_event(
                    be_event,
                    &mut term_emulator,
                    &mut *driver,
                    pty_fd,
                    &mut needs_exit,
                    &mut needs_render,
                )?;
                if needs_exit {
                    break;
                }
            }
        }

        if needs_render {
            trace!("Calling renderer.draw()");
            renderer
                .draw(&mut term_emulator, &mut *driver)
                .context("Renderer failed to draw")?;
            needs_render = false;
        }
    }

    info!("Event loop finished. Cleaning up...");

    // 7. Cleanup
    // Remove FDs from epoll (optional, as epoll_fd will be closed)
    // Safety: FFI call
    unsafe {
        if driver.get_event_fd().is_some() {
            epoll_ctl(
                epoll_fd,
                EPOLL_CTL_DEL,
                driver.get_event_fd().unwrap(),
                std::ptr::null_mut(),
            );
        }
        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, pty_fd, std::ptr::null_mut());
        libc::close(epoll_fd);
        debug!("Epoll instance closed.");
    }

    // Explicitly call driver cleanup, which can return Result
    driver.cleanup().context("Driver cleanup failed")?;
    info!("Driver cleaned up.");

    // PTY master file will be closed when `_pty_master_file` goes out of scope.
    // `pty_fd` is the raw fd, also managed by `_pty_master_file`.

    // Optionally, wait for the child shell process to exit
    info!("Waiting for child shell (PID: {}) to exit...", child_pid);
    let mut status: c_int = 0;
    // Safety: FFI call
    if unsafe { libc::waitpid(child_pid, &mut status, 0) } < 0 {
        warn!(
            "waitpid failed for child PID {}: {}",
            child_pid,
            std::io::Error::last_os_error()
        );
    } else {
        if libc::WIFEXITED(status) {
            info!(
                "Child shell exited with status: {}",
                libc::WEXITSTATUS(status)
            );
        } else if libc::WIFSIGNALED(status) {
            info!(
                "Child shell terminated by signal: {}",
                libc::WTERMSIG(status)
            );
        }
    }

    info!("myterm orchestrator finished.");
    Ok(())
}

fn main() {
    // Initialize logger (e.g., env_logger)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Locale initialization is handled lazily by the static LocaleController
    // in `term/unicode.rs` when `get_char_display_width` is first called.

    if let Err(e) = run_terminal_orchestrator() {
        error!("Critical error in myterm: {:?}", e);
        // Print the full error chain for debugging
        let mut cause = e.source();
        while let Some(inner_cause) = cause {
            error!("  Caused by: {:?}", inner_cause);
            cause = inner_cause.source();
        }
        std::process::exit(1);
    }
}
