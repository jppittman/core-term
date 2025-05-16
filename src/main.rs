// In src/main.rs (or lib.rs)

// Declare modules
pub mod ansi;
pub mod backends;
pub mod glyph;
pub mod os; // Contains `pub mod pty;` in src/os/mod.rs
pub mod orchestrator; // Our new orchestrator
pub mod renderer;
pub mod term;

// Use statements for items needed in main.rs
use crate::{
    backends::{console::ConsoleDriver, x11::XDriver, BackendEvent, Driver}, // Example drivers
    orchestrator::{AppOrchestrator, OrchestratorStatus},
    os::pty::{NixPty, PtyChannel, PtyConfig, PtyError}, // Use concrete NixPty for setup
    renderer::{Renderer, RendererInterface}, // Assuming Renderer implements RendererInterface
    term::{TerminalEmulator, TerminalInterface}, // Use concrete TerminalEmulator for setup
    ansi::AnsiParser,
};
use std::os::unix::io::{AsRawFd, RawFd}; // For FD handling with epoll

// Logging
use log::{debug, error, info, trace, warn};

// For epoll (if used)
use nix::sys::epoll::{self, EpollEvent, EpollFlags, EpollOp}; // Example with nix epoll

// Constants for epoll tokens (example)
const PTY_EPOLL_TOKEN: u64 = 1;
const DRIVER_EPOLL_TOKEN: u64 = 2;


fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_micros()
        .init();

    info!("Starting myterm orchestrator...");

    // --- Configuration ---
    // TODO: Load from config file or command line arguments
    let shell_path = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
    let shell_args: Vec<String> = Vec::new(); // e.g., parse from config or fixed like ["-l"]
    let shell_args_str: Vec<&str> = shell_args.iter().map(AsRef::as_ref).collect();

    // Initial dimensions (these might be updated by driver after window creation)
    let mut initial_cols = backends::DEFAULT_WINDOW_WIDTH_CHARS as u16;
    let mut initial_rows = backends::DEFAULT_WINDOW_HEIGHT_CHARS as u16;


    // --- Setup PTY ---
    // STYLE GUIDE: Idempotency - NixPty::spawn_with_config is not idempotent.
    // STYLE GUIDE: Error handling - unwrap_or_else for critical setup.
    let pty_config = PtyConfig {
        command_executable: &shell_path,
        args: &shell_args_str, // Ensure args[0] is command_executable if needed by your exec logic
        initial_cols,
        initial_rows,
    };
    let mut pty_channel = NixPty::spawn_with_config(&pty_config).unwrap_or_else(|e| {
        eprintln!("Fatal: Failed to spawn PTY: {}", e);
        std::process::exit(1);
    });
    let pty_fd = pty_channel.as_raw_fd();
    log::info!("PTY spawned with master fd: {}", pty_fd);


    // --- Setup Driver (e.g., X11 or Console) ---
    // STYLE GUIDE: Idempotency - Driver::new() might not be idempotent.
    // TODO: Select driver based on config or environment
    let mut driver: Box<dyn Driver> = match XDriver::new() {
        Ok(d) => Box::new(d),
        Err(e) => {
            warn!("Failed to initialize X11 driver: {}. Falling back to ConsoleDriver.", e);
            Box::new(ConsoleDriver::new().unwrap_or_else(|ce| {
                eprintln!("Fatal: Failed to initialize ConsoleDriver: {}", ce);
                std::process::exit(1);
            }))
        }
    };
    info!("Driver initialized.");

    // Update initial dimensions from driver if possible
    let (display_width_px, display_height_px) = driver.get_display_dimensions_pixels();
    let (char_width_px, char_height_px) = driver.get_font_dimensions();
    if char_width_px > 0 && char_height_px > 0 {
        initial_cols = (display_width_px / char_width_px as u16).max(1);
        initial_rows = (display_height_px / char_height_px as u16).max(1);
        if let Err(e) = pty_channel.resize(initial_cols, initial_rows) {
            warn!("Failed to resize PTY to initial driver dimensions: {}", e);
        }
    }
    info!("Initial terminal dimensions set to: {}x{} cells", initial_cols, initial_rows);


    // --- Setup Terminal Emulator, Parser, Renderer ---
    // STYLE GUIDE: Idempotency - new() methods are not usually idempotent.
    let mut term_emulator = TerminalEmulator::new(
        initial_cols as usize,
        initial_rows as usize,
        1000, // Scrollback limit
    );
    let ansi_parser = AnsiParser::new(); // Assuming a simple constructor
    let mut renderer = Renderer::new(); // Assuming Renderer implements RendererInterface


    // --- Create AppOrchestrator ---
    let mut orchestrator = AppOrchestrator::new(
        &mut pty_channel,
        &mut term_emulator,
        ansi_parser,
        &mut renderer,
        &mut *driver, // Deref Box to get &mut dyn Driver
    );
    info!("AppOrchestrator created.");


    // --- Event Loop (using epoll as an example) ---
    let epoll_fd = epoll::epoll_create1(epoll::EpollCreateFlags::EPOLL_CLOEXEC)
        .map_err(|e| anyhow::anyhow!("Failed to create epoll instance: {}", e))?;
    
    let mut pty_epoll_event = EpollEvent::new(EpollFlags::EPOLLIN, PTY_EPOLL_TOKEN);
    epoll::epoll_ctl(epoll_fd, EpollOp::EpollCtlAdd, pty_fd, &mut pty_epoll_event)
        .map_err(|e| anyhow::anyhow!("Failed to add PTY FD to epoll: {}", e))?;
    log::trace!("PTY FD {} added to epoll for read events.", pty_fd);

    if let Some(driver_event_fd) = orchestrator.driver.get_event_fd() {
        let mut driver_epoll_event = EpollEvent::new(EpollFlags::EPOLLIN, DRIVER_EPOLL_TOKEN);
        epoll::epoll_ctl(epoll_fd, EpollOp::EpollCtlAdd, driver_event_fd, &mut driver_epoll_event)
            .map_err(|e| anyhow::anyhow!("Failed to add Driver FD to epoll: {}", e))?;
        log::trace!("Driver FD {} added to epoll for read events.", driver_event_fd);
    } else {
        log::info!("Driver does not provide an event FD. Main loop will need to poll driver manually if it doesn't use other means.");
        // This would require a different event loop structure if driver needs polling.
    }
    
    let mut events = [EpollEvent::empty(); 2]; // Buffer for epoll_wait
    info!("Starting main event loop...");

    'main_loop: loop {
        // STYLE GUIDE: Avoid magic numbers for timeout. -1 blocks indefinitely.
        // A timeout (e.g., 16ms for ~60Hz) can be used for periodic tasks or if driver needs polling.
        let num_events = match epoll::epoll_wait(epoll_fd, &mut events, -1) {
            Ok(n) => n,
            Err(e) if e == nix::errno::Errno::EINTR => {
                log::trace!("epoll_wait interrupted, continuing.");
                continue; // Interrupted by signal, retry.
            }
            Err(e) => {
                error!("epoll_wait failed: {}", e);
                break 'main_loop; // Exit on unrecoverable epoll error.
            }
        };

        log::trace!("epoll_wait returned {} events.", num_events);

        for i in 0..num_events {
            let event_token = events[i].data();
            match event_token {
                PTY_EPOLL_TOKEN => {
                    log::trace!("Event on PTY FD.");
                    match orchestrator.process_pty_events() {
                        Ok(OrchestratorStatus::Running) => {}
                        Ok(OrchestratorStatus::Shutdown) => {
                            info!("PTY indicated shutdown. Exiting main loop.");
                            break 'main_loop;
                        }
                        Err(e) => {
                            error!("Error processing PTY events: {}. Exiting.", e);
                            break 'main_loop;
                        }
                    }
                }
                DRIVER_EPOLL_TOKEN => {
                    log::trace!("Event on Driver FD.");
                    // The driver is responsible for reading from its own FD.
                    // process_driver_events will call driver.process_events().
                    match orchestrator.process_driver_events() {
                        Ok(OrchestratorStatus::Running) => {}
                        Ok(OrchestratorStatus::Shutdown) => {
                            info!("Driver indicated shutdown. Exiting main loop.");
                            break 'main_loop;
                        }
                        Err(msg) => { // process_driver_events returns Result<Status, String>
                            error!("Error processing driver events: {}. Exiting.", msg);
                            break 'main_loop;
                        }
                    }
                }
                _ => {
                    warn!("Unknown epoll token: {}", event_token);
                }
            }
        }

        // Perform rendering once per loop iteration if anything flagged it.
        if let Err(e) = orchestrator.render_if_needed() {
            error!("Rendering failed: {}. Exiting.", e);
            break 'main_loop;
        }
    }

    info!("Exiting myterm orchestrator. Cleaning up...");
    // NixPty's Drop will handle closing master_fd and attempting to terminate child.
    // Driver's Drop (or an explicit cleanup method if needed) should handle its resources.
    orchestrator.driver.cleanup()?; // Explicit cleanup for driver
    info!("Cleanup complete.");
    Ok(())
}
