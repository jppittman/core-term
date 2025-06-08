use crate::platform::actions::PlatformAction;
use crate::platform::backends::{
    cocoa::CocoaDriver, BackendEvent, CursorVisibility, Driver, PlatformState,
};
use crate::platform::os::PtyActionCommand; // Correct path for PtyActionCommand
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;
use anyhow::{Context, Result};
use std::sync::mpsc::{channel, Receiver, Sender};

pub struct MacosPlatform {
    driver: CocoaDriver,
    // Channels for PTY interaction (example names)
    // pty_event_receiver receives events from the PTY backend (e.g., data output)
    pty_event_receiver: Receiver<BackendEvent>, // Assuming PTY generates BackendEvents directly for now
    // pty_action_sender sends commands to the PTY backend (e.g., write input, resize)
    pty_action_sender: Sender<PtyActionCommand>,

    // Channels for UI interaction (example names)
    // ui_action_sender sends commands to the UI Driver (CocoaDriver)
    // MacosPlatform itself is the client of CocoaDriver, so it calls methods on driver directly.
    // However, if CocoaDriver needed to send events back to MacosPlatform *asynchronously*,
    // a channel like ui_event_receiver would be needed. For now, driver.poll_event() is synchronous.

    // Shell command and args, stored if needed for PTY backend re-creation or management
    _shell_command: String,
    _shell_args: Vec<String>,
}

impl Platform for MacosPlatform {
    fn new(
        initial_pty_cols: u16,
        initial_pty_rows: u16,
        shell_command: String,
        shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)>
    where
        Self: Sized,
    {
        println!(
            "MacosPlatform: Initializing with PTY cols={}, rows={}, command='{}', args='{:?}'",
            initial_pty_cols, initial_pty_rows, &shell_command, &shell_args
        );

        let driver = CocoaDriver::new().context("Failed to initialize CocoaDriver")?;
        let initial_platform_state = driver.get_platform_state();

        // --- PTY Channel Setup (Placeholders) ---
        // In a real scenario, a PTY backend (e.g., using forkpty) would be created here.
        // It would get one end of pty_action_channel and pty_event_channel.
        let (pty_action_sender_to_backend, _pty_action_receiver_in_backend) =
            channel::<PtyActionCommand>();
        let (_pty_event_sender_from_backend, pty_event_receiver_for_platform) =
            channel::<BackendEvent>();
        // TODO: Spawn PTY backend thread/task, passing shell_command, shell_args,
        //       initial_pty_cols, initial_pty_rows, _pty_action_receiver_in_backend,
        //       and _pty_event_sender_from_backend.

        println!("MacosPlatform: CocoaDriver initialized, PTY channels created (placeholders).");

        Ok((
            MacosPlatform {
                driver,
                pty_event_receiver: pty_event_receiver_for_platform,
                pty_action_sender: pty_action_sender_to_backend,
                _shell_command: shell_command,
                _shell_args: shell_args,
            },
            initial_platform_state,
        ))
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        let mut events: Vec<PlatformEvent> = Vec::new();

        // 1. Poll UI Driver for events
        match self.driver.process_events() {
            Ok(mut driver_events) => {
                for event in driver_events.drain(..) {
                    // println!("MacosPlatform: Received UI event: {:?}", event);
                    events.push(PlatformEvent::BackendEvent(event));
                }
            }
            Err(e) => {
                // Log error or handle critical failure
                eprintln!("MacosPlatform: Error polling UI driver: {}", e);
                // Depending on error type, might propagate or attempt recovery
            }
        }

        // 2. Poll PTY for events (e.g., data from shell)
        match self.pty_event_receiver.try_recv() {
            Ok(pty_event) => {
                // println!("MacosPlatform: Received PTY event: {:?}", pty_event);
                events.push(PlatformEvent::BackendEvent(pty_event));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => { /* No PTY event */ }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // PTY backend might have exited. This could be a shutdown signal.
                println!("MacosPlatform: PTY event channel disconnected.");
                // Consider generating a specific PlatformEvent::Shutdown or similar
                // Or propagate error to signal AppOrchestrator to stop
                return Err(anyhow::anyhow!("PTY backend disconnected"));
            }
        }
        Ok(events)
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        // println!("MacosPlatform: dispatch_actions({:?})", actions);
        for action in actions {
            match action {
                PlatformAction::Write(data) => {
                    self.pty_action_sender
                        .send(PtyActionCommand::Write(data))
                        .context("Failed to send Write command to PTY")?;
                }
                PlatformAction::ResizePty { cols, rows } => {
                    self.pty_action_sender
                        .send(PtyActionCommand::Resize { cols, rows })
                        .context("Failed to send Resize command to PTY")?;
                }
                PlatformAction::Render(render_commands) => {
                    self.driver
                        .execute_render_commands(render_commands)
                        .context("MacosPlatform: Failed to dispatch Render to driver")?;
                    // As per previous note, PresentFrame might be implicitly handled by driver
                    // or explicitly called. Let's add an explicit PresentFrame after Render.
                    self.driver
                        .present()
                        .context("MacosPlatform: Failed to dispatch PresentFrame to driver")?;
                }
                PlatformAction::SetTitle(title) => {
                    self.driver.set_title(&title);
                    // .context("MacosPlatform: Failed to dispatch SetTitle to driver")?; // set_title doesn't return Result
                }
                PlatformAction::RingBell => {
                    self.driver.bell();
                    // .context("MacosPlatform: Failed to dispatch RingBell to driver")?; // bell doesn't return Result
                }
                PlatformAction::CopyToClipboard(text) => {
                    // Using a placeholder atom (0) for now. A real implementation would use a Cocoa-specific way.
                    self.driver.own_selection(0, text);
                    // .context("MacosPlatform: Failed to dispatch CopyToClipboard to driver")?; // own_selection doesn't return Result
                }
                PlatformAction::SetCursorVisibility(visible) => {
                    let visibility = if visible {
                        CursorVisibility::Shown
                    } else {
                        CursorVisibility::Hidden
                    };
                    self.driver.set_cursor_visibility(visibility);
                    // .context("MacosPlatform: Failed to dispatch SetCursorVisibility to driver")?; // set_cursor_visibility doesn't return Result
                }
                PlatformAction::RequestPaste => {
                    // Using placeholder atoms (0, 0) for now.
                    // A real implementation would use Cocoa-specific APIs for pasteboard interaction.
                    self.driver.request_selection_data(0, 0);
                    // .context("MacosPlatform: Failed to dispatch RequestPaste to driver")?; // request_selection_data doesn't return Result
                }
            }
        }
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        // println!("MacosPlatform: Getting current platform state from driver");
        self.driver.get_platform_state()
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("MacosPlatform: cleanup() called. Cleaning up driver...");
        self.driver.cleanup()
    }
}

// Keep the Drop trait for cleanup if MacosPlatform ever holds resources
impl Drop for MacosPlatform {
    fn drop(&mut self) {
        // Placeholder cleanup logic
        println!("MacosPlatform dropped. Shell: '{}'", self._shell_command);
        // Consider sending a shutdown command to PTY backend if it's running.
    }
}
