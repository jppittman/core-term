// src/io/event_monitor_actor/mod.rs

pub mod parser_thread;
pub mod read_thread;
pub mod write_thread;

use crate::ansi::commands::AnsiCommand;
use crate::io::event_monitor_actor::parser_thread::{create_parser_actor, NoControl, NoManagement};
use crate::io::event_monitor_actor::read_thread::ReadThread;
use crate::io::event_monitor_actor::write_thread::WriteThread;
use crate::io::pty::NixPty;
use crate::io::traits::EventSource;
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{Receiver, SyncSender};

/// Actor that monitors PTY events and manages I/O threads.
///
/// This actor owns:
/// 1. `ReadThread`: Polls PTY for reading, sends raw bytes to ParserThread
/// 2. `WriteThread`: Receives bytes from App, writes to PTY (owns PTY)
/// 3. `ParserThread` (via handle): Parses raw bytes into AnsiCommands
pub struct EventMonitorActor {
    _read_thread: ReadThread,
    _write_thread: WriteThread,
    // We don't keep parser thread handle here, it runs detached (ActorScheduler logic)
    // But we could keep the thread handle returned by create_parser_actor if we wanted to join it.
    // For now, we let it run.
    _parser_thread_handle: std::thread::JoinHandle<()>,
}

impl EventMonitorActor {
    /// Spawns the event monitor actor and its sub-threads.
    ///
    /// # Arguments
    ///
    /// * `pty` - The PTY handle (consumed)
    /// * `app_pty_rx_sender` - Channel to send parsed ANSI commands to the App
    /// * `pty_write_rx` - Channel to receive bytes from App to write to PTY
    pub fn spawn<S>(
        pty: S,
        app_pty_rx_sender: SyncSender<Vec<AnsiCommand>>,
        pty_write_rx: Receiver<Vec<u8>>,
    ) -> Result<Self>
    where
        S: EventSource + Into<NixPty> + Clone + 'static,
    {
        info!("Spawning EventMonitorActor...");

        // Clone PTY for reading (EventSource)
        let pty_read = pty.clone();

        // Convert to NixPty for writing (WriteThread takes ownership)
        let pty_write: NixPty = pty.into();

        // 1. Create/Spawn Parser Actor
        let (parser_handle, parser_thread_handle) = create_parser_actor(app_pty_rx_sender);

        // 2. Spawn Read Thread (Reader -> Parser)
        let read_thread =
            ReadThread::spawn(pty_read, parser_handle).context("Failed to spawn ReadThread")?;

        // 3. Spawn Write Thread (App -> Writer -> PTY)
        let write_thread =
            WriteThread::spawn(pty_write, pty_write_rx).context("Failed to spawn WriteThread")?;

        Ok(Self {
            _read_thread: read_thread,
            _write_thread: write_thread,
            _parser_thread_handle: parser_thread_handle,
        })
    }
}
