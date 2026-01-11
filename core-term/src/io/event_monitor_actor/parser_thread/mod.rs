// src/io/event_monitor_actor/parser_thread/mod.rs

use crate::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
use actor_scheduler::{Actor, ActorScheduler, ActorStatus};
use log::*;
use std::sync::mpsc::{Sender, SyncSender};
use std::thread::JoinHandle;

/// Control message for parser thread (unused)
#[derive(Debug, Clone)]
pub struct NoControl;

/// Management message for parser thread (unused)
#[derive(Debug, Clone)]
pub struct NoManagement;

pub struct ParserThread {
    join_handle: Option<JoinHandle<()>>,
}

impl ParserThread {
    /// Spawns the parser thread.
    ///
    /// # Arguments
    ///
    /// * `rx` - Receiver for raw byte batches
    /// * `cmd_tx` - Sender for parsed ANSI command batches
    /// * `recycler_tx` - Sender for recycling buffers back to read thread
    pub fn spawn(
        mut rx: ActorScheduler<Vec<u8>, NoControl, NoManagement>,
        cmd_tx: SyncSender<Vec<AnsiCommand>>,
        recycler_tx: Sender<Vec<u8>>,
    ) -> anyhow::Result<Self> {
        let handle = std::thread::Builder::new()
            .name("pty-parser".to_string())
            .spawn(move || {
                debug!("Parser thread started");
                let mut parser_actor = ParserActor {
                    parser: AnsiProcessor::new(),
                    cmd_tx,
                    recycler_tx,
                };
                rx.run(&mut parser_actor);
                debug!("Parser thread exited");
            })?;

        Ok(Self {
            join_handle: Some(handle),
        })
    }
}

impl Drop for ParserThread {
    fn drop(&mut self) {
        if let Some(handle) = self.join_handle.take() {
            // We can't easily interrupt the actor loop unless the channel closes.
            // The read thread dropping will close the sender, which will stop `rx.run()`.
            let _ = handle.join();
        }
    }
}

struct ParserActor {
    parser: AnsiProcessor,
    cmd_tx: SyncSender<Vec<AnsiCommand>>,
    recycler_tx: Sender<Vec<u8>>,
}

impl Actor<Vec<u8>, NoControl, NoManagement> for ParserActor {
    fn handle_data(&mut self, data: Vec<u8>) -> Result<(), actor_scheduler::ActorError> {
        if data.is_empty() {
            // Even if empty, recycle it
            let _ = self.recycler_tx.send(data);
            return Ok(());
        }

        // Process raw bytes through ANSI parser (AnsiProcessor implements AnsiParser trait)
        // process_bytes returns Vec<AnsiCommand>
        let commands = self.parser.process_bytes(&data);

        // Recycle buffer
        // We ignore the error because the read thread might have exited
        let _ = self.recycler_tx.send(data);

        if !commands.is_empty() {
            // Send batch of parsed commands to app
            if let Err(e) = self.cmd_tx.send(commands) {
                warn!("Parser failed to send commands to app: {}", e);
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, _msg: NoControl) -> Result<(), actor_scheduler::ActorError> {
        // No control messages
        Ok(())
    }

    fn handle_management(&mut self, _msg: NoManagement) -> Result<(), actor_scheduler::ActorError> {
        // No management messages
        Ok(())
    }

    fn park(&mut self, _hint: ActorStatus) -> ActorStatus {
        // No periodic tasks
        ActorStatus::Idle
    }
}
