// src/io/event_monitor_actor/parser_thread/mod.rs

use crate::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
use actor_scheduler::{Actor, ActorScheduler, ParkHint};
use log::*;
use std::sync::mpsc::SyncSender;
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
    pub fn spawn(
        mut rx: ActorScheduler<Vec<u8>, NoControl, NoManagement>,
        cmd_tx: SyncSender<Vec<AnsiCommand>>,
    ) -> anyhow::Result<Self> {
        let handle = std::thread::Builder::new()
            .name("pty-parser".to_string())
            .spawn(move || {
                debug!("Parser thread started");
                let mut parser_actor = ParserActor {
                    parser: AnsiProcessor::new(),
                    cmd_tx,
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
}

impl Actor<Vec<u8>, NoControl, NoManagement> for ParserActor {
    fn handle_data(&mut self, data: Vec<u8>) {
        if data.is_empty() {
            return;
        }

        // Process raw bytes through ANSI parser (AnsiProcessor implements AnsiParser trait)
        // process_bytes returns Vec<AnsiCommand>
        let commands = self.parser.process_bytes(&data);

        if !commands.is_empty() {
            // Send batch of parsed commands to app
            if let Err(e) = self.cmd_tx.send(commands) {
                warn!("Parser failed to send commands to app: {}", e);
            }
        }
    }

    fn handle_control(&mut self, _msg: NoControl) {
        // No control messages
    }

    fn handle_management(&mut self, _msg: NoManagement) {
        // No management messages
    }

    fn park(&mut self, _hint: ParkHint) -> ParkHint {
        // No periodic tasks
        ParkHint::Wait
    }
}
