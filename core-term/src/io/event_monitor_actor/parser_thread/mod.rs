// src/io/event_monitor_actor/parser_thread/mod.rs

use crate::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
use crate::io::traits::PtySender;
use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, SystemStatus,
};
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
        cmd_tx: Box<dyn PtySender>,
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
            if let Err(panic_payload) = handle.join() {
                if std::thread::panicking() {
                    // Already unwinding - can't double-panic, just log
                    eprintln!(
                        "Parser thread panicked (during unwind): {:?}",
                        panic_payload
                    );
                } else {
                    // Propagate the panic - this is a fatal error
                    std::panic::resume_unwind(panic_payload);
                }
            }
        }
    }
}

struct ParserActor {
    parser: AnsiProcessor,
    cmd_tx: Box<dyn PtySender>,
    recycler_tx: Sender<Vec<u8>>,
}

impl Actor<Vec<u8>, NoControl, NoManagement> for ParserActor {
    fn handle_data(&mut self, data: Vec<u8>) -> HandlerResult {
        if data.is_empty() {
            // Even if empty, recycle it (ignore error if read thread exited)
            self.recycler_tx.send(data).ok();
            return Ok(());
        }

        // Process raw bytes through ANSI parser (AnsiProcessor implements AnsiParser trait)
        // process_bytes returns Vec<AnsiCommand>
        let commands = self.parser.process_bytes(&data);

        // Recycle buffer back to read thread (ignore error if it exited)
        self.recycler_tx.send(data).ok();

        if !commands.is_empty() {
            // Send batch of parsed commands to app
            if let Err(e) = self.cmd_tx.send(commands) {
                warn!("Parser failed to send commands to app: {}", e);
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, _msg: NoControl) -> HandlerResult {
        // No control messages
        Ok(())
    }

    fn handle_management(&mut self, _msg: NoManagement) -> HandlerResult {
        // No management messages
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // No periodic tasks
        Ok(ActorStatus::Idle)
    }
}
