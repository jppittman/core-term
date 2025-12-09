//! ANSI Parser thread for EventMonitorActor.
//!
//! This module handles ANSI parsing in a dedicated thread, receiving raw bytes
//! from the read thread and sending parsed commands to the worker.

use crate::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
use anyhow::{Context, Result};
use log::*;
use actor_scheduler::{Actor, ActorScheduler};
use std::sync::mpsc::SyncSender;
use std::thread::{self, JoinHandle};

/// Maximum commands to accumulate before sending batch.
/// Batching reduces channel overhead while keeping latency reasonable.
const MAX_COMMANDS_PER_BATCH: usize = 1000;

/// Empty type for unused Control lane in parser scheduler
#[derive(Debug, Clone)]
pub enum NoControl {}

/// Empty type for unused Management lane in parser scheduler
#[derive(Debug, Clone)]
pub enum NoManagement {}

/// Parser state that implements Actor for the parser actor.
pub(super) struct ParserState {
    ansi_parser: AnsiProcessor,
    command_batch: Vec<AnsiCommand>,
    cmd_tx: SyncSender<Vec<AnsiCommand>>,
}

impl ParserState {
    pub(super) fn new(cmd_tx: SyncSender<Vec<AnsiCommand>>) -> Self {
        Self {
            ansi_parser: AnsiProcessor::new(),
            command_batch: Vec::new(),
            cmd_tx,
        }
    }

    fn flush_batch(&mut self) {
        if !self.command_batch.is_empty() {
            let batch = std::mem::take(&mut self.command_batch);
            let batch_len = batch.len();

            if let Err(_) = self.cmd_tx.send(batch) {
                error!("Parser: App channel disconnected");
            } else {
                trace!("Parser: Flushed {} commands to app", batch_len);
            }
        }
    }
}

impl Actor<Vec<u8>, NoControl, NoManagement> for ParserState {
    fn handle_data(&mut self, raw_bytes: Vec<u8>) {
        // Parse bytes into ANSI commands
        let mut ansi_commands = self.ansi_parser.process_bytes(&raw_bytes);

        if !ansi_commands.is_empty() {
            debug!(
                "Parser: Parsed {} ANSI commands from {} bytes",
                ansi_commands.len(),
                raw_bytes.len()
            );

            // Accumulate commands in batch
            self.command_batch.append(&mut ansi_commands);

            // Send batch if it's large enough (amortizes channel overhead)
            while self.command_batch.len() >= MAX_COMMANDS_PER_BATCH {
                let batch: Vec<_> = self.command_batch.drain(..MAX_COMMANDS_PER_BATCH).collect();
                let batch_len = batch.len();

                match self.cmd_tx.send(batch) {
                    Ok(()) => {
                        trace!("Parser: Sent batch of {} commands to app", batch_len);
                    }
                    Err(_) => {
                        error!("Parser: Worker channel disconnected");
                        return;
                    }
                }
            }
        }

        // Flush remaining commands after each byte batch (keeps latency low)
        self.flush_batch();
    }

    fn handle_control(&mut self, _msg: NoControl) {
        // No control messages for parser
    }

    fn handle_management(&mut self, _msg: NoManagement) {
        // No management messages for parser
    }

}

// ParserThread is no longer needed - parser is spawned using actor_scheduler::spawn()
