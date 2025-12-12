// src/io/event_monitor_actor/parser_thread/mod.rs

use crate::ansi::commands::AnsiCommand;
use crate::ansi::{AnsiParser, AnsiProcessor}; // Use AnsiProcessor from ansi/mod.rs
use actor_scheduler::{Actor, ActorHandle, ActorScheduler};
use std::sync::mpsc::SyncSender;
use std::thread;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoControl;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoManagement;

/// Parser Actor: Receives raw bytes, parses ANSI, sends to App.
pub struct ParserThread {
    parser: AnsiProcessor,
    app_tx: SyncSender<Vec<AnsiCommand>>,
}

impl ParserThread {
    fn new(app_tx: SyncSender<Vec<AnsiCommand>>) -> Self {
        Self {
            parser: AnsiProcessor::new(),
            app_tx,
        }
    }
}

impl Actor<Vec<u8>, NoControl, NoManagement> for ParserThread {
    fn handle_data(&mut self, msg: Vec<u8>) {
        // Parse raw bytes using AnsiParser trait method
        let commands = self.parser.process_bytes(&msg);

        // Send batch to App
        if !commands.is_empty() {
            if let Err(_) = self.app_tx.send(commands) {
                log::warn!("ParserThread: App disconnected, cannot send commands");
            }
        }
    }

    fn handle_control(&mut self, _msg: NoControl) {}
    fn handle_management(&mut self, _msg: NoManagement) {}

    fn park(&mut self) {}
}

/// Creates and spawns the Parser Actor.
///
/// # Arguments
/// * `app_tx` - Channel to send parsed commands to the application.
///
/// # Returns
/// * `ActorHandle` to send raw bytes to this parser.
/// * `JoinHandle` of the parser thread.
pub fn create_parser_actor(
    app_tx: SyncSender<Vec<AnsiCommand>>,
) -> (
    ActorHandle<Vec<u8>, NoControl, NoManagement>,
    thread::JoinHandle<()>,
) {
    let (handle, mut scheduler) = ActorScheduler::<Vec<u8>, NoControl, NoManagement>::new(
        1024, // burst limit
        4096, // buffer size
    );

    let mut parser_actor = ParserThread::new(app_tx);

    let thread_handle = thread::Builder::new()
        .name("pty-parser".to_string())
        .spawn(move || {
            scheduler.run(&mut parser_actor);
        })
        .expect("Failed to spawn parser thread");

    (handle, thread_handle)
}
