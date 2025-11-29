use crate::channel::{EngineCommand, EngineSender};
use anyhow::{Context, Result};
use log::*;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug)]
pub enum VsyncCommand {
    SetFps(u32),
}

pub struct VsyncActor {
    thread_handle: Option<JoinHandle<()>>,
    control_tx: Sender<VsyncCommand>,
}

impl VsyncActor {
    pub fn spawn(engine_tx: EngineSender, initial_fps: u32) -> Result<Self> {
        let (control_tx, control_rx) = mpsc::channel();

        let thread_handle = thread::Builder::new()
            .name("vsync".to_string())
            .spawn(move || {
                run_vsync_loop(engine_tx, control_rx, initial_fps);
            })
            .context("Failed to spawn vsync thread")?;

        Ok(Self {
            thread_handle: Some(thread_handle),
            control_tx,
        })
    }

    pub fn set_fps(&self, fps: u32) -> Result<()> {
        self.control_tx.send(VsyncCommand::SetFps(fps)).context("Failed to send SetFps command")
    }
}

impl Drop for VsyncActor {
    fn drop(&mut self) {
        // Dropping control_tx closes the channel, which causes run_vsync_loop to exit
        // when it sees Disconnected.
        // However, if it's sleeping in recv_timeout, it won't see it until timeout.
        // This causes the join to block for up to 1 frame.
        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                error!("VsyncActor thread panicked: {:?}", e);
            }
        }
    }
}

fn run_vsync_loop(engine_tx: EngineSender, control_rx: Receiver<VsyncCommand>, mut fps: u32) {
    info!("VsyncActor: Started (target: {} FPS)", fps);

    loop {
        // Clamp FPS to avoid division by zero or excessive values
        let safe_fps = fps.max(1).min(240);
        let frame_duration = Duration::from_secs_f64(1.0 / safe_fps as f64);

        match control_rx.recv_timeout(frame_duration) {
            Ok(cmd) => {
                match cmd {
                    VsyncCommand::SetFps(new_fps) => {
                        info!("VsyncActor: FPS changed to {}", new_fps);
                        fps = new_fps;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Timeout means frame duration passed. Tick!
                if engine_tx.send(EngineCommand::Tick).is_err() {
                    info!("VsyncActor: Engine channel closed, exiting");
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                 debug!("VsyncActor: Control channel closed, exiting");
                 break;
            }
        }
    }
    debug!("VsyncActor: Thread exiting");
}
