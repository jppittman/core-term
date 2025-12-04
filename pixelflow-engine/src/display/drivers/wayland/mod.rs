//! Wayland driver module.

mod protocol;
mod listeners;
mod state;

use self::listeners::REGISTRY_LISTENER;
use self::protocol::*;
use self::state::WaylandState;
use crate::channel::{DriverCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use anyhow::{anyhow, Result};
use log::info;
use pixelflow_render::color::Bgra;
use std::ffi::c_void;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use wayland_sys::client::*;

pub struct WaylandDisplayDriver {
    cmd_tx: SyncSender<DriverCommand<Bgra>>,
    waker: WaylandWaker,
    run_state: Option<RunState>,
}

struct RunState {
    cmd_rx: Receiver<DriverCommand<Bgra>>,
    pipe_read: OwnedFd,
    engine_tx: EngineSender<Bgra>,
}

#[derive(Clone)]
struct WaylandWaker {
    fd: Arc<OwnedFd>,
}

impl WaylandWaker {
    fn wake(&self) -> Result<()> {
        let buf = [1u8];
        let ret = unsafe {
            libc::write(self.fd.as_raw_fd(), buf.as_ptr() as *const c_void, 1)
        };
        if ret < 0 {
            // Ignore EAGAIN/EWOULDBLOCK
        }
        Ok(())
    }
}

impl Clone for WaylandDisplayDriver {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            waker: self.waker.clone(),
            run_state: None,
        }
    }
}

impl DisplayDriver for WaylandDisplayDriver {
    type Pixel = Bgra;

    fn new(engine_tx: EngineSender<Bgra>) -> Result<Self> {
        let (cmd_tx, cmd_rx) = sync_channel(16);
        let (pipe_read, pipe_write) = create_pipe()?;

        let waker = WaylandWaker {
            fd: Arc::new(pipe_write),
        };

        Ok(Self {
            cmd_tx,
            waker,
            run_state: Some(RunState {
                cmd_rx,
                pipe_read,
                engine_tx,
            }),
        })
    }

    fn send(&self, cmd: DriverCommand<Bgra>) -> Result<()> {
        let mut cmd = cmd;
        loop {
            match self.cmd_tx.try_send(cmd) {
                Ok(()) => {
                    self.waker.wake()?;
                    return Ok(());
                }
                Err(TrySendError::Full(returned)) => {
                    self.waker.wake()?;
                    cmd = returned;
                    std::thread::yield_now();
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(anyhow!("Wayland driver channel disconnected"));
                }
            }
        }
    }

    fn run(&self) -> Result<()> {
        let run_state = self
            .run_state
            .as_ref()
            .ok_or_else(|| anyhow!("Only original driver can run"))?;

        // Initialize interfaces
        unsafe { protocol::init_interfaces(); }

        unsafe {
            let display = wl_display_connect(ptr::null());
            if display.is_null() {
                return Err(anyhow!("Failed to connect to Wayland display"));
            }

            let mut state = WaylandState::new(display, run_state.engine_tx.clone())?;

            // Get registry (opcode 1 on wl_display)
            let registry = wl_proxy_marshal_constructor(
                display as *mut wl_proxy,
                1,
                &wl_registry_interface
            );

            wl_proxy_add_listener(registry, &REGISTRY_LISTENER as *const _ as *mut extern "C" fn(), &mut state as *mut _ as *mut c_void);

            // Roundtrip to get globals
            wl_display_roundtrip(display);

            // Verify globals
            if state.compositor.is_null() { return Err(anyhow!("Missing wl_compositor")); }
            if state.shm.is_null() { return Err(anyhow!("Missing wl_shm")); }
            if state.wm_base.is_null() { return Err(anyhow!("Missing xdg_wm_base")); }

            info!("Wayland: Event loop starting");

            let wayland_fd = wl_display_get_fd(display);
            let pipe_fd = run_state.pipe_read.as_raw_fd();

            let mut poll_fds = [
                libc::pollfd {
                    fd: wayland_fd,
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: pipe_fd,
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];

            while state.running {
                // Prepare read
                while wl_display_prepare_read(display) != 0 {
                    wl_display_dispatch_pending(display);
                }

                // Flush
                wl_display_flush(display);

                // Poll
                let ret = libc::poll(poll_fds.as_mut_ptr(), 2, -1);
                if ret < 0 {
                    wl_display_cancel_read(display);
                    let err = std::io::Error::last_os_error();
                    if err.kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    break;
                }

                // Read events
                if poll_fds[0].revents & libc::POLLIN != 0 {
                    wl_display_read_events(display);
                } else {
                    wl_display_cancel_read(display);
                }

                // Dispatch
                if wl_display_dispatch_pending(display) < 0 {
                    break;
                }

                // Handle commands
                if poll_fds[1].revents & libc::POLLIN != 0 {
                    let mut buf = [0u8; 128];
                    libc::read(pipe_fd, buf.as_mut_ptr() as *mut c_void, 128);

                    while let Ok(cmd) = run_state.cmd_rx.try_recv() {
                        state.handle_command(cmd);
                    }
                }
            }

            info!("Wayland: Shutdown");
            wl_display_disconnect(display);
        }

        Ok(())
    }
}

fn create_pipe() -> Result<(OwnedFd, OwnedFd)> {
    let mut fds: [RawFd; 2] = [0; 2];
    unsafe {
        if libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok((
            OwnedFd::from_raw_fd(fds[0]),
            OwnedFd::from_raw_fd(fds[1]),
        ))
    }
}
