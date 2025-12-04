//! Wayland DisplayDriver implementation.
//!
//! Uses `wayland-client` and `libc` to implement a native Wayland backend.
//! Handles SHM buffer management manually to reduce dependencies.

use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, WindowId};
use crate::input::{KeySymbol, Modifiers};
use anyhow::{anyhow, Context, Result};
use log::{error, info, trace, warn};
use pixelflow_render::color::Bgra;
use pixelflow_render::Frame;
use std::collections::VecDeque;
use std::ffi::{c_void, CString};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use wayland_client::{
    protocol::{
        wl_buffer, wl_callback, wl_compositor, wl_keyboard, wl_pointer, wl_registry, wl_seat,
        wl_shm, wl_shm_pool, wl_surface,
    },
    Connection, Dispatch, QueueHandle, WEnum,
};
use wayland_cursor::CursorTheme;
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};
use xkbcommon::xkb;

// --- Constants ---
const SHM_FORMAT: wl_shm::Format = wl_shm::Format::Argb8888;

// --- Driver Struct ---

pub struct WaylandDisplayDriver {
    cmd_tx: SyncSender<DriverCommand<Bgra>>,
    waker: WaylandWaker,
    /// Only present on the original instance created by `new`
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
            run_state: None, // Clones can't run
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
            .ok_or_else(|| anyhow!("Only original driver can run (this is a clone)"))?;

        // Initialize Wayland connection
        let conn = Connection::connect_to_env()?;

        let mut event_queue = conn.new_event_queue::<WaylandState>();
        let qh = event_queue.handle();

        // Initialize State
        let mut state = WaylandState::new(conn.clone(), qh.clone(), run_state.engine_tx.clone())?;

        // Get registry and roundtrip to bind globals
        let _registry = conn.display().get_registry(&qh, ());
        event_queue.roundtrip(&mut state)?;

        // Verify required globals and init cursor
        if state.compositor.is_none() { return Err(anyhow!("Missing wl_compositor")); }
        if let Some(shm) = state.shm.clone() {
            if let Ok(theme) = CursorTheme::load(&conn, shm, 24) {
                state.cursor_theme = Some(theme);
                if let Some(compositor) = &state.compositor {
                    state.cursor_surface = Some(compositor.create_surface(&qh, ()));
                }
            } else {
                warn!("Failed to load cursor theme");
            }
        } else {
            return Err(anyhow!("Missing wl_shm"));
        }
        if state.wm_base.is_none() { return Err(anyhow!("Missing xdg_wm_base")); }

        info!("Wayland: Event loop starting");

        let wayland_fd = conn.backend().poll_fd().as_raw_fd();
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

        // Main Loop
        loop {
            // 1. Flush buffers
            let _ = conn.flush();

            // 2. Prepare read
            let guard = conn.prepare_read();

            // 3. Poll
            let ret = unsafe { libc::poll(poll_fds.as_mut_ptr(), 2, -1) };
            if ret < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(anyhow!("Poll error: {}", err));
            }

            // 4. Read Wayland events
            if let Some(guard) = guard {
                if poll_fds[0].revents & libc::POLLIN != 0 {
                    if let Err(e) = guard.read() {
                        error!("Wayland read error: {:?}", e);
                    }
                } else {
                    // Drop guard to cancel read
                }
            }

            // 5. Dispatch Wayland events
            if let Err(e) = event_queue.dispatch_pending(&mut state) {
                error!("Wayland dispatch error: {}", e);
                break;
            }

            // 6. Handle Commands
            if poll_fds[1].revents & libc::POLLIN != 0 {
                // Drain pipe
                let mut buf = [0u8; 128];
                unsafe { libc::read(pipe_fd, buf.as_mut_ptr() as *mut c_void, 128) };

                // Process pending commands
                while let Ok(cmd) = run_state.cmd_rx.try_recv() {
                    state.handle_command(cmd);
                }
            }

            if !state.running {
                break;
            }
        }

        info!("Wayland: Event loop stopped");
        Ok(())
    }
}

// --- Helpers ---

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

// --- Manual Shm Map (replacing memmap2) ---

struct ShmMap {
    ptr: *mut u8,
    len: usize,
}

impl Drop for ShmMap {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe { libc::munmap(self.ptr as *mut c_void, self.len) };
        }
    }
}

// --- Wayland State ---

struct WaylandState {
    conn: Connection,
    qh: QueueHandle<WaylandState>,
    engine_tx: EngineSender<Bgra>,
    running: bool,

    // Globals
    compositor: Option<wl_compositor::WlCompositor>,
    shm: Option<wl_shm::WlShm>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    seat: Option<wl_seat::WlSeat>,

    // Window
    surface: Option<wl_surface::WlSurface>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    toplevel: Option<xdg_toplevel::XdgToplevel>,
    window_id: WindowId,
    configured: bool,

    // Dimensions
    width: u32,
    height: u32,

    // Input
    pointer: Option<wl_pointer::WlPointer>,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    cursor_theme: Option<CursorTheme>,
    cursor_surface: Option<wl_surface::WlSurface>,
    cursor_slot: Option<Slot>,
    xkb_context: xkb::Context,
    xkb_state: Option<xkb::State>,
    modifiers: Modifiers,

    // Rendering
    buffers: VecDeque<Slot>,
    pending_frame_callback: Option<wl_callback::WlCallback>,
    pending_present_complete: Option<Frame<Bgra>>,
}

struct Slot {
    buffer: wl_buffer::WlBuffer,
    pool: wl_shm_pool::WlShmPool,
    map: ShmMap,
    width: u32,
    height: u32,
    free: bool,
}

impl WaylandState {
    fn new(
        conn: Connection,
        qh: QueueHandle<WaylandState>,
        engine_tx: EngineSender<Bgra>,
    ) -> Result<Self> {
        Ok(Self {
            conn,
            qh,
            engine_tx,
            running: true,
            compositor: None,
            shm: None,
            wm_base: None,
            seat: None,
            surface: None,
            xdg_surface: None,
            toplevel: None,
            window_id: WindowId(0),
            configured: false,
            width: 800,
            height: 600,
            pointer: None,
            keyboard: None,
            cursor_theme: None,
            cursor_surface: None,
            cursor_slot: None,
            xkb_context: xkb::Context::new(xkb::CONTEXT_NO_FLAGS),
            xkb_state: None,
            modifiers: Modifiers::empty(),
            buffers: VecDeque::new(),
            pending_frame_callback: None,
            pending_present_complete: None,
        })
    }

    fn handle_command(&mut self, cmd: DriverCommand<Bgra>) {
        match cmd {
            DriverCommand::CreateWindow { id, width, height, title } => {
                self.create_window(id, width, height, title);
            }
            DriverCommand::Shutdown => {
                info!("Wayland: Shutdown requested");
                self.running = false;
            }
            DriverCommand::Present { frame, .. } => {
                self.present_frame(frame);
            }
            DriverCommand::SetTitle { title, .. } => {
                if let Some(toplevel) = &self.toplevel {
                    toplevel.set_title(title);
                }
            }
            DriverCommand::SetSize { width, height, .. } => {
                self.width = width;
                self.height = height;
            }
            #[allow(deprecated)]
            DriverCommand::Configure(_) => {}
            DriverCommand::DestroyWindow { .. } => {
                self.running = false;
            }
            _ => {
                trace!("Wayland: Unimplemented command {:?}", cmd);
            }
        }
    }

    fn create_window(&mut self, id: WindowId, width: u32, height: u32, title: String) {
        if self.surface.is_some() {
            return;
        }

        let compositor = match &self.compositor {
            Some(c) => c,
            None => { error!("Cannot create window: no compositor"); return; }
        };
        let wm_base = match &self.wm_base {
            Some(w) => w,
            None => { error!("Cannot create window: no wm_base"); return; }
        };

        self.window_id = id;
        self.width = width;
        self.height = height;

        let surface = compositor.create_surface(&self.qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &self.qh, ());
        let toplevel = xdg_surface.get_toplevel(&self.qh, ());

        toplevel.set_title(title);
        toplevel.set_app_id("pixelflow".to_string());

        surface.commit();

        self.surface = Some(surface);
        self.xdg_surface = Some(xdg_surface);
        self.toplevel = Some(toplevel);

        info!("Wayland: Surface created");
    }

    fn present_frame(&mut self, frame: Frame<Bgra>) {
        if self.surface.is_none() {
            let _ = self.engine_tx.send(EngineCommand::PresentComplete(frame));
            return;
        }

        // Clean up stale free buffers
        self.buffers.retain(|slot| {
            if slot.free && (slot.width != frame.width || slot.height != frame.height) {
                slot.buffer.destroy();
                slot.pool.destroy();
                false
            } else {
                true
            }
        });

        let slot_idx = match self.get_free_slot(frame.width, frame.height) {
            Ok(idx) => idx,
            Err(e) => {
                error!("Failed to create buffer: {}", e);
                let _ = self.engine_tx.send(EngineCommand::PresentComplete(frame));
                return;
            }
        };
        let slot = &mut self.buffers[slot_idx];

        let len = (frame.width * frame.height * 4) as usize;
        if slot.map.len < len {
            error!("Buffer too small!");
            let _ = self.engine_tx.send(EngineCommand::PresentComplete(frame));
            return;
        }

        unsafe {
            let src_ptr = frame.data.as_ptr() as *const u8;
            let dst_ptr = slot.map.ptr;
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);
        }

        let surface = self.surface.as_ref().unwrap();
        surface.attach(Some(&slot.buffer), 0, 0);
        surface.damage(0, 0, frame.width as i32, frame.height as i32);

        let callback = surface.frame(&self.qh, ());
        self.pending_frame_callback = Some(callback);

        surface.commit();

        slot.free = false;
        self.pending_present_complete = Some(frame);
    }

    fn get_free_slot(&mut self, width: u32, height: u32) -> Result<usize> {
        for (i, slot) in self.buffers.iter().enumerate() {
            if slot.free && slot.width == width && slot.height == height {
                return Ok(i);
            }
        }

        let slot = self.create_shm_buffer(width, height)?;
        self.buffers.push_back(slot);
        Ok(self.buffers.len() - 1)
    }

    fn create_shm_buffer(&self, width: u32, height: u32) -> Result<Slot> {
        let shm = self.shm.as_ref().ok_or_else(|| anyhow!("No SHM global"))?;
        let size = width * height * 4;

        let fd = create_memfd(size as usize).context("Failed to create memfd")?;

        let pool = shm.create_pool(fd.as_fd(), size as i32, &self.qh, ());
        let buffer = pool.create_buffer(0, width as i32, height as i32, (width * 4) as i32, SHM_FORMAT, &self.qh, ());

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }

        Ok(Slot {
            buffer,
            pool,
            map: ShmMap { ptr: ptr as *mut u8, len: size as usize },
            width,
            height,
            free: true,
        })
    }
}

// --- Dispatch Implementations ---

impl Dispatch<wl_registry::WlRegistry, ()> for WaylandState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            match interface.as_str() {
                "wl_compositor" => {
                    let compositor = registry.bind::<wl_compositor::WlCompositor, _, _>(name, 1.min(version), qh, ());
                    state.compositor = Some(compositor);
                }
                "wl_shm" => {
                    let shm = registry.bind::<wl_shm::WlShm, _, _>(name, 1.min(version), qh, ());
                    state.shm = Some(shm);
                }
                "xdg_wm_base" => {
                    let wm_base = registry.bind::<xdg_wm_base::XdgWmBase, _, _>(name, 1.min(version), qh, ());
                    state.wm_base = Some(wm_base);
                }
                "wl_seat" => {
                    let seat = registry.bind::<wl_seat::WlSeat, _, _>(name, 1.min(version), qh, ());
                    state.seat = Some(seat);
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_compositor::WlCompositor, ()> for WaylandState {
    fn event(
        _: &mut Self,
        _: &wl_compositor::WlCompositor,
        _: wl_compositor::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<wl_shm::WlShm, ()> for WaylandState {
    fn event(
        _: &mut Self,
        _: &wl_shm::WlShm,
        _: wl_shm::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for WaylandState {
    fn event(
        _: &mut Self,
        _: &wl_shm_pool::WlShmPool,
        _: wl_shm_pool::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<wl_buffer::WlBuffer, ()> for WaylandState {
    fn event(
        state: &mut Self,
        buffer: &wl_buffer::WlBuffer,
        event: wl_buffer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_buffer::Event::Release = event {
            for slot in &mut state.buffers {
                if &slot.buffer == buffer {
                    slot.free = true;
                    break;
                }
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for WaylandState {
    fn event(
        _: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for WaylandState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);

            if !state.configured {
                state.configured = true;
                let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::WindowCreated {
                    id: state.window_id,
                    width_px: state.width,
                    height_px: state.height,
                    scale: 1.0,
                }));
            }
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for WaylandState {
    fn event(
        state: &mut Self,
        _: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure { width, height, .. } => {
                if width > 0 && height > 0 {
                    state.width = width as u32;
                    state.height = height as u32;

                    let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::Resized {
                        id: state.window_id,
                        width_px: width as u32,
                        height_px: height as u32,
                    }));
                }
            }
            xdg_toplevel::Event::Close => {
                let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::CloseRequested {
                    id: state.window_id,
                }));
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_surface::WlSurface, ()> for WaylandState {
    fn event(
        _: &mut Self,
        _: &wl_surface::WlSurface,
        _: wl_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<wl_callback::WlCallback, ()> for WaylandState {
    fn event(
        state: &mut Self,
        _: &wl_callback::WlCallback,
        event: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            if let Some(frame) = state.pending_present_complete.take() {
                let _ = state.engine_tx.send(EngineCommand::PresentComplete(frame));
            }
            state.pending_frame_callback = None;
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for WaylandState {
    fn event(
        state: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities { capabilities } = event {
            if let WEnum::Value(caps) = capabilities {
                 if caps.contains(wl_seat::Capability::Pointer) && state.pointer.is_none() {
                    state.pointer = Some(seat.get_pointer(qh, ()));
                }
                if caps.contains(wl_seat::Capability::Keyboard) && state.keyboard.is_none() {
                    state.keyboard = Some(seat.get_keyboard(qh, ()));
                }
            }
        }
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for WaylandState {
    fn event(
        state: &mut Self,
        pointer: &wl_pointer::WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let id = state.window_id;
        match event {
            wl_pointer::Event::Enter { serial, surface_x: _, surface_y: _, surface: _ } => {
                // Set cursor
                if let (Some(theme), Some(_cursor_surface)) = (&mut state.cursor_theme, &state.cursor_surface) {
                    if let Some(cursor) = theme.get_cursor("left_ptr") {
                        let image = &cursor[0];
                        let (_width, _height) = image.dimensions();
                        let (_hx, _hy) = image.hotspot();

                        // FIXME: wayland-cursor 0.31 CursorImageBuffer data access is opaque/tricky.
                        // Disabling cursor set for now to ensure compilation.
                        // Future work: use image.chunk() if available or fix data access.
                        /*
                        if let Some(old) = state.cursor_slot.take() {
                            old.buffer.destroy();
                            old.pool.destroy();
                        }

                        if let Ok(mut slot) = state.create_shm_buffer(width, height) {
                             // Need access to pixel data from image
                             // let data: &[u8] = ...;
                             // ... copy and attach ...
                             // pointer.set_cursor(serial, Some(cursor_surface), hx as i32, hy as i32);
                             // state.cursor_slot = Some(slot);
                        }
                        */
                        // Suppress unused warnings
                        let _ = (serial, pointer);
                    }
                }
            }
            wl_pointer::Event::Motion { surface_x, surface_y, .. } => {
                let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::MouseMove {
                    id,
                    x: surface_x as i32,
                    y: surface_y as i32,
                    modifiers: state.modifiers,
                }));
            }
            wl_pointer::Event::Button { button, state: bstate, .. } => {
                let btn = match button {
                    272 => 1,
                    274 => 2,
                    273 => 3,
                    b => (b - 272 + 1) as u8,
                };

                let event = match bstate {
                    WEnum::Value(wl_pointer::ButtonState::Pressed) => DisplayEvent::MouseButtonPress {
                        id,
                        button: btn,
                        x: 0,
                        y: 0,
                        modifiers: state.modifiers,
                    },
                    WEnum::Value(wl_pointer::ButtonState::Released) => DisplayEvent::MouseButtonRelease {
                        id,
                        button: btn,
                        x: 0,
                        y: 0,
                        modifiers: state.modifiers,
                    },
                    _ => return,
                };
                let _ = state.engine_tx.send(EngineCommand::DisplayEvent(event));
            }
            wl_pointer::Event::Axis { value, axis, .. } => {
                let (dx, dy) = match axis {
                    WEnum::Value(wl_pointer::Axis::VerticalScroll) => (0.0, -(value as f32) / 10.0),
                    WEnum::Value(wl_pointer::Axis::HorizontalScroll) => (-(value as f32) / 10.0, 0.0),
                    _ => (0.0, 0.0),
                };
                let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::MouseScroll {
                    id,
                    dx,
                    dy,
                    x: 0,
                    y: 0,
                    modifiers: state.modifiers,
                }));
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for WaylandState {
    fn event(
        state: &mut Self,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let id = state.window_id;
        match event {
            wl_keyboard::Event::Keymap { format, fd, size } => {
                if format == WEnum::Value(wl_keyboard::KeymapFormat::XkbV1) {
                    unsafe {
                        let ptr = libc::mmap(
                            std::ptr::null_mut(),
                            size as usize,
                            libc::PROT_READ,
                            libc::MAP_PRIVATE,
                            fd.as_raw_fd(),
                            0
                        );
                        if ptr != libc::MAP_FAILED {
                            let s = std::slice::from_raw_parts(ptr as *const u8, size as usize);
                            if let Ok(s) = std::str::from_utf8(s) {
                                let s = s.trim_end_matches('\0');
                                if let Some(keymap) = xkb::Keymap::new_from_string(
                                    &state.xkb_context,
                                    s.to_string(),
                                    xkb::KEYMAP_FORMAT_TEXT_V1,
                                    xkb::KEYMAP_COMPILE_NO_FLAGS,
                                ) {
                                    state.xkb_state = Some(xkb::State::new(&keymap));
                                }
                            }
                            libc::munmap(ptr, size as usize);
                        }
                    }
                }
            }
            wl_keyboard::Event::Enter { .. } => {
                let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::FocusGained { id }));
            }
            wl_keyboard::Event::Leave { .. } => {
                let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::FocusLost { id }));
            }
            wl_keyboard::Event::Modifiers { mods_depressed, mods_latched, mods_locked, group, .. } => {
                if let Some(xkb_state) = &mut state.xkb_state {
                    xkb_state.update_mask(mods_depressed, mods_latched, mods_locked, 0, 0, group);

                    let mut mods = Modifiers::empty();
                    if xkb_state.mod_name_is_active(xkb::MOD_NAME_SHIFT, xkb::STATE_MODS_EFFECTIVE) {
                        mods.insert(Modifiers::SHIFT);
                    }
                    if xkb_state.mod_name_is_active(xkb::MOD_NAME_CTRL, xkb::STATE_MODS_EFFECTIVE) {
                        mods.insert(Modifiers::CONTROL);
                    }
                    if xkb_state.mod_name_is_active(xkb::MOD_NAME_ALT, xkb::STATE_MODS_EFFECTIVE) {
                        mods.insert(Modifiers::ALT);
                    }
                    if xkb_state.mod_name_is_active(xkb::MOD_NAME_LOGO, xkb::STATE_MODS_EFFECTIVE) {
                        mods.insert(Modifiers::SUPER);
                    }
                    state.modifiers = mods;
                }
            }
            wl_keyboard::Event::Key { key, state: kstate, .. } => {
                if let Some(xkb_state) = &mut state.xkb_state {
                    let keycode = key + 8;

                    if kstate == WEnum::Value(wl_keyboard::KeyState::Pressed) {
                        let text = xkb_state.key_get_utf8(keycode.into());
                        let text = if text.is_empty() { None } else { Some(text) };

                        let keysym = xkb_state.key_get_one_sym(keycode.into());
                        let symbol = xkeysym_to_keysymbol(keysym);

                        let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::Key {
                            id,
                            symbol,
                            modifiers: state.modifiers,
                            text,
                        }));
                    }
                }
            }
            _ => {}
        }
    }
}

// --- Key Mapping ---

fn xkeysym_to_keysymbol(keysym: xkb::Keysym) -> KeySymbol {
    use xkbcommon::xkb::keysyms::*;
    match u32::from(keysym) {
         KEY_Return => KeySymbol::Enter,
         KEY_BackSpace => KeySymbol::Backspace,
         KEY_Tab => KeySymbol::Tab,
         KEY_Escape => KeySymbol::Escape,
         KEY_Home => KeySymbol::Home,
         KEY_Left => KeySymbol::Left,
         KEY_Up => KeySymbol::Up,
         KEY_Right => KeySymbol::Right,
         KEY_Down => KeySymbol::Down,
         KEY_Page_Up => KeySymbol::PageUp,
         KEY_Page_Down => KeySymbol::PageDown,
         KEY_End => KeySymbol::End,
         KEY_Insert => KeySymbol::Insert,
         KEY_Delete => KeySymbol::Delete,
         KEY_F1 => KeySymbol::F1,
         KEY_F2 => KeySymbol::F2,
         KEY_F3 => KeySymbol::F3,
         KEY_F4 => KeySymbol::F4,
         KEY_F5 => KeySymbol::F5,
         KEY_F6 => KeySymbol::F6,
         KEY_F7 => KeySymbol::F7,
         KEY_F8 => KeySymbol::F8,
         KEY_F9 => KeySymbol::F9,
         KEY_F10 => KeySymbol::F10,
         KEY_F11 => KeySymbol::F11,
         KEY_F12 => KeySymbol::F12,
         KEY_Shift_L | KEY_Shift_R => KeySymbol::Shift,
         KEY_Control_L | KEY_Control_R => KeySymbol::Control,
         KEY_Alt_L | KEY_Alt_R => KeySymbol::Alt,
         KEY_Super_L | KEY_Super_R => KeySymbol::Super,
         _ => KeySymbol::Unknown,
    }
}

fn create_memfd(size: usize) -> Result<OwnedFd> {
    let name = CString::new("pixelflow-shm")?;
    let fd = unsafe {
        libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC)
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    if unsafe { libc::ftruncate(fd, size as i64) } < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(owned)
}
