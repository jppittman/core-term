//! Wayland DisplayDriver implementation using raw FFI.
//!
//! Replaces wayland-client to remove downcast-rs dependency.
//! Implements XDG Shell protocol manually.

use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayEvent, WindowId};
use crate::input::{KeySymbol, Modifiers};
use anyhow::{anyhow, Result};
use log::{info};
use pixelflow_render::color::Bgra;
use pixelflow_render::Frame;
use std::collections::VecDeque;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use wayland_sys::client::*;
use wayland_sys::common::*;
use xkbcommon::xkb;

// --- Type Aliases for Opaque Proxies ---
#[allow(non_camel_case_types)]
type wl_registry = wl_proxy;
#[allow(non_camel_case_types)]
type wl_compositor = wl_proxy;
#[allow(non_camel_case_types)]
type wl_shm = wl_proxy;
#[allow(non_camel_case_types)]
type wl_shm_pool = wl_proxy;
#[allow(non_camel_case_types)]
type wl_buffer = wl_proxy;
#[allow(non_camel_case_types)]
type wl_seat = wl_proxy;
#[allow(non_camel_case_types)]
type wl_surface = wl_proxy;
#[allow(non_camel_case_types)]
type wl_pointer = wl_proxy;
#[allow(non_camel_case_types)]
type wl_keyboard = wl_proxy;
#[allow(non_camel_case_types)]
type wl_callback = wl_proxy;
#[allow(non_camel_case_types)]
type wl_array = wayland_sys::common::wl_array;

// --- Constants ---
const WL_SHM_FORMAT_ARGB8888: u32 = 0;

// --- Externs for core interfaces ---
extern "C" {
    pub static wl_registry_interface: wl_interface;
    pub static wl_compositor_interface: wl_interface;
    pub static wl_shm_interface: wl_interface;
    pub static wl_shm_pool_interface: wl_interface;
    pub static wl_buffer_interface: wl_interface;
    pub static wl_seat_interface: wl_interface;
    pub static wl_surface_interface: wl_interface;
    pub static wl_pointer_interface: wl_interface;
    pub static wl_keyboard_interface: wl_interface;
    pub static wl_callback_interface: wl_interface;
    pub static wl_output_interface: wl_interface;
    pub static wl_region_interface: wl_interface;
}

// --- Driver Struct ---

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
        unsafe { init_interfaces(); }

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

// --- Wayland State ---

struct WaylandState {
    #[allow(dead_code)]
    display: *mut wl_display,
    engine_tx: EngineSender<Bgra>,
    running: bool,

    // Globals
    compositor: *mut wl_compositor,
    shm: *mut wl_shm,
    wm_base: *mut wl_proxy, // xdg_wm_base
    seat: *mut wl_seat,

    // Window
    surface: *mut wl_surface,
    xdg_surface: *mut wl_proxy,
    xdg_toplevel: *mut wl_proxy,
    window_id: WindowId,
    configured: bool,

    // Dimensions
    width: u32,
    height: u32,

    // Input
    pointer: *mut wl_pointer,
    keyboard: *mut wl_keyboard,
    xkb_context: xkb::Context,
    xkb_state: Option<xkb::State>,
    modifiers: Modifiers,

    // Buffers
    buffers: VecDeque<Slot>,
}

struct Slot {
    buffer: *mut wl_buffer,
    #[allow(dead_code)]
    pool: *mut wl_shm_pool,
    ptr: *mut u8,
    size: usize,
    width: u32,
    height: u32,
    free: bool,
}

impl WaylandState {
    fn new(display: *mut wl_display, engine_tx: EngineSender<Bgra>) -> Result<Self> {
        Ok(Self {
            display,
            engine_tx,
            running: true,
            compositor: ptr::null_mut(),
            shm: ptr::null_mut(),
            wm_base: ptr::null_mut(),
            seat: ptr::null_mut(),
            surface: ptr::null_mut(),
            xdg_surface: ptr::null_mut(),
            xdg_toplevel: ptr::null_mut(),
            window_id: WindowId(0),
            configured: false,
            width: 800,
            height: 600,
            pointer: ptr::null_mut(),
            keyboard: ptr::null_mut(),
            xkb_context: xkb::Context::new(xkb::CONTEXT_NO_FLAGS),
            xkb_state: None,
            modifiers: Modifiers::empty(),
            buffers: VecDeque::new(),
        })
    }

    fn handle_command(&mut self, cmd: DriverCommand<Bgra>) {
        match cmd {
            DriverCommand::CreateWindow { id, width, height, title } => {
                self.create_window(id, width, height, &title);
            }
            DriverCommand::Shutdown => {
                self.running = false;
            }
            DriverCommand::Present { frame, .. } => {
                self.present_frame(frame);
            }
            DriverCommand::SetTitle { title, .. } => {
                if !self.xdg_toplevel.is_null() {
                    unsafe {
                        let c_title = CString::new(title).unwrap_or_default();
                        // xdg_toplevel.set_title is opcode 2
                        wl_proxy_marshal(self.xdg_toplevel, 2, c_title.as_ptr());
                    }
                }
            }
            DriverCommand::DestroyWindow { .. } => {
                self.running = false;
            }
            _ => {}
        }
    }

    fn create_window(&mut self, id: WindowId, width: u32, height: u32, title: &str) {
        if !self.surface.is_null() { return; }
        if self.compositor.is_null() || self.wm_base.is_null() { return; }

        unsafe {
            self.window_id = id;
            self.width = width;
            self.height = height;

            // wl_compositor.create_surface (opcode 0)
            let surface = wl_proxy_marshal_constructor(
                self.compositor as *mut wl_proxy,
                0,
                &wl_surface_interface
            ) as *mut wl_surface;
            self.surface = surface;

            // xdg_wm_base.get_xdg_surface (opcode 2)
            let xdg_surface = wl_proxy_marshal_constructor(
                self.wm_base,
                2,
                &xdg_surface_interface,
                surface
            );
            self.xdg_surface = xdg_surface;
            wl_proxy_add_listener(xdg_surface, &XDG_SURFACE_LISTENER as *const _ as *mut extern "C" fn(), self as *mut _ as *mut c_void);

            // xdg_surface.get_toplevel (opcode 1)
            let toplevel = wl_proxy_marshal_constructor(
                xdg_surface,
                1,
                &xdg_toplevel_interface
            );
            self.xdg_toplevel = toplevel;
            wl_proxy_add_listener(toplevel, &XDG_TOPLEVEL_LISTENER as *const _ as *mut extern "C" fn(), self as *mut _ as *mut c_void);

            let c_title = CString::new(title).unwrap_or_default();
            // xdg_toplevel.set_title (opcode 2)
            wl_proxy_marshal(toplevel, 2, c_title.as_ptr());

            let c_app_id = CString::new("pixelflow").unwrap();
            // xdg_toplevel.set_app_id (opcode 3)
            wl_proxy_marshal(toplevel, 3, c_app_id.as_ptr());

            // wl_surface.commit (opcode 6)
            wl_proxy_marshal(surface as *mut wl_proxy, 6);
        }
    }

    fn present_frame(&mut self, frame: Frame<Bgra>) {
        if self.surface.is_null() {
            let _ = self.engine_tx.send(EngineCommand::PresentComplete(frame));
            return;
        }

        let slot_idx = match self.get_free_slot(frame.width, frame.height) {
            Ok(idx) => idx,
            Err(_) => {
                let _ = self.engine_tx.send(EngineCommand::PresentComplete(frame));
                return;
            }
        };

        let slot = &mut self.buffers[slot_idx];
        let len = (frame.width * frame.height * 4) as usize;

        unsafe {
            ptr::copy_nonoverlapping(frame.data.as_ptr() as *const u8, slot.ptr, len);

            // wl_surface.attach (opcode 1)
            wl_proxy_marshal(self.surface as *mut wl_proxy, 1, slot.buffer, 0, 0);

            // wl_surface.damage (opcode 2)
            wl_proxy_marshal(self.surface as *mut wl_proxy, 2, 0, 0, frame.width as i32, frame.height as i32);

            // wl_surface.frame (opcode 3) -> new_id wl_callback
            let callback = wl_proxy_marshal_constructor(
                self.surface as *mut wl_proxy,
                3,
                &wl_callback_interface
            ) as *mut wl_callback;

            // Pack frame into callback data to send back on done
            let frame_box = Box::new(FrameContext { tx: self.engine_tx.clone(), frame });
            wl_proxy_add_listener(
                callback as *mut wl_proxy,
                &FRAME_LISTENER as *const _ as *mut extern "C" fn(),
                Box::into_raw(frame_box) as *mut c_void
            );

            // wl_surface.commit (opcode 6)
            wl_proxy_marshal(self.surface as *mut wl_proxy, 6);

            slot.free = false;
        }
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
        let size = (width * height * 4) as usize;
        let fd = create_memfd(size)?;

        unsafe {
            // wl_shm.create_pool (opcode 0)
            let pool = wl_proxy_marshal_constructor(
                self.shm as *mut wl_proxy,
                0,
                &wl_shm_pool_interface,
                fd.as_raw_fd(),
                size as i32
            ) as *mut wl_shm_pool;

            // wl_shm_pool.create_buffer (opcode 0)
            let buffer = wl_proxy_marshal_constructor(
                pool as *mut wl_proxy,
                0,
                &wl_buffer_interface,
                0, // offset
                width as i32,
                height as i32,
                (width * 4) as i32, // stride
                WL_SHM_FORMAT_ARGB8888
            ) as *mut wl_buffer;

            wl_proxy_add_listener(buffer as *mut wl_proxy, &BUFFER_LISTENER as *const _ as *mut extern "C" fn(), self as *const _ as *mut c_void);

            let ptr = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0
            );

            if ptr == libc::MAP_FAILED {
                return Err(anyhow!("mmap failed"));
            }

            Ok(Slot {
                buffer,
                pool,
                ptr: ptr as *mut u8,
                size,
                width,
                height,
                free: true,
            })
        }
    }
}

// --- Listeners ---

extern "C" fn registry_global(data: *mut c_void, registry: *mut wl_registry, name: u32, interface: *const c_char, version: u32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        let interface_cstr = CStr::from_ptr(interface);
        let interface_str = interface_cstr.to_string_lossy();

        if interface_str == "wl_compositor" {
            state.compositor = wl_proxy_marshal_constructor(
                registry as *mut wl_proxy,
                0, // bind
                &wl_compositor_interface,
                name,
                1.min(version)
            ) as *mut wl_compositor;
        } else if interface_str == "wl_shm" {
            state.shm = wl_proxy_marshal_constructor(
                registry as *mut wl_proxy,
                0,
                &wl_shm_interface,
                name,
                1.min(version)
            ) as *mut wl_shm;
        } else if interface_str == "xdg_wm_base" {
            state.wm_base = wl_proxy_marshal_constructor(
                registry as *mut wl_proxy,
                0,
                &xdg_wm_base_interface,
                name,
                1.min(version)
            );
            wl_proxy_add_listener(state.wm_base, &XDG_WM_BASE_LISTENER as *const _ as *mut extern "C" fn(), state as *mut _ as *mut c_void);
        } else if interface_str == "wl_seat" {
             state.seat = wl_proxy_marshal_constructor(
                registry as *mut wl_proxy,
                0,
                &wl_seat_interface,
                name,
                1.min(version)
            ) as *mut wl_seat;
            wl_proxy_add_listener(state.seat as *mut wl_proxy, &SEAT_LISTENER as *const _ as *mut extern "C" fn(), state as *mut _ as *mut c_void);
        }
    }
}

extern "C" fn registry_global_remove(_data: *mut c_void, _registry: *mut wl_registry, _name: u32) {}

#[repr(C)]
struct wl_registry_listener {
    global: extern "C" fn(*mut c_void, *mut wl_registry, u32, *const c_char, u32),
    global_remove: extern "C" fn(*mut c_void, *mut wl_registry, u32),
}

static REGISTRY_LISTENER: wl_registry_listener = wl_registry_listener {
    global: registry_global,
    global_remove: registry_global_remove,
};

// --- XDG WM Base ---

extern "C" fn xdg_wm_base_ping(_data: *mut c_void, wm_base: *mut wl_proxy, serial: u32) {
    unsafe {
        // pong opcode 3
        wl_proxy_marshal(wm_base, 3, serial);
    }
}

// Struct layout must match wayland-scanner output
#[repr(C)]
struct xdg_wm_base_listener {
    ping: extern "C" fn(*mut c_void, *mut wl_proxy, u32),
}

static XDG_WM_BASE_LISTENER: xdg_wm_base_listener = xdg_wm_base_listener {
    ping: xdg_wm_base_ping,
};

// --- XDG Surface ---

extern "C" fn xdg_surface_configure(data: *mut c_void, xdg_surface: *mut wl_proxy, serial: u32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        // ack_configure opcode 4
        wl_proxy_marshal(xdg_surface, 4, serial);

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

#[repr(C)]
struct xdg_surface_listener {
    configure: extern "C" fn(*mut c_void, *mut wl_proxy, u32),
}

static XDG_SURFACE_LISTENER: xdg_surface_listener = xdg_surface_listener {
    configure: xdg_surface_configure,
};

// --- XDG Toplevel ---

extern "C" fn xdg_toplevel_configure(data: *mut c_void, _toplevel: *mut wl_proxy, width: i32, height: i32, _states: *mut wl_array) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
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
}

extern "C" fn xdg_toplevel_close(data: *mut c_void, _toplevel: *mut wl_proxy) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::CloseRequested {
            id: state.window_id,
        }));
    }
}

#[repr(C)]
struct xdg_toplevel_listener {
    configure: extern "C" fn(*mut c_void, *mut wl_proxy, i32, i32, *mut wl_array),
    close: extern "C" fn(*mut c_void, *mut wl_proxy),
}

static XDG_TOPLEVEL_LISTENER: xdg_toplevel_listener = xdg_toplevel_listener {
    configure: xdg_toplevel_configure,
    close: xdg_toplevel_close,
};

// --- Buffer ---

extern "C" fn buffer_release(data: *mut c_void, buffer: *mut wl_buffer) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        for slot in &mut state.buffers {
            if slot.buffer == buffer {
                slot.free = true;
                break;
            }
        }
    }
}

#[repr(C)]
struct wl_buffer_listener {
    release: extern "C" fn(*mut c_void, *mut wl_buffer),
}

static BUFFER_LISTENER: wl_buffer_listener = wl_buffer_listener {
    release: buffer_release,
};

// --- Frame Callback ---

extern "C" fn frame_done_wrapper(data: *mut c_void, callback: *mut wl_callback, _time: u32) {
    unsafe {
        let ctx = Box::from_raw(data as *mut FrameContext);
        let _ = ctx.tx.send(EngineCommand::PresentComplete(ctx.frame));
        wl_proxy_destroy(callback as *mut wl_proxy);
    }
}

struct FrameContext {
    tx: EngineSender<Bgra>,
    frame: Frame<Bgra>,
}

#[repr(C)]
struct wl_callback_listener {
    done: extern "C" fn(*mut c_void, *mut wl_callback, u32),
}

static FRAME_LISTENER: wl_callback_listener = wl_callback_listener {
    done: frame_done_wrapper,
};

// --- Seat ---

extern "C" fn seat_capabilities(data: *mut c_void, seat: *mut wl_seat, caps: u32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);

        if (caps & 1) != 0 && state.pointer.is_null() {
            // pointer
            let pointer = wl_proxy_marshal_constructor(seat as *mut wl_proxy, 0, &wl_pointer_interface) as *mut wl_pointer;
            state.pointer = pointer;
            wl_proxy_add_listener(pointer as *mut wl_proxy, &POINTER_LISTENER as *const _ as *mut extern "C" fn(), state as *mut _ as *mut c_void);
        }

        if (caps & 2) != 0 && state.keyboard.is_null() {
            // keyboard
            let keyboard = wl_proxy_marshal_constructor(seat as *mut wl_proxy, 1, &wl_keyboard_interface) as *mut wl_keyboard;
            state.keyboard = keyboard;
            wl_proxy_add_listener(keyboard as *mut wl_proxy, &KEYBOARD_LISTENER as *const _ as *mut extern "C" fn(), state as *mut _ as *mut c_void);
        }
    }
}

extern "C" fn seat_name(_data: *mut c_void, _seat: *mut wl_seat, _name: *const c_char) {}

#[repr(C)]
struct wl_seat_listener {
    capabilities: extern "C" fn(*mut c_void, *mut wl_seat, u32),
    name: extern "C" fn(*mut c_void, *mut wl_seat, *const c_char),
}

static SEAT_LISTENER: wl_seat_listener = wl_seat_listener {
    capabilities: seat_capabilities,
    name: seat_name,
};

// --- Pointer ---

extern "C" fn pointer_enter(_data: *mut c_void, _ptr: *mut wl_pointer, _serial: u32, _surf: *mut wl_surface, _x: i32, _y: i32) {}
extern "C" fn pointer_leave(_data: *mut c_void, _ptr: *mut wl_pointer, _serial: u32, _surf: *mut wl_surface) {}

extern "C" fn pointer_motion(data: *mut c_void, _ptr: *mut wl_pointer, _time: u32, x: i32, y: i32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::MouseMove {
            id: state.window_id,
            x: x >> 8, // wayland sends fixed point 24.8
            y: y >> 8,
            modifiers: state.modifiers,
        }));
    }
}

extern "C" fn pointer_button(data: *mut c_void, _ptr: *mut wl_pointer, _serial: u32, _time: u32, button: u32, state_w: u32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        // linux input codes: 272=left, 273=right, 274=middle
        let btn = match button {
            272 => 1,
            274 => 2,
            273 => 3,
            b => (b - 272 + 1) as u8,
        };
        let evt = if state_w == 1 { // pressed
             DisplayEvent::MouseButtonPress { id: state.window_id, button: btn, x: 0, y: 0, modifiers: state.modifiers }
        } else {
             DisplayEvent::MouseButtonRelease { id: state.window_id, button: btn, x: 0, y: 0, modifiers: state.modifiers }
        };
        let _ = state.engine_tx.send(EngineCommand::DisplayEvent(evt));
    }
}

extern "C" fn pointer_axis(data: *mut c_void, _ptr: *mut wl_pointer, _time: u32, axis: u32, value: i32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        let val = (value as f32) / 256.0;
        let (dx, dy) = if axis == 0 { (0.0, -val/10.0) } else { (-val/10.0, 0.0) };
        let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::MouseScroll {
            id: state.window_id, dx, dy, x: 0, y: 0, modifiers: state.modifiers
        }));
    }
}

extern "C" fn pointer_frame(_data: *mut c_void, _ptr: *mut wl_pointer) {}
extern "C" fn pointer_axis_source(_data: *mut c_void, _ptr: *mut wl_pointer, _source: u32) {}
extern "C" fn pointer_axis_stop(_data: *mut c_void, _ptr: *mut wl_pointer, _time: u32, _axis: u32) {}
extern "C" fn pointer_axis_discrete(_data: *mut c_void, _ptr: *mut wl_pointer, _axis: u32, _discrete: i32) {}

#[repr(C)]
struct wl_pointer_listener {
    enter: extern "C" fn(*mut c_void, *mut wl_pointer, u32, *mut wl_surface, i32, i32),
    leave: extern "C" fn(*mut c_void, *mut wl_pointer, u32, *mut wl_surface),
    motion: extern "C" fn(*mut c_void, *mut wl_pointer, u32, i32, i32),
    button: extern "C" fn(*mut c_void, *mut wl_pointer, u32, u32, u32, u32),
    axis: extern "C" fn(*mut c_void, *mut wl_pointer, u32, u32, i32),
    frame: extern "C" fn(*mut c_void, *mut wl_pointer),
    axis_source: extern "C" fn(*mut c_void, *mut wl_pointer, u32),
    axis_stop: extern "C" fn(*mut c_void, *mut wl_pointer, u32, u32),
    axis_discrete: extern "C" fn(*mut c_void, *mut wl_pointer, u32, i32),
}

static POINTER_LISTENER: wl_pointer_listener = wl_pointer_listener {
    enter: pointer_enter,
    leave: pointer_leave,
    motion: pointer_motion,
    button: pointer_button,
    axis: pointer_axis,
    frame: pointer_frame,
    axis_source: pointer_axis_source,
    axis_stop: pointer_axis_stop,
    axis_discrete: pointer_axis_discrete,
};

// --- Keyboard ---

extern "C" fn keyboard_keymap(data: *mut c_void, _kbd: *mut wl_keyboard, format: u32, fd: i32, size: u32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        if format == 1 { // XKB_V1
             let ptr = libc::mmap(ptr::null_mut(), size as usize, libc::PROT_READ, libc::MAP_PRIVATE, fd, 0);
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
        libc::close(fd);
    }
}

extern "C" fn keyboard_enter(data: *mut c_void, _kbd: *mut wl_keyboard, _serial: u32, _surf: *mut wl_surface, _keys: *mut wl_array) {
     unsafe {
        let state = &mut *(data as *mut WaylandState);
        let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::FocusGained { id: state.window_id }));
     }
}

extern "C" fn keyboard_leave(data: *mut c_void, _kbd: *mut wl_keyboard, _serial: u32, _surf: *mut wl_surface) {
     unsafe {
        let state = &mut *(data as *mut WaylandState);
        let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::FocusLost { id: state.window_id }));
     }
}

#[allow(non_upper_case_globals)]
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

extern "C" fn keyboard_key(data: *mut c_void, _kbd: *mut wl_keyboard, _serial: u32, _time: u32, key: u32, state_w: u32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
        if let Some(xkb_state) = &mut state.xkb_state {
            let keycode = key + 8;
            if state_w == 1 { // pressed
                let text = xkb_state.key_get_utf8(keycode.into());
                let text = if text.is_empty() { None } else { Some(text) };

                let keysym = xkb_state.key_get_one_sym(keycode.into());
                let symbol = xkeysym_to_keysymbol(keysym);

                let _ = state.engine_tx.send(EngineCommand::DisplayEvent(DisplayEvent::Key {
                    id: state.window_id,
                    symbol,
                    modifiers: state.modifiers,
                    text,
                }));
            }
        }
    }
}

extern "C" fn keyboard_modifiers(data: *mut c_void, _kbd: *mut wl_keyboard, _serial: u32, mods_depressed: u32, mods_latched: u32, mods_locked: u32, group: u32) {
    unsafe {
        let state = &mut *(data as *mut WaylandState);
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
}

extern "C" fn keyboard_repeat_info(_data: *mut c_void, _kbd: *mut wl_keyboard, _rate: i32, _delay: i32) {}

#[repr(C)]
struct wl_keyboard_listener {
    keymap: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, i32, u32),
    enter: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, *mut wl_surface, *mut wl_array),
    leave: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, *mut wl_surface),
    key: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, u32, u32, u32),
    modifiers: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, u32, u32, u32, u32),
    repeat_info: extern "C" fn(*mut c_void, *mut wl_keyboard, i32, i32),
}

static KEYBOARD_LISTENER: wl_keyboard_listener = wl_keyboard_listener {
    keymap: keyboard_keymap,
    enter: keyboard_enter,
    leave: keyboard_leave,
    key: keyboard_key,
    modifiers: keyboard_modifiers,
    repeat_info: keyboard_repeat_info,
};


// Strings
const XDG_WM_BASE_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"xdg_wm_base\0") };
const XDG_SURFACE_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"xdg_surface\0") };
const XDG_TOPLEVEL_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"xdg_toplevel\0") };

// xdg_wm_base requests
// 0: destroy
// 1: create_positioner(new_id xdg_positioner)
// 2: get_xdg_surface(new_id xdg_surface, object surface)
// 3: pong(uint serial)

static mut XDG_WM_BASE_REQUESTS: [wl_message; 4] = [
    wl_message { name: b"destroy\0".as_ptr() as *const c_char, signature: b"\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"create_positioner\0".as_ptr() as *const c_char, signature: b"n\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"get_xdg_surface\0".as_ptr() as *const c_char, signature: b"no\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"pong\0".as_ptr() as *const c_char, signature: b"u\0".as_ptr() as *const c_char, types: ptr::null() },
];

static mut XDG_WM_BASE_EVENTS: [wl_message; 1] = [
    wl_message { name: b"ping\0".as_ptr() as *const c_char, signature: b"u\0".as_ptr() as *const c_char, types: ptr::null() },
];

// xdg_surface requests
static mut XDG_SURFACE_REQUESTS: [wl_message; 5] = [
    wl_message { name: b"destroy\0".as_ptr() as *const c_char, signature: b"\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"get_toplevel\0".as_ptr() as *const c_char, signature: b"n\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"get_popup\0".as_ptr() as *const c_char, signature: b"n?oo\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_window_geometry\0".as_ptr() as *const c_char, signature: b"iiii\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"ack_configure\0".as_ptr() as *const c_char, signature: b"u\0".as_ptr() as *const c_char, types: ptr::null() },
];

static mut XDG_SURFACE_EVENTS: [wl_message; 1] = [
    wl_message { name: b"configure\0".as_ptr() as *const c_char, signature: b"u\0".as_ptr() as *const c_char, types: ptr::null() },
];

// xdg_toplevel requests
static mut XDG_TOPLEVEL_REQUESTS: [wl_message; 14] = [
    wl_message { name: b"destroy\0".as_ptr() as *const c_char, signature: b"\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_parent\0".as_ptr() as *const c_char, signature: b"?o\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_title\0".as_ptr() as *const c_char, signature: b"s\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_app_id\0".as_ptr() as *const c_char, signature: b"s\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"show_window_menu\0".as_ptr() as *const c_char, signature: b"ouii\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"move\0".as_ptr() as *const c_char, signature: b"ou\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"resize\0".as_ptr() as *const c_char, signature: b"ouu\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_max_size\0".as_ptr() as *const c_char, signature: b"ii\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_min_size\0".as_ptr() as *const c_char, signature: b"ii\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_maximized\0".as_ptr() as *const c_char, signature: b"\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"unset_maximized\0".as_ptr() as *const c_char, signature: b"\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_fullscreen\0".as_ptr() as *const c_char, signature: b"?o\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"unset_fullscreen\0".as_ptr() as *const c_char, signature: b"\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"set_minimized\0".as_ptr() as *const c_char, signature: b"\0".as_ptr() as *const c_char, types: ptr::null() },
];

static mut XDG_TOPLEVEL_EVENTS: [wl_message; 2] = [
    wl_message { name: b"configure\0".as_ptr() as *const c_char, signature: b"iia\0".as_ptr() as *const c_char, types: ptr::null() },
    wl_message { name: b"close\0".as_ptr() as *const c_char, signature: b"\0".as_ptr() as *const c_char, types: ptr::null() },
];


// Interfaces
static mut XDG_WM_BASE_GET_XDG_SURFACE_TYPES: [*const wl_interface; 2] = [ptr::null(), ptr::null()];
static mut XDG_SURFACE_GET_TOPLEVEL_TYPES: [*const wl_interface; 1] = [ptr::null()];
static mut XDG_TOPLEVEL_SET_FULLSCREEN_TYPES: [*const wl_interface; 1] = [ptr::null()];

pub static mut xdg_wm_base_interface: wl_interface = wl_interface {
    name: XDG_WM_BASE_NAME.as_ptr(),
    version: 1,
    request_count: 4,
    requests: unsafe { XDG_WM_BASE_REQUESTS.as_ptr() },
    event_count: 1,
    events: unsafe { XDG_WM_BASE_EVENTS.as_ptr() },
};

pub static mut xdg_surface_interface: wl_interface = wl_interface {
    name: XDG_SURFACE_NAME.as_ptr(),
    version: 1,
    request_count: 5,
    requests: unsafe { XDG_SURFACE_REQUESTS.as_ptr() },
    event_count: 1,
    events: unsafe { XDG_SURFACE_EVENTS.as_ptr() },
};

pub static mut xdg_toplevel_interface: wl_interface = wl_interface {
    name: XDG_TOPLEVEL_NAME.as_ptr(),
    version: 1,
    request_count: 14,
    requests: unsafe { XDG_TOPLEVEL_REQUESTS.as_ptr() },
    event_count: 2,
    events: unsafe { XDG_TOPLEVEL_EVENTS.as_ptr() },
};

unsafe fn init_interfaces() {
    XDG_WM_BASE_GET_XDG_SURFACE_TYPES[0] = ptr::addr_of!(xdg_surface_interface);
    XDG_WM_BASE_GET_XDG_SURFACE_TYPES[1] = ptr::addr_of!(wl_surface_interface);
    XDG_WM_BASE_REQUESTS[2].types = XDG_WM_BASE_GET_XDG_SURFACE_TYPES.as_ptr();

    XDG_SURFACE_GET_TOPLEVEL_TYPES[0] = ptr::addr_of!(xdg_toplevel_interface);
    XDG_SURFACE_REQUESTS[1].types = XDG_SURFACE_GET_TOPLEVEL_TYPES.as_ptr();

    XDG_TOPLEVEL_SET_FULLSCREEN_TYPES[0] = ptr::addr_of!(wl_output_interface);
    XDG_TOPLEVEL_REQUESTS[11].types = XDG_TOPLEVEL_SET_FULLSCREEN_TYPES.as_ptr();
}
