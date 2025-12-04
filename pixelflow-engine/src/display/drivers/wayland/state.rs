//! Wayland state management.

use super::protocol::*;
use super::listeners::*;
use crate::channel::{DriverCommand, EngineCommand, EngineSender};
use crate::display::messages::{WindowId};
use crate::input::{Modifiers};
use anyhow::{anyhow, Result};
use pixelflow_render::color::Bgra;
use pixelflow_render::Frame;
use std::collections::VecDeque;
use std::ffi::{c_void, CString};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr;
use wayland_sys::client::*;
use xkbcommon::xkb;

// --- Wayland State ---

pub struct WaylandState {
    #[allow(dead_code)]
    pub display: *mut wl_display,
    pub engine_tx: EngineSender<Bgra>,
    pub running: bool,

    // Globals
    pub compositor: *mut wl_compositor,
    pub shm: *mut wl_shm,
    pub wm_base: *mut wl_proxy, // xdg_wm_base
    pub seat: *mut wl_seat,

    // Window
    pub surface: *mut wl_surface,
    pub xdg_surface: *mut wl_proxy,
    pub xdg_toplevel: *mut wl_proxy,
    pub window_id: WindowId,
    pub configured: bool,

    // Dimensions
    pub width: u32,
    pub height: u32,

    // Input
    pub pointer: *mut wl_pointer,
    pub keyboard: *mut wl_keyboard,
    pub xkb_context: xkb::Context,
    pub xkb_state: Option<xkb::State>,
    pub modifiers: Modifiers,

    // Buffers
    pub buffers: VecDeque<Slot>,
}

pub struct Slot {
    pub buffer: *mut wl_buffer,
    #[allow(dead_code)]
    pub pool: *mut wl_shm_pool,
    pub ptr: *mut u8,
    pub size: usize,
    pub width: u32,
    pub height: u32,
    pub free: bool,
}

// Struct to pass context to frame callback
pub struct FrameContext {
    pub tx: EngineSender<Bgra>,
    pub frame: Frame<Bgra>,
}

impl WaylandState {
    pub fn new(display: *mut wl_display, engine_tx: EngineSender<Bgra>) -> Result<Self> {
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

    pub fn handle_command(&mut self, cmd: DriverCommand<Bgra>) {
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
