//! Wayland event listeners.

use super::protocol::*;
use super::state::{WaylandState, FrameContext};
use crate::channel::EngineCommand;
use crate::display::messages::{DisplayEvent};
use crate::input::{KeySymbol, Modifiers};
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use wayland_sys::client::*;
use xkbcommon::xkb;

// --- Registry ---

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
pub struct wl_registry_listener {
    pub global: extern "C" fn(*mut c_void, *mut wl_registry, u32, *const c_char, u32),
    pub global_remove: extern "C" fn(*mut c_void, *mut wl_registry, u32),
}

pub static REGISTRY_LISTENER: wl_registry_listener = wl_registry_listener {
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

#[repr(C)]
pub struct xdg_wm_base_listener {
    pub ping: extern "C" fn(*mut c_void, *mut wl_proxy, u32),
}

pub static XDG_WM_BASE_LISTENER: xdg_wm_base_listener = xdg_wm_base_listener {
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
pub struct xdg_surface_listener {
    pub configure: extern "C" fn(*mut c_void, *mut wl_proxy, u32),
}

pub static XDG_SURFACE_LISTENER: xdg_surface_listener = xdg_surface_listener {
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
pub struct xdg_toplevel_listener {
    pub configure: extern "C" fn(*mut c_void, *mut wl_proxy, i32, i32, *mut wl_array),
    pub close: extern "C" fn(*mut c_void, *mut wl_proxy),
}

pub static XDG_TOPLEVEL_LISTENER: xdg_toplevel_listener = xdg_toplevel_listener {
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
pub struct wl_buffer_listener {
    pub release: extern "C" fn(*mut c_void, *mut wl_buffer),
}

pub static BUFFER_LISTENER: wl_buffer_listener = wl_buffer_listener {
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

#[repr(C)]
pub struct wl_callback_listener {
    pub done: extern "C" fn(*mut c_void, *mut wl_callback, u32),
}

pub static FRAME_LISTENER: wl_callback_listener = wl_callback_listener {
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
pub struct wl_seat_listener {
    pub capabilities: extern "C" fn(*mut c_void, *mut wl_seat, u32),
    pub name: extern "C" fn(*mut c_void, *mut wl_seat, *const c_char),
}

pub static SEAT_LISTENER: wl_seat_listener = wl_seat_listener {
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
pub struct wl_pointer_listener {
    pub enter: extern "C" fn(*mut c_void, *mut wl_pointer, u32, *mut wl_surface, i32, i32),
    pub leave: extern "C" fn(*mut c_void, *mut wl_pointer, u32, *mut wl_surface),
    pub motion: extern "C" fn(*mut c_void, *mut wl_pointer, u32, i32, i32),
    pub button: extern "C" fn(*mut c_void, *mut wl_pointer, u32, u32, u32, u32),
    pub axis: extern "C" fn(*mut c_void, *mut wl_pointer, u32, u32, i32),
    pub frame: extern "C" fn(*mut c_void, *mut wl_pointer),
    pub axis_source: extern "C" fn(*mut c_void, *mut wl_pointer, u32),
    pub axis_stop: extern "C" fn(*mut c_void, *mut wl_pointer, u32, u32),
    pub axis_discrete: extern "C" fn(*mut c_void, *mut wl_pointer, u32, i32),
}

pub static POINTER_LISTENER: wl_pointer_listener = wl_pointer_listener {
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
pub struct wl_keyboard_listener {
    pub keymap: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, i32, u32),
    pub enter: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, *mut wl_surface, *mut wl_array),
    pub leave: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, *mut wl_surface),
    pub key: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, u32, u32, u32),
    pub modifiers: extern "C" fn(*mut c_void, *mut wl_keyboard, u32, u32, u32, u32, u32),
    pub repeat_info: extern "C" fn(*mut c_void, *mut wl_keyboard, i32, i32),
}

pub static KEYBOARD_LISTENER: wl_keyboard_listener = wl_keyboard_listener {
    keymap: keyboard_keymap,
    enter: keyboard_enter,
    leave: keyboard_leave,
    key: keyboard_key,
    modifiers: keyboard_modifiers,
    repeat_info: keyboard_repeat_info,
};
