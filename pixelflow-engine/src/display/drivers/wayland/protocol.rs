//! Wayland protocol definitions and FFI bindings.

use std::ffi::{c_char, c_int, CStr};
use std::ptr;
use wayland_sys::common::{wl_interface, wl_message};
use wayland_sys::client::wl_proxy;

// --- Type Aliases for Opaque Proxies ---
#[allow(non_camel_case_types)]
pub type wl_registry = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_compositor = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_shm = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_shm_pool = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_buffer = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_seat = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_surface = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_pointer = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_keyboard = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_callback = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_array = wayland_sys::common::wl_array;
#[allow(non_camel_case_types)]
pub type wl_output = wl_proxy;
#[allow(non_camel_case_types)]
pub type wl_region = wl_proxy;

// --- Constants ---
pub const WL_SHM_FORMAT_ARGB8888: u32 = 0;

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

// --- Interface Definitions for XDG Shell ---

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

pub unsafe fn init_interfaces() {
    XDG_WM_BASE_GET_XDG_SURFACE_TYPES[0] = ptr::addr_of!(xdg_surface_interface);
    XDG_WM_BASE_GET_XDG_SURFACE_TYPES[1] = ptr::addr_of!(wl_surface_interface);
    XDG_WM_BASE_REQUESTS[2].types = XDG_WM_BASE_GET_XDG_SURFACE_TYPES.as_ptr();

    XDG_SURFACE_GET_TOPLEVEL_TYPES[0] = ptr::addr_of!(xdg_toplevel_interface);
    XDG_SURFACE_REQUESTS[1].types = XDG_SURFACE_GET_TOPLEVEL_TYPES.as_ptr();

    XDG_TOPLEVEL_SET_FULLSCREEN_TYPES[0] = ptr::addr_of!(wl_output_interface);
    XDG_TOPLEVEL_REQUESTS[11].types = XDG_TOPLEVEL_SET_FULLSCREEN_TYPES.as_ptr();
}
