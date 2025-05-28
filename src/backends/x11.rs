#![allow(non_snake_case)] // Allow non-snake case for X11 types

// Import logging macros
use log::{debug, error, info, trace, warn};

// Crate-level imports
use crate::backends::{
    BackendEvent, CellCoords, CellRect, Driver, MouseButton, MouseEventType, TextRunStyle,
};
use crate::color::{Color, NamedColor};
use crate::glyph::AttrFlags;
use crate::keys::{KeySymbol, Modifiers};

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::ffi::CString;
use std::mem;
use std::os::unix::io::RawFd;
use std::ptr;

// Libc imports
use libc::{c_char, c_int, c_uint};

// X11 library imports
use x11::keysym;
use x11::xft;
use x11::xlib;
use x11::xrender::{XGlyphInfo, XRenderColor};

// --- Constants ---
const DEFAULT_FONT_NAME: &str = "Inconsolata:size=10";
const MIN_FONT_WIDTH: u32 = 1;
const MIN_FONT_HEIGHT: u32 = 1;
const KEY_TEXT_BUFFER_SIZE: usize = 32;

const ANSI_COLOR_COUNT: usize = 16;
const TOTAL_PREALLOC_COLORS: usize = ANSI_COLOR_COUNT;

const DEFAULT_WINDOW_WIDTH_CHARS: usize = 80;
const DEFAULT_WINDOW_HEIGHT_CHARS: usize = 24;

const XRENDER_ALPHA_OPAQUE: u16 = 0xffff;

const XC_XTERM: c_uint = 152;

struct Rgb16Components {
    r: u16,
    g: u16,
    b: u16,
}

pub struct XDriver {
    display: *mut xlib::Display,
    screen: c_int,
    window: xlib::Window,
    colormap: xlib::Colormap,
    visual: *mut xlib::Visual,
    xft_font: *mut xft::XftFont,
    xft_draw: *mut xft::XftDraw,
    xft_ansi_colors: Vec<xft::XftColor>,
    xft_color_cache_rgb: HashMap<(u8, u8, u8), xft::XftColor>,
    font_width: u32,
    font_height: u32,
    font_ascent: u32,
    current_pixel_width: u16,
    current_pixel_height: u16,
    wm_delete_window: xlib::Atom,
    protocols_atom: xlib::Atom,
    clear_gc: xlib::GC,
    has_focus: bool,
    is_native_cursor_visible: bool,
}

// Helper methods specific to XDriver
impl XDriver {
    fn x_state_to_modifiers(state: c_uint) -> Modifiers {
        let mut modifiers = Modifiers::empty();
        if (state & xlib::ShiftMask) != 0 {
            modifiers.insert(Modifiers::SHIFT);
        }
        if (state & xlib::ControlMask) != 0 {
            modifiers.insert(Modifiers::CONTROL);
        }
        if (state & xlib::Mod1Mask) != 0 {
            modifiers.insert(Modifiers::ALT);
        }
        if (state & xlib::Mod4Mask) != 0 {
            modifiers.insert(Modifiers::SUPER);
        }
        // Note: Button masks (Button1Mask, etc.) are part of `state` but usually
        // handled by checking `xbutton.button` or `xmotion.state` for specific button states.
        // The Modifiers enum here is for keyboard-like modifiers.
        modifiers
    }

    fn x_button_event_to_backend_event(
        &self,
        xbutton: &xlib::XButtonEvent,
        event_type: MouseEventType,
    ) -> BackendEvent {
        let col = if self.font_width > 0 {
            (xbutton.x as usize) / self.font_width as usize
        } else {
            0
        };
        let row = if self.font_height > 0 {
            (xbutton.y as usize) / self.font_height as usize
        } else {
            0
        };
        let button = match xbutton.button {
            xlib::Button1 => MouseButton::Left,
            xlib::Button2 => MouseButton::Middle,
            xlib::Button3 => MouseButton::Right,
            xlib::Button4 => MouseButton::ScrollUp,
            xlib::Button5 => MouseButton::ScrollDown,
            6 => MouseButton::ScrollLeft,
            7 => MouseButton::ScrollRight,
            _ => MouseButton::Unknown,
        };
        let modifiers = Self::x_state_to_modifiers(xbutton.state);

        BackendEvent::Mouse {
            col,
            row,
            event_type,
            button,
            modifiers,
        }
    }

    fn x_motion_event_to_backend_event(&self, xmotion: &xlib::XMotionEvent) -> BackendEvent {
        let col = if self.font_width > 0 {
            (xmotion.x as usize) / self.font_width as usize
        } else {
            0
        };
        let row = if self.font_height > 0 {
            (xmotion.y as usize) / self.font_height as usize
        } else {
            0
        };
        let event_type = MouseEventType::Move;
        let button = if (xmotion.state & xlib::Button1Mask) != 0 {
            MouseButton::Left
        } else if (xmotion.state & xlib::Button2Mask) != 0 {
            MouseButton::Middle
        } else if (xmotion.state & xlib::Button3Mask) != 0 {
            MouseButton::Right
        } else {
            MouseButton::Unknown
        };
        let modifiers = Self::x_state_to_modifiers(xmotion.state);

        BackendEvent::Mouse {
            col,
            row,
            event_type,
            button,
            modifiers,
        }
    }
    // ... (other private XDriver methods like load_font, init_xft_ansi_colors, etc. remain here)
    // These are unchanged from the original file and are omitted for brevity in this diff.
    // The full content will be in the final corrected file.

    /// Loads the primary font using Xft and calculates basic font metrics.
    fn load_font(&mut self) -> Result<()> {
        debug!("Loading font: {}", DEFAULT_FONT_NAME);
        let font_name_cstr =
            CString::new(DEFAULT_FONT_NAME).context("Failed to create CString for font name")?;

        self.xft_font =
            unsafe { xft::XftFontOpenName(self.display, self.screen, font_name_cstr.as_ptr()) };
        if self.xft_font.is_null() {
            return Err(anyhow::anyhow!(
                "XftFontOpenName failed for font: '{}'. Ensure font is installed and accessible.",
                DEFAULT_FONT_NAME
            ));
        }
        debug!("Font '{}' loaded successfully.", DEFAULT_FONT_NAME);

        let font_info_ptr = self.xft_font;
        self.font_height = unsafe { ((*font_info_ptr).ascent + (*font_info_ptr).descent) as u32 };
        self.font_ascent = unsafe { (*font_info_ptr).ascent as u32 };

        let mut extents: XGlyphInfo = unsafe { mem::zeroed() };
        let sample_char_cstr = CString::new("M").expect("CString::new for 'M' failed.");
        unsafe {
            xft::XftTextExtentsUtf8(
                self.display,
                self.xft_font,
                sample_char_cstr.as_ptr() as *const u8,
                sample_char_cstr.as_bytes().len() as c_int,
                &mut extents,
            );
        }
        self.font_width = extents.xOff as u32;

        if self.font_width < MIN_FONT_WIDTH || self.font_height < MIN_FONT_HEIGHT {
            return Err(anyhow::anyhow!(
                "Font dimensions (W:{}, H:{}) below minimum (W:{}, H:{}).",
                self.font_width,
                self.font_height,
                MIN_FONT_WIDTH,
                MIN_FONT_HEIGHT
            ));
        }
        info!(
            "Font metrics: Width={}, Height={}, Ascent={}",
            self.font_width, self.font_height, self.font_ascent
        );
        Ok(())
    }

    fn init_xft_ansi_colors(&mut self) -> Result<()> {
        debug!(
            "Initializing {} preallocated ANSI Xft colors.",
            ANSI_COLOR_COUNT
        );
        self.xft_ansi_colors
            .resize_with(ANSI_COLOR_COUNT, || unsafe { mem::zeroed() });

        for i in 0..ANSI_COLOR_COUNT {
            let named_color_enum = NamedColor::from_index(i as u8);
            let rgb_color = named_color_enum.to_rgb_color();
            let (r_u8, g_u8, b_u8) = match rgb_color {
                Color::Rgb(r, g, b) => (r, g, b),
                _ => {
                    warn!(
                        "NamedColor::to_rgb_color did not return Color::Rgb for {:?}. Defaulting to black.",
                        named_color_enum
                    );
                    (0, 0, 0)
                }
            };

            let color_components = Rgb16Components {
                r: ((r_u8 as u16) << 8) | (r_u8 as u16),
                g: ((g_u8 as u16) << 8) | (g_u8 as u16),
                b: ((b_u8 as u16) << 8) | (b_u8 as u16),
            };
            self.alloc_specific_xft_color_into_slice(
                i,
                color_components,
                &format!("ANSI {} ({:?})", i, named_color_enum),
            )?;
        }
        debug!("Preallocated ANSI Xft colors initialized.");
        Ok(())
    }

    fn alloc_specific_xft_color_into_slice(
        &mut self,
        index: usize,
        color_comps: Rgb16Components,
        name_for_log: &str,
    ) -> Result<()> {
        let render_color = XRenderColor {
            red: color_comps.r,
            green: color_comps.g,
            blue: color_comps.b,
            alpha: XRENDER_ALPHA_OPAQUE,
        };
        if unsafe {
            xft::XftColorAllocValue(
                self.display,
                self.visual,
                self.colormap,
                &render_color,
                &mut self.xft_ansi_colors[index],
            )
        } == 0
        {
            return Err(anyhow::anyhow!(
                "XftColorAllocValue failed for {}",
                name_for_log
            ));
        }
        trace!(
            "Allocated XftColor for {} (idx {}, pixel: {})",
            name_for_log, index, &self.xft_ansi_colors[index].pixel
        );
        Ok(())
    }

    fn create_window(
        &mut self,
        pixel_width: u16,
        pixel_height: u16,
        bg_pixel_val: xlib::Atom,
    ) -> Result<()> {
        unsafe {
            let root_window = xlib::XRootWindow(self.display, self.screen);
            let border_width = 0;

            let mut attributes: xlib::XSetWindowAttributes = mem::zeroed();
            attributes.colormap = self.colormap;
            attributes.background_pixel = bg_pixel_val;
            attributes.border_pixel = bg_pixel_val;
            attributes.event_mask = xlib::ExposureMask
                | xlib::KeyPressMask
                | xlib::StructureNotifyMask
                | xlib::FocusChangeMask
                | xlib::ButtonPressMask
                | xlib::ButtonReleaseMask
                | xlib::PointerMotionMask;

            self.window = xlib::XCreateWindow(
                self.display,
                root_window,
                0,
                0,
                pixel_width as c_uint,
                pixel_height as c_uint,
                border_width,
                xlib::XDefaultDepth(self.display, self.screen),
                xlib::InputOutput as c_uint,
                self.visual,
                xlib::CWColormap | xlib::CWBackPixel | xlib::CWBorderPixel | xlib::CWEventMask,
                &mut attributes,
            );
        }
        if self.window == 0 {
            return Err(anyhow::anyhow!("XCreateWindow failed."));
        }
        debug!(
            "X window created (ID: {}), initial size: {}x{}px",
            self.window, pixel_width, pixel_height
        );
        self.current_pixel_width = pixel_width;
        self.current_pixel_height = pixel_height;
        Ok(())
    }

    fn create_gc(&mut self) -> Result<()> {
        let gc_values: xlib::XGCValues = unsafe { mem::zeroed() };
        self.clear_gc = unsafe {
            xlib::XCreateGC(
                self.display,
                self.window,
                0,
                &gc_values as *const _ as *mut _,
            )
        };
        if self.clear_gc.is_null() {
            return Err(anyhow::anyhow!("XCreateGC failed."));
        }
        debug!("Graphics Context (GC) for clearing created.");
        Ok(())
    }

    fn setup_wm_protocols_and_hints(&mut self) {
        unsafe {
            self.wm_delete_window = xlib::XInternAtom(
                self.display,
                b"WM_DELETE_WINDOW\0".as_ptr() as *const i8,
                xlib::False,
            );
            self.protocols_atom = xlib::XInternAtom(
                self.display,
                b"WM_PROTOCOLS\0".as_ptr() as *const i8,
                xlib::False,
            );

            if self.wm_delete_window != 0 && self.protocols_atom != 0 {
                xlib::XSetWMProtocols(
                    self.display,
                    self.window,
                    [self.wm_delete_window].as_mut_ptr(),
                    1,
                );
                debug!("WM_PROTOCOLS (WM_DELETE_WINDOW) registered.");
            } else {
                warn!(
                    "Failed to get WM_DELETE_WINDOW or WM_PROTOCOLS atom. Window close button might not work as expected."
                );
            }

            let title_cstr = CString::new("core-term").expect("CString::new for title failed.");
            xlib::XStoreName(
                self.display,
                self.window,
                title_cstr.as_ptr() as *mut c_char,
            );

            let net_wm_name_atom = xlib::XInternAtom(
                self.display,
                b"_NET_WM_NAME\0".as_ptr() as *const i8,
                xlib::False,
            );
            let utf8_string_atom = xlib::XInternAtom(
                self.display,
                b"UTF8_STRING\0".as_ptr() as *const i8,
                xlib::False,
            );
            if net_wm_name_atom != 0 && utf8_string_atom != 0 {
                xlib::XChangeProperty(
                    self.display,
                    self.window,
                    net_wm_name_atom,
                    utf8_string_atom,
                    8,
                    xlib::PropModeReplace,
                    title_cstr.as_ptr() as *const u8,
                    title_cstr.as_bytes().len() as c_int,
                );
                debug!("Window title set via XStoreName and _NET_WM_NAME.");
            } else {
                debug!(
                    "Window title set via XStoreName only (_NET_WM_NAME or UTF8_STRING atom not found)."
                );
            }

            let mut size_hints: xlib::XSizeHints = mem::zeroed();
            size_hints.flags = xlib::PResizeInc | xlib::PMinSize;
            size_hints.width_inc = self.font_width as c_int;
            size_hints.height_inc = self.font_height as c_int;
            size_hints.min_width = self.font_width as c_int;
            size_hints.min_height = self.font_height as c_int;
            xlib::XSetWMNormalHints(self.display, self.window, &mut size_hints);
            debug!("WM size hints set.");
        }
    }

    fn resolve_concrete_xft_color(&mut self, color: Color) -> Result<xft::XftColor> {
        match color {
            Color::Default => {
                error!(
                    "XDriver::resolve_concrete_xft_color received Color::Default. This is a bug."
                );
                self.cached_rgb_to_xft_color(0, 0, 0)
                    .context("Fallback to black failed after Color::Default error")
            }
            Color::Named(named_color) => Ok(self.xft_ansi_colors[named_color as u8 as usize]),
            Color::Indexed(idx) => {
                let rgb_equivalent = crate::color::convert_to_rgb_color(Color::Indexed(idx));
                if let Color::Rgb(r, g, b) = rgb_equivalent {
                    trace!(
                        "XDriver: Approximating Indexed({}) to RGB({},{},{}) for XftColor.",
                        idx, r, g, b
                    );
                    self.cached_rgb_to_xft_color(r, g, b)
                } else {
                    error!(
                        "Failed to convert Indexed({}) to RGB. Defaulting to black.",
                        idx
                    );
                    self.cached_rgb_to_xft_color(0, 0, 0)
                }
            }
            Color::Rgb(r, g, b) => self.cached_rgb_to_xft_color(r, g, b),
        }
    }

    fn cached_rgb_to_xft_color(&mut self, r_u8: u8, g_u8: u8, b_u8: u8) -> Result<xft::XftColor> {
        if let Some(cached_color) = self.xft_color_cache_rgb.get(&(r_u8, g_u8, b_u8)) {
            return Ok(*cached_color);
        }

        let color_components = Rgb16Components {
            r: ((r_u8 as u16) << 8) | (r_u8 as u16),
            g: ((g_u8 as u16) << 8) | (g_u8 as u16),
            b: ((b_u8 as u16) << 8) | (b_u8 as u16),
        };

        let render_color = XRenderColor {
            red: color_components.r,
            green: color_components.g,
            blue: color_components.b,
            alpha: XRENDER_ALPHA_OPAQUE,
        };
        let mut new_xft_color: xft::XftColor = unsafe { mem::zeroed() };

        if unsafe {
            xft::XftColorAllocValue(
                self.display,
                self.visual,
                self.colormap,
                &render_color,
                &mut new_xft_color,
            )
        } == 0
        {
            Err(anyhow::anyhow!(
                "XftColorAllocValue failed for RGB({},{},{})",
                r_u8,
                g_u8,
                b_u8
            ))
        } else {
            self.xft_color_cache_rgb
                .insert((r_u8, g_u8, b_u8), new_xft_color);
            Ok(new_xft_color)
        }
    }
}

impl Drop for XDriver {
    fn drop(&mut self) {
        info!("Dropping XDriver instance, performing cleanup.");
        if let Err(e) = self.cleanup() {
            error!("Error during XDriver cleanup in drop: {}", e);
        }
    }
}

// Helper function to translate X11 KeySym to our KeySymbol
// (This function remains largely the same as provided in the original file,
//  it's a standalone function not part of the XDriver struct directly)
// ... (xkeysym_to_keysymbol function as provided in the original file) ...
fn xkeysym_to_keysymbol<T: IntoXKeySym>(keysym_val_in: T, text: &str) -> KeySymbol {
    let keysym_val = keysym_val_in.into_xkeysym();
    if keysym_val > (u32::MAX as xlib::KeySym) {
        log::debug!(
            "Received high keysym value: 0x{:X}, not in standard u32 XK_* range.",
            keysym_val
        );
        if !text.is_empty() && text.chars().all(|c| c != '\u{FFFD}') {
            if let Some(ch) = text.chars().next() {
                return KeySymbol::Char(ch);
            }
        }
        return KeySymbol::Unknown;
    }

    let keysym_u32 = keysym_val as u32;

    match keysym_u32 {
        keysym::XK_Shift_L | keysym::XK_Shift_R => KeySymbol::Shift,
        keysym::XK_Control_L | keysym::XK_Control_R => KeySymbol::Control,
        keysym::XK_Alt_L | keysym::XK_Alt_R | keysym::XK_Meta_L | keysym::XK_Meta_R => {
            KeySymbol::Alt
        }
        keysym::XK_Super_L | keysym::XK_Super_R | keysym::XK_Hyper_L | keysym::XK_Hyper_R => {
            KeySymbol::Super
        }
        keysym::XK_Caps_Lock => KeySymbol::CapsLock,
        keysym::XK_Num_Lock => KeySymbol::NumLock,
        keysym::XK_Return => KeySymbol::Enter,
        keysym::XK_KP_Enter => KeySymbol::KeypadEnter,
        keysym::XK_Linefeed => KeySymbol::Char('\n'),
        keysym::XK_BackSpace => KeySymbol::Backspace,
        keysym::XK_Tab | keysym::XK_KP_Tab | keysym::XK_ISO_Left_Tab => KeySymbol::Tab,
        keysym::XK_Escape => KeySymbol::Escape,
        keysym::XK_Home => KeySymbol::Home,
        keysym::XK_Left => KeySymbol::Left,
        keysym::XK_Up => KeySymbol::Up,
        keysym::XK_Right => KeySymbol::Right,
        keysym::XK_Down => KeySymbol::Down,
        keysym::XK_Page_Up => KeySymbol::PageUp,
        keysym::XK_Page_Down => KeySymbol::PageDown,
        keysym::XK_End => KeySymbol::End,
        keysym::XK_Insert => KeySymbol::Insert,
        keysym::XK_Delete => KeySymbol::Delete,
        keysym::XK_F1 => KeySymbol::F1,
        keysym::XK_F2 => KeySymbol::F2,
        keysym::XK_F3 => KeySymbol::F3,
        keysym::XK_F4 => KeySymbol::F4,
        keysym::XK_F5 => KeySymbol::F5,
        keysym::XK_F6 => KeySymbol::F6,
        keysym::XK_F7 => KeySymbol::F7,
        keysym::XK_F8 => KeySymbol::F8,
        keysym::XK_F9 => KeySymbol::F9,
        keysym::XK_F10 => KeySymbol::F10,
        keysym::XK_F11 => KeySymbol::F11,
        keysym::XK_F12 => KeySymbol::F12,
        keysym::XK_F13 => KeySymbol::F13,
        keysym::XK_F14 => KeySymbol::F14,
        keysym::XK_F15 => KeySymbol::F15,
        keysym::XK_F16 => KeySymbol::F16,
        keysym::XK_F17 => KeySymbol::F17,
        keysym::XK_F18 => KeySymbol::F18,
        keysym::XK_F19 => KeySymbol::F19,
        keysym::XK_F20 => KeySymbol::F20,
        keysym::XK_F21 => KeySymbol::F21,
        keysym::XK_F22 => KeySymbol::F22,
        keysym::XK_F23 => KeySymbol::F23,
        keysym::XK_F24 => KeySymbol::F24,
        keysym::XK_KP_0 | keysym::XK_KP_Insert => KeySymbol::Keypad0,
        keysym::XK_KP_1 | keysym::XK_KP_End => KeySymbol::Keypad1,
        keysym::XK_KP_2 | keysym::XK_KP_Down => KeySymbol::Keypad2,
        keysym::XK_KP_3 | keysym::XK_KP_Page_Down => KeySymbol::Keypad3,
        keysym::XK_KP_4 | keysym::XK_KP_Left => KeySymbol::Keypad4,
        keysym::XK_KP_5 | keysym::XK_KP_Begin => KeySymbol::Keypad5,
        keysym::XK_KP_6 | keysym::XK_KP_Right => KeySymbol::Keypad6,
        keysym::XK_KP_7 | keysym::XK_KP_Home => KeySymbol::Keypad7,
        keysym::XK_KP_8 | keysym::XK_KP_Up => KeySymbol::Keypad8,
        keysym::XK_KP_9 | keysym::XK_KP_Page_Up => KeySymbol::Keypad9,
        keysym::XK_KP_Decimal | keysym::XK_KP_Delete | keysym::XK_KP_Separator => {
            KeySymbol::KeypadDecimal
        }
        keysym::XK_KP_Add => KeySymbol::KeypadPlus,
        keysym::XK_KP_Subtract => KeySymbol::KeypadMinus,
        keysym::XK_KP_Multiply => KeySymbol::KeypadMultiply,
        keysym::XK_KP_Divide => KeySymbol::KeypadDivide,
        keysym::XK_KP_Equal => KeySymbol::KeypadEquals,
        keysym::XK_KP_Space => KeySymbol::Char(' '),
        keysym::XK_Print | keysym::XK_Sys_Req => KeySymbol::PrintScreen,
        keysym::XK_Scroll_Lock => KeySymbol::ScrollLock,
        keysym::XK_Pause | keysym::XK_Break => KeySymbol::Pause,
        keysym::XK_Menu => KeySymbol::Menu,
        _ => {
            if !text.is_empty() && text.chars().all(|c| c != '\u{FFFD}') {
                if let Some(ch) = text.chars().next() {
                    return KeySymbol::Char(ch);
                }
            }
            log::trace!(
                "Unhandled u32 keysym 0x{:X} with text '{}', mapping to Unknown",
                keysym_u32,
                text
            );
            KeySymbol::Unknown
        }
    }
}

// Trait for converting to XKeySym, useful for xkeysym_to_keysymbol
pub trait IntoXKeySym: Sized + Copy + Clone {
    fn into_xkeysym(self) -> xlib::KeySym;
}

impl IntoXKeySym for u32 {
    #[inline]
    fn into_xkeysym(self) -> xlib::KeySym {
        self as xlib::KeySym
    }
}

impl IntoXKeySym for xlib::KeySym {
    #[inline]
    fn into_xkeysym(self) -> xlib::KeySym {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libc::c_uint;
    use x11::keysym;
    use x11::xlib;

    fn mock_xkey_event(state: u32, keycode: u32) -> xlib::XKeyEvent {
        xlib::XKeyEvent {
            type_: xlib::KeyPress,
            serial: 0,
            send_event: 0,
            display: std::ptr::null_mut(),
            window: 0,
            root: 0,
            subwindow: 0,
            time: 0,
            x: 0,
            y: 0,
            x_root: 0,
            y_root: 0,
            state: state as std::os::raw::c_uint,
            keycode: keycode as std::os::raw::c_uint,
            same_screen: 0,
        }
    }

    #[test]
    fn test_modifier_mapping() {
        let mut event = xlib::XEvent {
            key: mock_xkey_event(0, 0),
        };
        event.type_ = xlib::KeyPress;

        event.key.state = 0;
        assert_eq!(
            XDriver::x_state_to_modifiers(unsafe { event.key }.state),
            Modifiers::empty()
        );

        event.key.state = xlib::ShiftMask;
        assert_eq!(
            XDriver::x_state_to_modifiers(unsafe { event.key }.state),
            Modifiers::SHIFT
        );

        event.key.state = xlib::ControlMask;
        assert_eq!(
            XDriver::x_state_to_modifiers(unsafe { event.key }.state),
            Modifiers::CONTROL
        );

        event.key.state = xlib::Mod1Mask;
        assert_eq!(
            XDriver::x_state_to_modifiers(unsafe { event.key }.state),
            Modifiers::ALT
        );

        event.key.state = xlib::Mod4Mask;
        assert_eq!(
            XDriver::x_state_to_modifiers(unsafe { event.key }.state),
            Modifiers::SUPER
        );

        event.key.state = xlib::ShiftMask | xlib::ControlMask;
        assert_eq!(
            XDriver::x_state_to_modifiers(unsafe { event.key }.state),
            Modifiers::SHIFT | Modifiers::CONTROL
        );
    }

    #[test]
    fn test_xkeysym_to_keysymbol_special_keys() {
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Return, ""),
            KeySymbol::Enter
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Escape, ""),
            KeySymbol::Escape
        );
    }

    #[test]
    fn test_xkeysym_to_keysymbolhar_input() {
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_a, "a"),
            KeySymbol::Char('a')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_A, "A"),
            KeySymbol::Char('A')
        );
    }

    fn mock_xdriver_for_mouse_tests() -> XDriver {
        XDriver {
            display: std::ptr::null_mut(),
            screen: 0,
            window: 0,
            colormap: 0,
            visual: std::ptr::null_mut(),
            xft_font: std::ptr::null_mut(),
            xft_draw: std::ptr::null_mut(),
            xft_ansi_colors: Vec::new(),
            xft_color_cache_rgb: HashMap::new(),
            font_width: 8,
            font_height: 16,
            font_ascent: 0,
            current_pixel_width: 0,
            current_pixel_height: 0,
            wm_delete_window: 0,
            protocols_atom: 0,
            clear_gc: std::ptr::null_mut(),
            has_focus: true,
            is_native_cursor_visible: true,
        }
    }

    fn mock_xbutton_event(x: i32, y: i32, button_code: u32, state_mask: u32) -> xlib::XButtonEvent {
        let mut event: xlib::XButtonEvent = unsafe { mem::zeroed() };
        event.type_ = xlib::ButtonPress;
        event.x = x;
        event.y = y;
        event.button = button_code as c_uint;
        event.state = state_mask as c_uint;
        event
    }

    #[test]
    fn test_x11_button_press_to_backend_event() {
        let driver = mock_xdriver_for_mouse_tests();
        let xbutton = mock_xbutton_event(30, 60, xlib::Button1, xlib::ShiftMask);

        let backend_event = driver.x_button_event_to_backend_event(&xbutton, MouseEventType::Press);

        match backend_event {
            BackendEvent::Mouse {
                col,
                row,
                event_type,
                button,
                modifiers,
            } => {
                assert_eq!(col, 3);
                assert_eq!(row, 3);
                assert_eq!(event_type, MouseEventType::Press);
                assert_eq!(button, MouseButton::Left);
                assert_eq!(modifiers, Modifiers::SHIFT);
            }
            _ => panic!("Incorrect event type returned"),
        }
    }

    fn mock_xmotion_event(x: i32, y: i32, state_mask: u32) -> xlib::XMotionEvent {
        let mut event: xlib::XMotionEvent = unsafe { mem::zeroed() };
        event.type_ = xlib::MotionNotify;
        event.x = x;
        event.y = y;
        event.state = state_mask as c_uint;
        event
    }

    #[test]
    fn test_x11_motion_notify_to_backend_event_drag() {
        let driver = mock_xdriver_for_mouse_tests();
        let xmotion = mock_xmotion_event(45, 85, xlib::Button1Mask | xlib::ControlMask);

        let backend_event = driver.x_motion_event_to_backend_event(&xmotion);

        match backend_event {
            BackendEvent::Mouse {
                col,
                row,
                event_type,
                button,
                modifiers,
            } => {
                assert_eq!(col, 5);
                assert_eq!(row, 5);
                assert_eq!(event_type, MouseEventType::Move);
                assert_eq!(button, MouseButton::Left);
                assert_eq!(modifiers, Modifiers::CONTROL); // x_state_to_modifiers doesn't add button masks currently
            }
            _ => panic!("Incorrect event type returned"),
        }
    }
}
