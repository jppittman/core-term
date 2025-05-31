// src/platform/backends/x11/event.rs
#![allow(non_snake_case)] // Allow non-snake case for X11 types

use super::connection::Connection;
use super::window::Window;
use crate::keys::{KeySymbol, Modifiers};
use crate::platform::backends::BackendEvent;

use anyhow::Result; // Context not used directly, anyhow for Result
use log::{debug, info, trace, warn};
use std::mem;
use std::ptr;

// X11 library imports
use libc::{c_char, c_int};
use x11::{keysym, xlib};

/// Buffer size for text obtained from `XLookupString`.
const KEY_TEXT_BUFFER_SIZE: usize = 32;

/// Processes all pending X11 events from the server.
///
/// This function polls the X server for events using `XPending` and then processes
/// each event in a loop with `XNextEvent`. It translates relevant X11 events
/// into generic `BackendEvent`s which are consumed by the application's core logic.
///
/// # Arguments
///
/// * `connection`: A reference to the X11 `Connection` object, used for Xlib calls.
/// * `window`: A mutable reference to the `Window` object. This is used to:
///     - Access window-specific information (e.g., atoms for client messages like `WM_DELETE_WINDOW`).
///     - Update window state (e.g., current dimensions upon receiving a resize event).
/// * `xdriver_has_focus`: A mutable reference to a boolean flag, managed by `XDriver`,
///   indicating if the application window currently has input focus. This flag is
///   updated by this function based on `FocusIn` and `FocusOut` XEvents.
///
/// # Returns
///
/// * `Ok(Vec<BackendEvent>)`: A vector containing all `BackendEvent`s generated
///   from the processed X11 events during this call. The vector may be empty if
///   no relevant events were pending or processed.
/// * `Err(anyhow::Error)`: If a critical error occurs during event processing.
///   Currently, this function's Xlib calls are not expected to return errors
///   that would propagate here, but `Result` is used for future-proofing or if
///   any fallible operations were to be added.
pub fn process_pending_events(
    connection: &Connection,
    window: &mut Window,
    xdriver_has_focus: &mut bool,
) -> Result<Vec<BackendEvent>> {
    let mut backend_events = Vec::new();
    let display = connection.display();

    // Loop while there are events pending on the X display connection.
    // SAFETY: `XPending` is safe to call with `display`, which is a valid pointer
    // from an active `Connection`. It returns the number of events in the queue.
    while unsafe { xlib::XPending(display) } > 0 {
        let mut xevent: xlib::XEvent = unsafe { mem::zeroed() };
        // SAFETY: `XNextEvent` is safe here. `display` is a valid pointer from an
        // active `Connection`. `xevent` is a valid mutable pointer to a zero-initialized
        // `XEvent` struct. The loop condition `XPending(display) > 0` ensures that
        // `XNextEvent` will not block indefinitely if no events are available,
        // though `XNextEvent` itself is designed to block until an event is received.
        unsafe { xlib::XNextEvent(display, &mut xevent) };

        // SAFETY: Accessing the `type_` field of the `XEvent` union is safe.
        // `XNextEvent` has successfully populated `xevent`, and `type_` is a common
        // discriminant field for all XEvent variants.
        let event_type = unsafe { xevent.type_ };

        match event_type {
            xlib::Expose => {
                // SAFETY: Accessing the `xexpose` union field of `xevent` is safe because
                // the `event_type` (derived from `xevent.type_`) has been confirmed
                // to be `xlib::Expose`.
                let expose_event = unsafe { xevent.expose };
                // Process only the last Expose event in a series (when count is 0).
                // The renderer handles the actual redrawing based on its own state.
                if expose_event.count == 0 {
                    trace!(
                        "XEvent: Expose (win: {}, x:{}, y:{}, w:{}, h:{}) - redraw will be handled by renderer",
                        expose_event.window, expose_event.x, expose_event.y, expose_event.width, expose_event.height
                    );
                }
            }
            xlib::ConfigureNotify => {
                // SAFETY: Accessing the `xconfigure` union field of `xevent` is safe because
                // the `event_type` has been confirmed to be `xlib::ConfigureNotify`.
                let configure_event = unsafe { xevent.configure };
                let (current_w, current_h) = window.current_dimensions_pixels();

                // Check if the window dimensions have actually changed.
                if current_w != configure_event.width as u16
                    || current_h != configure_event.height as u16
                {
                    debug!(
                        "XEvent: ConfigureNotify (resize from {}x{} to {}x{}) on window {}",
                        current_w,
                        current_h,
                        configure_event.width,
                        configure_event.height,
                        configure_event.window
                    );
                    // Update internal window dimension cache.
                    window.update_dimensions(
                        configure_event.width as u16,
                        configure_event.height as u16,
                    );

                    backend_events.push(BackendEvent::Resize {
                        width_px: configure_event.width as u16,
                        height_px: configure_event.height as u16,
                    });
                } else {
                    // Event might be for other changes (e.g., position), which we ignore for now.
                    trace!(
                        "XEvent: ConfigureNotify (no size change detected) on window {}",
                        configure_event.window
                    );
                }
            }
            xlib::KeyPress => {
                // SAFETY: Accessing the `xkey` union field of `xevent` is safe because
                // the `event_type` has been confirmed to be `xlib::KeyPress`.
                // A mutable reference `&mut xevent.key` is taken because `XLookupString`
                // requires a mutable pointer to the `XKeyEvent` structure, although it
                // typically does not modify it in ways that would affect other fields
                // beyond what's related to key translation state if an XIM were involved
                // (which it isn't here, as `XLookupString` is used without an XIM).
                let key_event = unsafe { &mut xevent.key };
                let mut x_keysym: xlib::KeySym = 0;
                let mut key_text_buffer = [0u8; KEY_TEXT_BUFFER_SIZE];

                // SAFETY: `XLookupString` is an Xlib FFI call.
                // - `key_event` is a valid pointer to an `XKeyEvent` (part of `xevent`).
                // - `key_text_buffer.as_mut_ptr()` provides a valid pointer to a writable buffer.
                // - `key_text_buffer.len()` correctly specifies the buffer's size.
                // - `&mut x_keysym` provides a valid pointer for storing the KeySym.
                // - `ptr::null_mut()` is acceptable for the XComposeStatus argument if compose
                //   sequence information is not needed.
                // The function populates `key_text_buffer` with translated characters and
                // `x_keysym` with the KeySym.
                let count = unsafe {
                    xlib::XLookupString(
                        key_event, // This is `*mut xlib::XKeyEvent`
                        key_text_buffer.as_mut_ptr() as *mut c_char,
                        key_text_buffer.len() as c_int,
                        &mut x_keysym,
                        ptr::null_mut(), // No XComposeStatus needed.
                    )
                };

                let text = if count > 0 {
                    String::from_utf8_lossy(&key_text_buffer[0..count as usize]).to_string()
                } else {
                    String::new()
                };

                // Determine active modifiers.
                let mut modifiers = Modifiers::empty();
                if (key_event.state & xlib::ShiftMask) != 0 {
                    modifiers.insert(Modifiers::SHIFT);
                }
                if (key_event.state & xlib::ControlMask) != 0 {
                    modifiers.insert(Modifiers::CONTROL);
                }
                if (key_event.state & xlib::Mod1Mask) != 0 {
                    modifiers.insert(Modifiers::ALT);
                } // Mod1Mask is typically Alt.
                if (key_event.state & xlib::Mod4Mask) != 0 {
                    modifiers.insert(Modifiers::SUPER);
                } // Mod4Mask is typically Super/Windows.

                let symbol = xkeysym_to_keysymbol(x_keysym, &text);

                debug!(
                    "XEvent: KeyPress (symbol: {:?}, keysym: {:X}, modifiers: {:?}, text: '{}') on window {}",
                    symbol, x_keysym, modifiers, text, key_event.window
                );
                backend_events.push(BackendEvent::Key {
                    symbol,
                    modifiers,
                    text,
                });
            }
            xlib::ClientMessage => {
                // SAFETY: Accessing the `xclient` union field of `xevent` is safe because
                // the `event_type` has been confirmed to be `xlib::ClientMessage`.
                let client_message_event = unsafe { xevent.client_message };
                // Check if this is a WM_DELETE_WINDOW message.
                if client_message_event.message_type == window.protocols_atom()
                    && client_message_event.data.as_longs()[0] as xlib::Atom
                        == window.wm_delete_window_atom()
                {
                    info!(
                        "XEvent: WM_DELETE_WINDOW received from window manager for window {}.",
                        client_message_event.window
                    );
                    backend_events.push(BackendEvent::CloseRequested);
                } else {
                    trace!(
                        "XEvent: Ignored ClientMessage (type: {}, format: {}) on window {}",
                        client_message_event.message_type,
                        client_message_event.format,
                        client_message_event.window
                    );
                }
            }
            xlib::FocusIn => {
                // SAFETY: Accessing the `xfocus` (or `xcrossing` as they share layout for `type_` and `window`)
                // union field of `xevent` is safe because the `event_type` has been
                // confirmed to be `xlib::FocusIn`. `XFocusChangeEvent` (which `xfocus` is)
                // and `XCrossingEvent` (which `xcrossing` is) share common initial fields
                // like `type`, `serial`, `send_event`, `display`, and `window`.
                let focus_event = unsafe { xevent.crossing }; // xfocus could also be used
                debug!(
                    "XEvent: FocusIn (type: {}) on window {}. Setting focus true.",
                    focus_event.type_, focus_event.window
                );
                *xdriver_has_focus = true;
                backend_events.push(BackendEvent::FocusGained);
            }
            xlib::FocusOut => {
                // SAFETY: Accessing the `xfocus` (or `xcrossing`) union field of `xevent` is safe
                // because the `event_type` has been confirmed to be `xlib::FocusOut`.
                // Similar to FocusIn, the relevant common fields are safely accessible.
                let focus_event = unsafe { xevent.crossing }; // xfocus could also be used
                debug!(
                    "XEvent: FocusOut (type: {}) on window {}. Setting focus false.",
                    focus_event.type_, focus_event.window
                );
                *xdriver_has_focus = false;
                backend_events.push(BackendEvent::FocusLost);
            }
            // TODO: Add handlers for other event types as needed (e.g., ButtonPress for mouse input).
            _ => {
                trace!("XEvent: Ignored (type: {})", event_type);
            }
        }
    }
    Ok(backend_events)
}

// --- Key Translation Logic ---
// This section contains helpers to convert X11 KeySym values to the application's KeySymbol enum.

/// A helper trait for converting types that might represent an XKeySym into the canonical `xlib::KeySym`.
/// This is used by `xkeysym_to_keysymbol` to abstract over `u32` (from `x11::keysym`) and `xlib::KeySym` itself.
trait IntoXKeySym: Sized + Copy + Clone {
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

/// Translates an X11 KeySym value and associated text (from `XLookupString`)
/// into the application-defined `KeySymbol`.
///
/// This function prioritizes the text from `XLookupString` if available and valid,
/// otherwise it maps specific KeySyms to their `KeySymbol` equivalents.
///
/// # Arguments
/// * `keysym_val_in`: The X11 KeySym value (can be `xlib::KeySym` or `u32`).
/// * `text`: The text string obtained from `XLookupString` for this key event.
///
/// # Returns
/// The corresponding `KeySymbol`.
fn xkeysym_to_keysymbol<T: IntoXKeySym>(keysym_val_in: T, text: &str) -> KeySymbol {
    // Prioritize text from XLookupString if it's a valid single character.
    // This handles cases like keypad numbers when NumLock is on.
    if !text.is_empty() {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() == 1 && chars[0] != '\u{FFFD}' {
            // If XLookupString provides a single valid character, use it.
            // This is typically the case for printable characters, including those
            // from keypad when NumLock is active.
            return KeySymbol::Char(chars[0]);
        }
        // If `text` is multiple characters (e.g., from dead keys) or invalid,
        // we might still fall back to keysym matching for special keys,
        // but generally, multi-char text from a single key press is complex
        // and might be better handled as a sequence of Char events if needed,
        // or by an IME. For now, we prioritize single valid chars from `text`.
        // If `text` is not a single valid char, proceed to keysym matching.
    }

    let keysym_val = keysym_val_in.into_xkeysym();

    // Handle cases where KeySym might be a Unicode character directly,
    // especially if it's outside the typical u32 range of standard XK_* constants.
    // xlib::KeySym can be u64 on some systems.
    if keysym_val > (u32::MAX as xlib::KeySym) {
        // Large keysym with no usable text (already checked text above).
        warn!(
            "Received high keysym value: 0x{:X} with no usable text (text='{}'), mapping to Unknown.",
            keysym_val, text
        );
        return KeySymbol::Unknown;
    }

    // Standard keysyms are within u32 range.
    let keysym_u32 = keysym_val as u32;

    // Match against known X11 keysym constants.
    // This is now primarily for non-printable keys or when XLookupString yields no text.
    match keysym_u32 {
        // Modifier Keys (when the key itself is pressed, not when used as a modifier)
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

        // Editing and Control Keys
        keysym::XK_Return => KeySymbol::Enter,
        keysym::XK_KP_Enter => KeySymbol::KeypadEnter, // Keypad Enter
        keysym::XK_Linefeed => KeySymbol::Char('\n'),  // Typically 0x0A, treated as Char
        keysym::XK_BackSpace => KeySymbol::Backspace,
        keysym::XK_Tab | keysym::XK_KP_Tab | keysym::XK_ISO_Left_Tab => KeySymbol::Tab, // Tab and Keypad Tab
        keysym::XK_Escape => KeySymbol::Escape,

        // Navigation Keys
        // Note: Keypad navigation keys (e.g., XK_KP_Home) often yield the same keysym as their main
        // counterparts when NumLock is off. XLookupString might return empty for these.
        keysym::XK_Home | keysym::XK_KP_Home => KeySymbol::Home,
        keysym::XK_Left | keysym::XK_KP_Left => KeySymbol::Left,
        keysym::XK_Up | keysym::XK_KP_Up => KeySymbol::Up,
        keysym::XK_Right | keysym::XK_KP_Right => KeySymbol::Right,
        keysym::XK_Down | keysym::XK_KP_Down => KeySymbol::Down,
        keysym::XK_Page_Up | keysym::XK_KP_Page_Up => KeySymbol::PageUp,
        keysym::XK_Page_Down | keysym::XK_KP_Page_Down => KeySymbol::PageDown,
        keysym::XK_End | keysym::XK_KP_End => KeySymbol::End,
        keysym::XK_Insert | keysym::XK_KP_Insert => KeySymbol::Insert, // KP_Insert is often KP_0 with NumLock
        keysym::XK_Delete | keysym::XK_KP_Delete => KeySymbol::Delete, // KP_Delete is often KP_Decimal with NumLock

        // Function Keys F1-F24
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

        // Keypad Symbols (these are for keys that are distinctly keypad operations,
        // or when XLookupString provides no text for them, e.g., NumLock is off for digits)
        // If `text` from XLookupString is a digit ('0'-'9'), that often takes precedence below.
        keysym::XK_KP_0 => KeySymbol::Keypad0,
        keysym::XK_KP_1 => KeySymbol::Keypad1,
        keysym::XK_KP_2 => KeySymbol::Keypad2,
        keysym::XK_KP_3 => KeySymbol::Keypad3,
        keysym::XK_KP_4 => KeySymbol::Keypad4,
        keysym::XK_KP_5 | keysym::XK_KP_Begin => KeySymbol::Keypad5, // XK_KP_Begin is often KP_5
        keysym::XK_KP_6 => KeySymbol::Keypad6,
        keysym::XK_KP_7 => KeySymbol::Keypad7,
        keysym::XK_KP_8 => KeySymbol::Keypad8,
        keysym::XK_KP_9 => KeySymbol::Keypad9,
        keysym::XK_KP_Decimal | keysym::XK_KP_Separator => KeySymbol::KeypadDecimal,
        keysym::XK_KP_Add => KeySymbol::KeypadPlus,
        keysym::XK_KP_Subtract => KeySymbol::KeypadMinus,
        keysym::XK_KP_Multiply => KeySymbol::KeypadMultiply,
        keysym::XK_KP_Divide => KeySymbol::KeypadDivide,
        keysym::XK_KP_Equal => KeySymbol::KeypadEquals,
        keysym::XK_KP_Space => KeySymbol::Char(' '), // Keypad space. Text might also be " ".

        // Other Common Keys
        keysym::XK_Print | keysym::XK_Sys_Req => KeySymbol::PrintScreen, // Sys_Req is often Shift+Print
        keysym::XK_Scroll_Lock => KeySymbol::ScrollLock,
        keysym::XK_Pause | keysym::XK_Break => KeySymbol::Pause, // Break is often Ctrl+Pause
        keysym::XK_Menu => KeySymbol::Menu,

        // Fallback: If we reached here, text was empty or not a single valid char,
        // and the keysym didn't match any special keys above.
        _ => {
            // The initial check for `text` at the function start should handle most
            // KeySymbol::Char cases. If we are here, it means `text` was not suitable,
            // or the keysym is for a non-char key not explicitly mapped.
            trace!(
                "Unhandled u32 keysym 0x{:X} with text '{}', mapping to KeySymbol::Unknown",
                keysym_u32,
                text
            );
            KeySymbol::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use x11::keysym;
    // xlib::XEvent and other xlib types are not needed for these specific tests of xkeysym_to_keysymbol.

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
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_BackSpace, ""),
            KeySymbol::Backspace
        );
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_Tab, ""), KeySymbol::Tab);
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Shift_L, ""),
            KeySymbol::Shift
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Control_R, ""),
            KeySymbol::Control
        );
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_Alt_L, ""), KeySymbol::Alt);
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Super_L, ""),
            KeySymbol::Super
        );
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_Home, ""), KeySymbol::Home);
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_Left, ""), KeySymbol::Left);
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_F1, ""), KeySymbol::F1);
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_F12, ""), KeySymbol::F12);
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Delete, ""),
            KeySymbol::Delete
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_Insert, ""),
            KeySymbol::Insert
        );
    }

    #[test]
    fn test_xkeysym_to_keysymbol_char_input() {
        // XLookupString provides the char; keysym might be generic (like XK_a) or specific.
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_a, "a"),
            KeySymbol::Char('a')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_A, "A"),
            KeySymbol::Char('A')
        ); // Shifted 'a'
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_b, "b"),
            KeySymbol::Char('b')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_eacute, "é"),
            KeySymbol::Char('é')
        ); // Char from compose/dead key
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_plus, "+"),
            KeySymbol::Char('+')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_space, " "),
            KeySymbol::Char(' ')
        );
        // Keypad space often also yields " " via XLookupString.
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Space, " "),
            KeySymbol::Char(' ')
        );
    }

    #[test]
    fn test_xkeysym_to_keysymbol_keypad_numbers_with_text() {
        // If XLookupString provides text (e.g., NumLock ON), it should be KeySymbol::Char.
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_0, "0"),
            KeySymbol::Char('0')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_1, "1"),
            KeySymbol::Char('1')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Decimal, "."),
            KeySymbol::Char('.')
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Add, "+"),
            KeySymbol::Char('+')
        );
    }

    #[test]
    fn test_xkeysym_to_keysymbol_keypad_symbols_no_text() {
        // If XLookupString provides no text (e.g., NumLock OFF for navigation keys,
        // or for operator keys like KP_Add that don't always produce text),
        // the function should map to specific Keypad KeySymbol variants.

        // Navigation keys (NumLock off typically)
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Home, ""),
            KeySymbol::Home
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Left, ""),
            KeySymbol::Left
        );
        assert_eq!(xkeysym_to_keysymbol(keysym::XK_KP_Up, ""), KeySymbol::Up);
        // ... etc. for other KP_nav keys matching their non-KP counterparts.

        // Operator keys that might not produce text via XLookupString
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Add, ""),
            KeySymbol::KeypadPlus
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Subtract, ""),
            KeySymbol::KeypadMinus
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Multiply, ""),
            KeySymbol::KeypadMultiply
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Divide, ""),
            KeySymbol::KeypadDivide
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_Equal, ""),
            KeySymbol::KeypadEquals
        );

        // Digit keysyms without text (simulating NumLock off for these specific keysyms)
        // The xkeysym_to_keysymbol function has explicit mappings for XK_KP_0 through XK_KP_9.
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_0, ""),
            KeySymbol::Keypad0
        );
        assert_eq!(
            xkeysym_to_keysymbol(keysym::XK_KP_1, ""),
            KeySymbol::Keypad1
        );
    }

    // Note: Modifier key (Shift, Ctrl, Alt, Super) extraction is part of the KeyPress event
    // handling logic within `process_pending_events`, not `xkeysym_to_keysymbol` itself.
    // Thus, `test_modifier_mapping` from `x11_old.rs` is not directly applicable here.
    // The correctness of modifier flags is verified by checking the `Modifiers` field
    // in the `BackendEvent::Key` produced by `process_pending_events`.
}
