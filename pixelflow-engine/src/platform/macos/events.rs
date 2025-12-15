//! Event Mapper
//!
//! Pure functions to map `NSEvent` to `DisplayEvent`.

use crate::api::private::WindowId;
use crate::display::messages::DisplayEvent;
use crate::input::{KeySymbol, Modifiers};
use crate::platform::macos::cocoa::{self, event_type, NSEvent};

/// Maps an `NSEvent` to a `DisplayEvent`, if applicable.
/// `window_height` is needed to flip Y coordinates (macOS origin is bottom-left).
pub fn map_event(event: NSEvent, window_height: f64) -> Option<DisplayEvent> {
    let ty = event.type_();

    match ty {
        event_type::KEY_DOWN => {
            let code = event.key_code();
            let text = event.characters();
            let mods = map_modifiers(event.modifier_flags());

            // Pressed is always true for KEY_DOWN.
            Some(DisplayEvent::Key {
                id: WindowId::PRIMARY,
                symbol: map_key(code),
                modifiers: mods,
                text: Some(text),
            })
        }
        event_type::LEFT_MOUSE_DOWN | event_type::RIGHT_MOUSE_DOWN => {
            let pos = event.location_in_window();
            let button = match ty {
                event_type::LEFT_MOUSE_DOWN => 1,
                event_type::RIGHT_MOUSE_DOWN => 3,
                _ => 1,
            };
            Some(DisplayEvent::MouseButtonPress {
                id: WindowId::PRIMARY,
                button,
                x: pos.x as i32,
                y: (window_height - pos.y) as i32,
                modifiers: map_modifiers(event.modifier_flags()),
            })
        }
        event_type::LEFT_MOUSE_UP | event_type::RIGHT_MOUSE_UP => {
            let pos = event.location_in_window();
            let button = match ty {
                event_type::LEFT_MOUSE_UP => 1,
                event_type::RIGHT_MOUSE_UP => 3,
                _ => 1,
            };
            Some(DisplayEvent::MouseButtonRelease {
                id: WindowId::PRIMARY,
                button,
                x: pos.x as i32,
                y: (window_height - pos.y) as i32,
                modifiers: map_modifiers(event.modifier_flags()),
            })
        }
        event_type::MOUSE_MOVED
        | event_type::LEFT_MOUSE_DRAGGED
        | event_type::RIGHT_MOUSE_DRAGGED => {
            let pos = event.location_in_window();
            Some(DisplayEvent::MouseMove {
                id: WindowId::PRIMARY,
                x: pos.x as i32,
                y: (window_height - pos.y) as i32,
                modifiers: map_modifiers(event.modifier_flags()),
            })
        }
        event_type::SCROLL_WHEEL => {
            // TODO: Scroll mapping requires deltaX/Y from cocoa wrapper
            None
        }
        _ => None,
    }
}

fn map_modifiers(flags: u64) -> Modifiers {
    let mut m = Modifiers::empty();
    // NSEventModifierFlags constants
    const NS_EVENT_MODIFIER_FLAG_SHIFT: u64 = 1 << 17;
    const NS_EVENT_MODIFIER_FLAG_CONTROL: u64 = 1 << 18;
    const NS_EVENT_MODIFIER_FLAG_OPTION: u64 = 1 << 19;
    const NS_EVENT_MODIFIER_FLAG_COMMAND: u64 = 1 << 20;

    if flags & NS_EVENT_MODIFIER_FLAG_SHIFT != 0 {
        m |= Modifiers::SHIFT;
    }
    if flags & NS_EVENT_MODIFIER_FLAG_CONTROL != 0 {
        m |= Modifiers::CONTROL;
    }
    if flags & NS_EVENT_MODIFIER_FLAG_OPTION != 0 {
        m |= Modifiers::ALT;
    }
    if flags & NS_EVENT_MODIFIER_FLAG_COMMAND != 0 {
        m |= Modifiers::SUPER;
    }
    m
}

fn map_key(code: u16) -> KeySymbol {
    match code {
        0x00 => KeySymbol::Char('a'),
        0x01 => KeySymbol::Char('s'),
        0x02 => KeySymbol::Char('d'),
        0x03 => KeySymbol::Char('f'),
        0x04 => KeySymbol::Char('h'),
        0x05 => KeySymbol::Char('g'),
        0x06 => KeySymbol::Char('z'),
        0x07 => KeySymbol::Char('x'),
        0x08 => KeySymbol::Char('c'),
        0x09 => KeySymbol::Char('v'),
        0x0B => KeySymbol::Char('b'),
        0x0C => KeySymbol::Char('q'),
        0x0D => KeySymbol::Char('w'),
        0x0E => KeySymbol::Char('e'),
        0x0F => KeySymbol::Char('r'),
        0x10 => KeySymbol::Char('y'),
        0x11 => KeySymbol::Char('t'),
        0x12 => KeySymbol::Char('1'),
        0x13 => KeySymbol::Char('2'),
        0x14 => KeySymbol::Char('3'),
        0x15 => KeySymbol::Char('4'),
        0x16 => KeySymbol::Char('6'),
        0x17 => KeySymbol::Char('5'),
        0x18 => KeySymbol::Char('='), // Equal?
        0x19 => KeySymbol::Char('9'),
        0x1A => KeySymbol::Char('7'),
        0x1B => KeySymbol::Char('-'),
        0x1C => KeySymbol::Char('8'),
        0x1D => KeySymbol::Char('0'),
        0x1E => KeySymbol::Char(']'),
        0x1F => KeySymbol::Char('o'),
        0x20 => KeySymbol::Char('u'),
        0x21 => KeySymbol::Char('['),
        0x22 => KeySymbol::Char('i'),
        0x23 => KeySymbol::Char('p'),
        0x24 => KeySymbol::Enter,
        0x25 => KeySymbol::Char('l'),
        0x26 => KeySymbol::Char('j'),
        0x27 => KeySymbol::Char('\''),
        0x28 => KeySymbol::Char('k'),
        0x29 => KeySymbol::Char(';'),
        0x2A => KeySymbol::Char('\\'),
        0x2B => KeySymbol::Char(','),
        0x2C => KeySymbol::Char('/'),
        0x2D => KeySymbol::Char('n'),
        0x2E => KeySymbol::Char('m'),
        0x2F => KeySymbol::Char('.'),
        0x30 => KeySymbol::Tab,
        0x31 => KeySymbol::Char(' '), // Space
        0x32 => KeySymbol::Char('`'),
        0x33 => KeySymbol::Backspace,
        0x35 => KeySymbol::Escape,
        // Arrows (Need to verify keycodes, these are common mac codes)
        0x7B => KeySymbol::Left,
        0x7C => KeySymbol::Right,
        0x7D => KeySymbol::Down,
        0x7E => KeySymbol::Up,
        _ => KeySymbol::Unknown,
    }
}
