// src/keys.rs

use crate::config::Config; // Added for Config type
use crate::term::action::UserInputAction; // Added for UserInputAction type
use bitflags::bitflags;
use serde::{Deserialize, Serialize}; // Import Serialize and Deserialize

bitflags! {
    /// Represents a keyboard modifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)] // Added Serialize, Deserialize
    pub struct Modifiers: u8 {
        const SHIFT = 1 << 0;
        const CONTROL = 1 << 1;
        const ALT = 1 << 2; // Also known as Option on macOS
        const SUPER = 1 << 3; // Also known as Windows key or Command key
        const CAPS_LOCK = 1 << 4;
        const NUM_LOCK = 1 << 5;
    }
}

/// Represents a key symbol.
///
/// This enum defines all possible keypresses in the project grammar.
/// It includes common keys like alphanumeric characters, function keys,
/// modifier keys, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)] // Added Serialize, Deserialize, Default
pub enum KeySymbol {
    // Alphanumeric keys
    Char(char),

    // Function keys
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
    F21,
    F22,
    F23,
    F24,

    // Modifier keys (when pressed and released without other keys)
    Shift,
    Control,
    Alt,
    Super,
    CapsLock,
    NumLock,

    // Navigation keys
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    Insert,
    Delete,

    // Other common keys
    Enter,
    Backspace,
    Tab,
    Escape,
    PrintScreen,
    ScrollLock,
    Pause,

    // Keypad keys
    Keypad0,
    Keypad1,
    Keypad2,
    Keypad3,
    Keypad4,
    Keypad5,
    Keypad6,
    Keypad7,
    Keypad8,
    Keypad9,
    KeypadEnter,
    KeypadPlus,
    KeypadMinus,
    KeypadMultiply,
    KeypadDivide,
    KeypadDecimal,
    KeypadEquals, // Usually not present, but for completeness

    // Special keys that might not have a direct character representation
    Menu, // Context menu key

    // Unidentified key
    #[default] // Make Unknown the default variant for KeySymbol
    Unknown,
}

impl KeySymbol {
    /// Returns true if the key symbol represents a modifier key.
    pub fn is_modifier(&self) -> bool {
        matches!(
            self,
            KeySymbol::Shift
                | KeySymbol::Control
                | KeySymbol::Alt
                | KeySymbol::Super
                | KeySymbol::CapsLock
                | KeySymbol::NumLock
        )
    }
}

/// Maps a given key symbol and modifiers to a `UserInputAction` based on the provided configuration.
///
/// It iterates through the keybindings defined in `config.keybindings.bindings`.
/// If a match is found, it returns a clone of the corresponding `UserInputAction`.
/// Otherwise, it returns `None`.
pub fn map_key_event_to_action(
    key_symbol: KeySymbol,
    modifiers: Modifiers,
    config: &Config,
) -> Option<UserInputAction> {
    for binding in &config.keybindings.bindings {
        if binding.key == key_symbol && binding.mods == modifiers {
            return Some(binding.action.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*; // To import KeySymbol, Modifiers, map_key_event_to_action
                  // Also imports Config and UserInputAction if they were in `super` scope, but they are not.
    use crate::config::{Config, Keybinding, KeybindingsConfig}; // For creating mock Config and Keybinding
    use crate::term::action::UserInputAction; // For expected actions

    // Helper to create a default config with specific keybindings for testing
    fn config_with_bindings(bindings: Vec<Keybinding>) -> Config {
        let mut cfg = Config::default(); // Get a default config
        // Config has keybindings: KeybindingsConfig, and KeybindingsConfig has bindings: Vec<Keybinding>
        cfg.keybindings = KeybindingsConfig { bindings }; // Assign new KeybindingsConfig
        cfg
    }

    #[test]
    fn test_map_key_found() {
        let bindings = vec![
            Keybinding {
                key: KeySymbol::Char('C'),
                mods: Modifiers::CONTROL | Modifiers::SHIFT,
                action: UserInputAction::InitiateCopy,
            },
            Keybinding {
                key: KeySymbol::Char('Q'),
                mods: Modifiers::CONTROL,
                action: UserInputAction::RequestQuit,
            },
        ];
        let config = config_with_bindings(bindings);

        let result = map_key_event_to_action(KeySymbol::Char('C'), Modifiers::CONTROL | Modifiers::SHIFT, &config);
        assert_eq!(result, Some(UserInputAction::InitiateCopy));

        let result_quit = map_key_event_to_action(KeySymbol::Char('Q'), Modifiers::CONTROL, &config);
        assert_eq!(result_quit, Some(UserInputAction::RequestQuit));
    }

    #[test]
    fn test_map_key_not_found_symbol_mismatch() {
        let bindings = vec![Keybinding {
            key: KeySymbol::Char('C'),
            mods: Modifiers::CONTROL | Modifiers::SHIFT,
            action: UserInputAction::InitiateCopy,
        }];
        let config = config_with_bindings(bindings);

        let result = map_key_event_to_action(KeySymbol::Char('X'), Modifiers::CONTROL | Modifiers::SHIFT, &config);
        assert_eq!(result, None);
    }

    #[test]
    fn test_map_key_not_found_modifier_mismatch() {
        let bindings = vec![Keybinding {
            key: KeySymbol::Char('C'),
            mods: Modifiers::CONTROL | Modifiers::SHIFT,
            action: UserInputAction::InitiateCopy,
        }];
        let config = config_with_bindings(bindings);

        let result = map_key_event_to_action(KeySymbol::Char('C'), Modifiers::CONTROL, &config);
        assert_eq!(result, None);
    }

    #[test]
    fn test_map_key_not_found_empty_bindings() {
        let config = config_with_bindings(vec![]); // Empty bindings
        let result = map_key_event_to_action(KeySymbol::Char('C'), Modifiers::CONTROL | Modifiers::SHIFT, &config);
        assert_eq!(result, None);
    }

    #[test]
    fn test_map_key_multiple_bindings_first_match() {
        let bindings = vec![
            Keybinding { // This one should be found first
                key: KeySymbol::Char('A'),
                mods: Modifiers::ALT,
                action: UserInputAction::RequestZoomIn,
            },
            Keybinding { // This one also matches but shouldn't be reached if first is found
                key: KeySymbol::Char('A'),
                mods: Modifiers::ALT,
                action: UserInputAction::RequestZoomOut, // Different action
            },
        ];
        let config = config_with_bindings(bindings);
        let result = map_key_event_to_action(KeySymbol::Char('A'), Modifiers::ALT, &config);
        assert_eq!(result, Some(UserInputAction::RequestZoomIn));
    }
}
