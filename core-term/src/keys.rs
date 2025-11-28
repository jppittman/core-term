// src/keys.rs

use crate::config::Config;
use crate::term::action::UserInputAction;
pub use pixelflow_engine::input::{KeySymbol, Modifiers};
use log::debug;

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
    config.keybindings.bindings.iter().find_map(|binding| {
        // Use the passed config
        if binding.key == key_symbol && binding.mods == modifiers {
            debug!(
                "Keybinding: {:?} + {:?} => {:?}",
                binding.mods, binding.key, &binding.action
            );
            return Some(binding.action.clone());
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Keybinding, KeybindingsConfig};
    use crate::term::action::UserInputAction;

    fn config_with_bindings(bindings: Vec<Keybinding>) -> Config {
        let mut cfg = Config::default();
        cfg.keybindings = KeybindingsConfig { bindings };
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

        let result = map_key_event_to_action(
            KeySymbol::Char('C'),
            Modifiers::CONTROL | Modifiers::SHIFT,
            &config,
        );
        assert_eq!(result, Some(UserInputAction::InitiateCopy));

        let result_quit =
            map_key_event_to_action(KeySymbol::Char('Q'), Modifiers::CONTROL, &config);
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

        let result = map_key_event_to_action(
            KeySymbol::Char('X'),
            Modifiers::CONTROL | Modifiers::SHIFT,
            &config,
        );
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
        let config = config_with_bindings(vec![]);
        let result = map_key_event_to_action(
            KeySymbol::Char('C'),
            Modifiers::CONTROL | Modifiers::SHIFT,
            &config,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_map_key_multiple_bindings_first_match() {
        let bindings = vec![
            Keybinding {
                key: KeySymbol::Char('A'),
                mods: Modifiers::ALT,
                action: UserInputAction::RequestZoomIn,
            },
            Keybinding {
                key: KeySymbol::Char('A'),
                mods: Modifiers::ALT,
                action: UserInputAction::RequestZoomOut,
            },
        ];
        let config = config_with_bindings(bindings);
        let result = map_key_event_to_action(KeySymbol::Char('A'), Modifiers::ALT, &config);
        assert_eq!(result, Some(UserInputAction::RequestZoomIn));
    }
}
