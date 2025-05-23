// src/orchestrator.rs
//! Orchestrates the main application flow, coordinating between the PTY,
//! terminal emulator, renderer, and backend driver. This module aims to encapsulate
//! the core event processing logic, making it testable and maintainable by
//! abstracting away direct OS calls and backend specifics.

use crate::{
    ansi::AnsiParser,
<<<<<<< Updated upstream
    backends::{BackendEvent, Driver, Modifiers as BackendModifiers, Keycode}, // Assuming Modifiers is part of BackendEvent or Driver
    config::{Config, KeySymbol, Modifiers as ConfigModifiers}, // Renamed to avoid conflict
=======
    backends::{BackendEvent, Driver}, // Modifiers and Keycode removed
    config::{Config, KeySymbol, Modifiers}, // Using Modifiers from config directly
>>>>>>> Stashed changes
    os::pty::PtyChannel,
    renderer::Renderer,
    term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalInterface, UserInputAction},
};
use anyhow::Error as AnyhowError;
use arboard; // Import arboard
use std::io::ErrorKind as IoErrorKind;

const PTY_READ_BUFFER_SIZE: usize = 4096;

/// Represents the status of the orchestrator after processing an event or an iteration of its loop.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OrchestratorStatus {
    /// The orchestrator processed events successfully and should continue running.
    Running,
    /// A shutdown signal was received (e.g., PTY EOF, Quit event from driver).
    /// The application should terminate gracefully.
    Shutdown,
}

/// Encapsulates the main application state and logic for orchestrating terminal components.
///
/// This struct uses trait objects for its main dependencies (`PtyChannel`, `TerminalInterface`,
/// `Driver`) to allow for mocking in tests and flexibility in choosing
/// concrete implementations. The `Renderer` is a concrete type.
pub struct AppOrchestrator<'a> {
    pty_channel: &'a mut dyn PtyChannel,
    term: &'a mut dyn TerminalInterface,
    parser: &'a mut dyn AnsiParser,
    pub renderer: Renderer,
    pub driver: &'a mut dyn Driver,
    config: &'a Config, // Added config field
    pty_read_buffer: [u8; PTY_READ_BUFFER_SIZE],
}

impl<'a> AppOrchestrator<'a> {
    /// Creates a new `AppOrchestrator`.
    pub fn new(
        pty_channel: &'a mut dyn PtyChannel,
        term: &'a mut dyn TerminalInterface,
        parser: &'a mut dyn AnsiParser,
        renderer: Renderer,
        driver: &'a mut dyn Driver,
        config: &'a Config, // Added config parameter
    ) -> Self {
        AppOrchestrator {
            pty_channel,
            term,
            parser,
            renderer,
            driver,
            config, // Initialize config field
            pty_read_buffer: [0; PTY_READ_BUFFER_SIZE],
        }
    }

    // Helper function to translate backend modifiers to config modifiers
<<<<<<< Updated upstream
    fn translate_modifiers(&self, backend_modifiers: BackendModifiers) -> ConfigModifiers {
        let mut config_mods = ConfigModifiers::empty();
        if backend_modifiers.contains(BackendModifiers::SHIFT) {
            config_mods |= ConfigModifiers::SHIFT;
        }
        if backend_modifiers.contains(BackendModifiers::CONTROL) {
            config_mods |= ConfigModifiers::CONTROL;
        }
        if backend_modifiers.contains(BackendModifiers::ALT) {
            config_mods |= ConfigModifiers::ALT;
        }
        if backend_modifiers.contains(BackendModifiers::SUPER) {
            config_mods |= ConfigModifiers::SUPER;
        }
        config_mods
=======
    fn translate_modifiers(&self, backend_modifiers: Modifiers) -> Modifiers {
        // backend_modifiers is already of type config::Modifiers, so direct assignment or selective mapping.
        // For now, this function might be redundant if BackendModifiers is already config::Modifiers.
        // However, if backend_modifiers came from a *different* Modifiers type (e.g. from a specific backend crate),
        // then this translation would be necessary.
        // Assuming backend_modifiers is already the correct type config::Modifiers.
        // If not, the commented-out logic below would be a starting point.

        // let mut config_mods = Modifiers::empty();
        // if backend_modifiers.contains(MOD_SHIFT_NAME_FROM_OTHER_CRATE) { // Example
        //     config_mods |= Modifiers::SHIFT;
        // }
        // // ... etc.
        // config_mods
        backend_modifiers // Assuming it's already the correct type.
>>>>>>> Stashed changes
    }

    // Helper function to translate backend keysym to config KeySymbol
    // This is a simplified version for now.
<<<<<<< Updated upstream
    fn translate_keysym(&self, keysym: Keycode, text: &Option<String>) -> KeySymbol {
        match keysym {
            // This mapping is highly dependent on the values provided by the specific Driver/BackendEvent
            // For now, assuming simple character mapping and some common keys.
            // A more robust solution would involve a comprehensive mapping based on XKB or similar.
            k if k >= Keycode::A && k <= Keycode::Z => {
                // Assuming Keycode::A represents 'A'
                // This needs to be adjusted based on actual Keycode values
                if let Some(txt) = text {
                    if !txt.is_empty() {
                        return KeySymbol::Char(txt.chars().next().unwrap_or('?'));
                    }
                }
                // Fallback if text is None or empty, try to derive from keysym if possible
                // This is a placeholder for a more complex mapping
                KeySymbol::Char((k as u8 - Keycode::A as u8 + b'A') as char)
            }
            k if k >= Keycode::Key0 && k <= Keycode::Key9 => {
                 if let Some(txt) = text {
                    if !txt.is_empty() {
                        return KeySymbol::Char(txt.chars().next().unwrap_or('?'));
                    }
                }
                KeySymbol::Char((k as u8 - Keycode::Key0 as u8 + b'0') as char)
            }
            Keycode::Backspace => KeySymbol::Backspace,
            Keycode::Tab => KeySymbol::Tab,
            Keycode::Return => KeySymbol::Return,
            Keycode::Escape => KeySymbol::Escape,
            Keycode::Space => KeySymbol::Space,
            Keycode::PageUp => KeySymbol::PageUp,
            Keycode::PageDown => KeySymbol::PageDown,
            Keycode::End => KeySymbol::End,
            Keycode::Home => KeySymbol::Home,
            Keycode::Left => KeySymbol::Left,
            Keycode::Up => KeySymbol::Up,
            Keycode::Right => KeySymbol::Right,
            Keycode::Down => KeySymbol::Down,
            Keycode::Insert => KeySymbol::Insert,
            Keycode::Delete => KeySymbol::Delete,
            Keycode::F1 => KeySymbol::F1,
            Keycode::F2 => KeySymbol::F2,
            Keycode::F3 => KeySymbol::F3,
            Keycode::F4 => KeySymbol::F4,
            Keycode::F5 => KeySymbol::F5,
            Keycode::F6 => KeySymbol::F6,
            Keycode::F7 => KeySymbol::F7,
            Keycode::F8 => KeySymbol::F8,
            Keycode::F9 => KeySymbol::F9,
            Keycode::F10 => KeySymbol::F10,
            Keycode::F11 => KeySymbol::F11,
            Keycode::F12 => KeySymbol::F12,
            // Add more mappings as needed
            _ => {
                // If text is available and represents a single char, use it.
                if let Some(txt) = text {
                    let mut chars = txt.chars();
                    if let (Some(c), None) = (chars.next(), chars.next()) {
                        return KeySymbol::Char(c);
                    }
                }
                KeySymbol::Unknown(keysym.0) // Store the raw keysym if unknown
            }
=======
    // The `keysym: Keycode` parameter will cause an error as Keycode is no longer imported.
    // This function needs to be refactored to take `keysym_u32: u32`.
    fn translate_keysym(&self, keysym_u32: u32, text: &Option<String>) -> KeySymbol {
        // If `text` provides a single character, it's often the most reliable representation,
        // especially considering IMEs or complex keyboard layouts.
        if let Some(txt_str) = text {
            let mut chars = txt_str.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                return KeySymbol::Char(c);
            }
            // If `text` is not a single char (e.g., empty for control keys, or multiple chars for dead keys),
            // then proceed to keysym mapping.
        }

        // This mapping is highly dependent on the values provided by the specific Driver/BackendEvent.
        // These are common X11 KeySym values. Other backends might use different values.
        // A more robust solution would involve the backend providing an abstract KeySymbol directly,
        // or using a more comprehensive keysym library.
        // Ref: /usr/include/X11/keysymdef.h
        const XK_BACKSPACE: u32 = 0xff08;
        const XK_TAB: u32 = 0xff09;
        const XK_RETURN: u32 = 0xff0d; // Enter key
        const XK_ESCAPE: u32 = 0xff1b;
        const XK_INSERT: u32 = 0xff63;
        const XK_DELETE: u32 = 0xffff;
        const XK_HOME: u32 = 0xff50;
        const XK_END: u32 = 0xff57;
        const XK_PAGE_UP: u32 = 0xff55;
        const XK_PAGE_DOWN: u32 = 0xff56;
        const XK_LEFT: u32 = 0xff51;
        const XK_UP: u32 = 0xff52;
        const XK_RIGHT: u32 = 0xff53;
        const XK_DOWN: u32 = 0xff54;
        const XK_F1: u32 = 0xffbe;
        const XK_F2: u32 = 0xffbf;
        const XK_F3: u32 = 0xffc0;
        const XK_F4: u32 = 0xffc1;
        const XK_F5: u32 = 0xffc2;
        const XK_F6: u32 = 0xffc3;
        const XK_F7: u32 = 0xffc4;
        const XK_F8: u32 = 0xffc5;
        const XK_F9: u32 = 0xffc6;
        const XK_F10: u32 = 0xffc7;
        const XK_F11: u32 = 0xffc8;
        const XK_F12: u32 = 0xffc9;
        // Space is tricky: X11 keysym is 0x020, but `text` field should have " ".
        // ASCII space is 32. If text field failed, this might be a fallback.
        const XK_SPACE: u32 = 0x0020;


        match keysym_u32 {
            XK_BACKSPACE => KeySymbol::Backspace,
            XK_TAB => KeySymbol::Tab,
            XK_RETURN => KeySymbol::Return,
            XK_ESCAPE => KeySymbol::Escape,
            XK_INSERT => KeySymbol::Insert,
            XK_DELETE => KeySymbol::Delete,
            XK_HOME => KeySymbol::Home,
            XK_END => KeySymbol::End,
            XK_PAGE_UP => KeySymbol::PageUp,
            XK_PAGE_DOWN => KeySymbol::PageDown,
            XK_LEFT => KeySymbol::Left,
            XK_UP => KeySymbol::Up,
            XK_RIGHT => KeySymbol::Right,
            XK_DOWN => KeySymbol::Down,
            XK_F1 => KeySymbol::F1,
            XK_F2 => KeySymbol::F2,
            XK_F3 => KeySymbol::F3,
            XK_F4 => KeySymbol::F4,
            XK_F5 => KeySymbol::F5,
            XK_F6 => KeySymbol::F6,
            XK_F7 => KeySymbol::F7,
            XK_F8 => KeySymbol::F8,
            XK_F9 => KeySymbol::F9,
            XK_F10 => KeySymbol::F10,
            XK_F11 => KeySymbol::F11,
            XK_F12 => KeySymbol::F12,
            XK_SPACE => KeySymbol::Char(' '), // Prefer text field, but map if text is empty.
            // Add more non-character keysyms here as needed.
            // For printable characters, text field is preferred.
            // If text field was empty but keysym indicates a printable char (e.g. 'a' to 'z'),
            // that logic could be added here, but it's less common.
            _ => KeySymbol::Unknown(keysym_u32),
>>>>>>> Stashed changes
        }
    }


    pub fn process_pty_events(&mut self) -> Result<OrchestratorStatus, AnyhowError> {
        log::trace!("Orchestrator: Processing available PTY data...");
        match self.pty_channel.read(&mut self.pty_read_buffer) {
            Ok(0) => {
                log::info!("Orchestrator: PTY EOF received. Signaling shutdown.");
                Ok(OrchestratorStatus::Shutdown)
            }
            Ok(count) => {
                log::debug!("Orchestrator: Read {} bytes from PTY.", count);
                let data_slice = &self.pty_read_buffer[..count];
                let pty_data_copy = data_slice.to_vec();
                self.interpret_pty_bytes_mut_access(&pty_data_copy);
                Ok(OrchestratorStatus::Running)
            }
            Err(e) if e.kind() == IoErrorKind::WouldBlock => {
                log::trace!("Orchestrator: PTY read would block (no new data available).");
                Ok(OrchestratorStatus::Running)
            }
            Err(e) => {
                log::error!("Orchestrator: Unrecoverable error reading from PTY: {}", e);
                Err(AnyhowError::from(e).context("PTY read error"))
            }
        }
    }

    fn interpret_pty_bytes_mut_access(&mut self, pty_data_slice: &[u8]) {
        let commands = self
            .parser
            .process_bytes(pty_data_slice)
            .into_iter()
            .map(EmulatorInput::Ansi);

        for command_input in commands {
            if let Some(action) = self.term.interpret_input(command_input) {
                self.handle_emulator_action(action);
            }
        }
    }

    pub fn process_driver_events(&mut self) -> Result<OrchestratorStatus, String> {
        log::trace!("Orchestrator: Processing available driver events...");
        let events = self.driver.process_events().map_err(|e| {
            let err_msg = format!("Orchestrator: Driver error processing events: {}", e);
            log::error!("{}", err_msg);
            err_msg
        })?;

        if events.is_empty() {
            log::trace!("Orchestrator: No new driver events.");
            return Ok(OrchestratorStatus::Running);
        }

        for event in events {
            log::debug!("Orchestrator: Handling BackendEvent: {:?}", event);
            if event == BackendEvent::CloseRequested {
                log::info!("Orchestrator: CloseRequested event received. Signaling shutdown.");
                return Ok(OrchestratorStatus::Shutdown);
            }
            self.handle_specific_driver_event(event);
        }
        Ok(OrchestratorStatus::Running)
    }

    fn handle_specific_driver_event(&mut self, event: BackendEvent) {
        match event {
            BackendEvent::Key {
                keysym,
                modifiers,
                text,
            } => {
<<<<<<< Updated upstream
                let translated_symbol = self.translate_keysym(keysym, &text);
=======
                let text_option = Some(text.clone()); // Create an Option<String> for translate_keysym
                let translated_symbol = self.translate_keysym(keysym, &text_option);
>>>>>>> Stashed changes
                let translated_modifiers = self.translate_modifiers(modifiers);

                let copy_binding = self.config.keybindings.copy;
                let paste_binding = self.config.keybindings.paste;

                let mut user_input_action = None;

                if translated_symbol == copy_binding.symbol
                    && translated_modifiers == copy_binding.modifiers
                {
                    user_input_action = Some(UserInputAction::InitiateCopy);
                } else if translated_symbol == paste_binding.symbol
                    && translated_modifiers == paste_binding.modifiers
                {
                    user_input_action = Some(UserInputAction::InitiatePaste);
                } else {
                    // Regular key input
                    user_input_action = Some(UserInputAction::KeyInput {
                        symbol: translated_symbol,
                        modifiers: translated_modifiers,
<<<<<<< Updated upstream
                        text, // Pass along the original text from backend for PTY
=======
                        text: Some(text), // Pass along the original text from backend for PTY
>>>>>>> Stashed changes
                    });
                }

                if let Some(uia) = user_input_action {
                    if let Some(action) = self.term.interpret_input(EmulatorInput::User(uia)) {
                        self.handle_emulator_action(action);
                    }
                }
            }
            BackendEvent::Resize {
                width_px,
                height_px,
            } => {
                let (char_width, char_height) = self.driver.get_font_dimensions();
                if char_width == 0 || char_height == 0 {
                    log::warn!(
                        "Orchestrator: Received resize but driver reported zero char dimensions ({}, {}). Ignoring resize.",
                        char_width,
                        char_height
                    );
                    return;
                }

                let new_cols = (width_px as usize / char_width.max(1)).max(1);
                let new_rows = (height_px as usize / char_height.max(1)).max(1);

                log::info!(
                    "Orchestrator: Resizing to {}x{} cells ({}x{} px, char_size: {}x{})",
                    new_cols,
                    new_rows,
                    width_px,
                    height_px,
                    char_width,
                    char_height
                );

                if let Err(e) = self.pty_channel.resize(new_cols as u16, new_rows as u16) {
                    log::warn!(
                        "Orchestrator: Failed to resize PTY to {}x{}: {}",
                        new_cols,
                        new_rows,
                        e
                    );
                }

                let resize_event = EmulatorInput::Control(ControlEvent::Resize {
                    cols: new_cols,
                    rows: new_rows,
                });
                if let Some(action) = self.term.interpret_input(resize_event) {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::FocusGained => {
                log::debug!("Orchestrator: FocusGained event.");
                self.driver.set_focus(true);
                if let Some(action) = self
                    .term
                    .interpret_input(EmulatorInput::User(UserInputAction::FocusGained))
                {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::FocusLost => {
                log::debug!("Orchestrator: FocusLost event.");
                self.driver.set_focus(false);
                if let Some(action) = self
                    .term
                    .interpret_input(EmulatorInput::User(UserInputAction::FocusLost))
                {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::CloseRequested => {
                log::warn!(
                    "Orchestrator: CloseRequested event unexpectedly reached handle_specific_driver_event."
                );
            }
            BackendEvent::Mouse { event_type, button, x, y, modifiers } => {
                // TODO: Translate BackendEvent::Mouse into UserInputAction::MouseInput
                //       and then pass it to self.term.interpret_input or a dedicated handler.
                //       This involves mapping cell coordinates if x,y are pixels,
                //       and deciding how/if to use button and modifiers here.
                log::debug!(
                    "Orchestrator: Received BackendEvent::Mouse: type={:?}, button={:?}, x={}, y={}, mods={:?}",
                    event_type, button, x, y, modifiers
                );
                // Example of how it might be translated and sent to terminal:
                // let (char_width, char_height) = self.driver.get_font_dimensions();
                // let cell_x = x / char_width.max(1);
                // let cell_y = y / char_height.max(1);
                // let uia = UserInputAction::MouseInput { /* ... */ };
                // if let Some(action) = self.term.interpret_input(EmulatorInput::User(uia)) {
                //     self.handle_emulator_action(action);
                // }
            }
        }
    }

    /// Handles actions signaled by the `TerminalInterface` implementation.
    fn handle_emulator_action(&mut self, action: EmulatorAction) {
        log::debug!("Orchestrator: Handling EmulatorAction: {:?}", action);
        match action {
            EmulatorAction::WritePty(data) => {
                if let Err(e) = self.pty_channel.write_all(&data) {
                    log::error!(
                        "Orchestrator: Failed to write_all {} bytes to PTY: {}",
                        data.len(),
                        e
                    );
                } else {
                    log::trace!("Orchestrator: Wrote {} bytes to PTY.", data.len());
                }
            }
            EmulatorAction::SetTitle(title) => {
                self.driver.set_title(&title);
            }
            EmulatorAction::RingBell => {
                self.driver.bell();
            }
            EmulatorAction::RequestRedraw => {
                log::trace!("Orchestrator: EmulatorAction::RequestRedraw received (now implicit).");
            }
            EmulatorAction::SetCursorVisibility(visible) => {
                log::trace!(
                    "Orchestrator: Setting driver cursor visibility to: {}",
                    visible
                );
                self.driver.set_cursor_visibility(visible);
            }
            EmulatorAction::CopyToClipboard(text) => {
                match arboard::Clipboard::new() {
                    Ok(mut clipboard) => {
                        if let Err(e) = clipboard.set_text(text) {
                            log::error!("Orchestrator: Failed to copy to clipboard: {}", e);
                        } else {
                            log::info!("Orchestrator: Copied text to clipboard.");
                        }
                    }
                    Err(e) => {
                        log::error!("Orchestrator: Failed to initialize clipboard for copy: {}", e);
                    }
                }
            }
            EmulatorAction::RequestClipboardContent => {
                match arboard::Clipboard::new() {
                    Ok(mut clipboard) => {
                        match clipboard.get_text() {
                            Ok(text) => {
                                log::info!("Orchestrator: Pasting text from clipboard.");
                                let paste_input =
                                    EmulatorInput::User(UserInputAction::PasteText(text));
                                // Recursively call interpret_input -> handle_emulator_action
                                // This is okay for one level. Deeper recursion could be an issue.
                                if let Some(action) = self.term.interpret_input(paste_input) {
                                    self.handle_emulator_action(action);
                                }
                            }
                            Err(e) => {
                                log::error!("Orchestrator: Failed to get text from clipboard: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        log::error!(
                            "Orchestrator: Failed to initialize clipboard for paste: {}",
                            e
                        );
                    }
                }
            }
        }
    }

    pub fn render_if_needed(&mut self) -> anyhow::Result<()> {
        log::trace!("Orchestrator: Calling renderer.draw().");
        self.renderer.draw(&mut *self.term, &mut *self.driver)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

