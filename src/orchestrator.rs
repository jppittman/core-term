// src/orchestrator.rs
//! Orchestrates the main application flow, coordinating between the PTY,
//! terminal emulator, renderer, and backend driver. This module aims to encapsulate
//! the core event processing logic, making it testable and maintainable by
//! abstracting away direct OS calls and backend specifics.

use crate::{
    ansi::AnsiParser,
    backends::{BackendEvent, Driver}, 
    config::{Config, KeySymbol, Modifiers}, 
    os::pty::PtyChannel,
    renderer::Renderer,
    term::{ControlEvent, EmulatorAction, EmulatorInput, TerminalInterface, UserInputAction},
};
use anyhow::Error as AnyhowError;
use arboard; // For clipboard interactions
use std::io::ErrorKind as IoErrorKind;

/// Buffer size for reading from the PTY.
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
    config: &'a Config,
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
        config: &'a Config,
    ) -> Self {
        AppOrchestrator {
            pty_channel,
            term,
            parser,
            renderer,
            driver,
            config,
            pty_read_buffer: [0; PTY_READ_BUFFER_SIZE],
        }
    }

    /// Translates backend-specific modifier flags to the application's `Modifiers` representation.
    ///
    /// Currently, this function assumes that the `Modifiers` type provided by the backend
    /// is already compatible with `crate::config::Modifiers`. If a backend provides modifiers
    /// in a different format, this function would need to perform the actual translation logic.
    ///
    /// # Arguments
    /// * `backend_modifiers`: The modifier flags received from the backend driver.
    ///
    /// # Returns
    /// The translated `Modifiers` compatible with the application's configuration.
    fn translate_modifiers(&self, backend_modifiers: Modifiers) -> Modifiers {
        // Assuming backend_modifiers is already of type config::Modifiers.
        // If not, translation logic would be needed here.
        // Example:
        // let mut config_mods = Modifiers::empty();
        // if backend_modifiers.contains(BACKEND_SHIFT_FLAG) { config_mods |= Modifiers::SHIFT; }
        // // ... and so on for other modifiers ...
        // config_mods
        backend_modifiers 
    }

    /// Translates a backend-specific keysym (key symbol) and optional text input
    /// into the application's `KeySymbol` representation.
    ///
    /// This function prioritizes the `text` input if it represents a single character,
    /// as this is often more reliable for international layouts or IMEs. If `text` is
    /// not a single character, it falls back to mapping common `keysym_u32` values
    /// (based on X11 KeySyms) to corresponding `KeySymbol` variants.
    ///
    /// # Arguments
    /// * `keysym_u32`: The raw keysym value (e.g., from X11) provided by the backend.
    /// * `text`: Optional text associated with the key event (e.g., "a", "A", or complex IME output).
    ///
    /// # Returns
    /// The translated `KeySymbol`. If the keysym is not recognized and `text` is not
    /// a single character, it returns `KeySymbol::Unknown` with the original `keysym_u32`.
    fn translate_keysym(&self, keysym_u32: u32, text: &Option<String>) -> KeySymbol {
        // Prefer single character text if available, as it's often more accurate for IMEs.
        if let Some(txt_str) = text {
            let mut chars = txt_str.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) { // Check if it's a single char
                return KeySymbol::Char(c);
            }
        }

        // X11 KeySym constants for common non-character keys.
        // Ref: /usr/include/X11/keysymdef.h or similar sources.
        const XK_BACKSPACE: u32 = 0xff08;
        const XK_TAB: u32 = 0xff09;
        const XK_RETURN: u32 = 0xff0d;
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
        const XK_F1: u32 = 0xffbe; const XK_F2: u32 = 0xffbf; const XK_F3: u32 = 0xffc0;
        const XK_F4: u32 = 0xffc1; const XK_F5: u32 = 0xffc2; const XK_F6: u32 = 0xffc3;
        const XK_F7: u32 = 0xffc4; const XK_F8: u32 = 0xffc5; const XK_F9: u32 = 0xffc6;
        const XK_F10: u32 = 0xffc7; const XK_F11: u32 = 0xffc8; const XK_F12: u32 = 0xffc9;
        const XK_SPACE: u32 = 0x0020; // Note: text field should usually provide ' ' for space.

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
            XK_F1 => KeySymbol::F1, XK_F2 => KeySymbol::F2, XK_F3 => KeySymbol::F3,
            XK_F4 => KeySymbol::F4, XK_F5 => KeySymbol::F5, XK_F6 => KeySymbol::F6,
            XK_F7 => KeySymbol::F7, XK_F8 => KeySymbol::F8, XK_F9 => KeySymbol::F9,
            XK_F10 => KeySymbol::F10, XK_F11 => KeySymbol::F11, XK_F12 => KeySymbol::F12,
            XK_SPACE => KeySymbol::Char(' '), // Fallback if text field was empty for space.
            // For other keysyms that might represent printable characters but text was empty,
            // further mapping could be added here (e.g., 0x0041 for 'A').
            // However, relying on the `text` field for printables is generally preferred.
            _ => KeySymbol::Unknown(keysym_u32),
        }
    }

    /// Processes data read from the PTY.
    /// Data is parsed into ANSI commands, which are then interpreted by the terminal emulator.
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
                // Create a copy for safe processing, as buffer might be reused.
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

    /// Interprets a slice of bytes read from the PTY.
    /// This involves parsing the bytes into ANSI commands and then feeding them
    /// into the terminal emulator.
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

    /// Processes events received from the backend driver.
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
            // Early exit for CloseRequested to prevent further processing.
            if event == BackendEvent::CloseRequested {
                log::info!("Orchestrator: CloseRequested event received. Signaling shutdown.");
                return Ok(OrchestratorStatus::Shutdown);
            }
            self.handle_specific_driver_event(event);
        }
        Ok(OrchestratorStatus::Running)
    }

    /// Handles a specific `BackendEvent` from the driver.
    fn handle_specific_driver_event(&mut self, event: BackendEvent) {
        match event {
            BackendEvent::Key { keysym, modifiers, text, } => {
                // Translate backend-specific key information to application-defined types.
                let text_option = if text.is_empty() { None } else { Some(text.clone()) };
                let translated_symbol = self.translate_keysym(keysym, &text_option);
                let translated_modifiers = self.translate_modifiers(modifiers);

                let mut user_input_action = None;

                // Check against configured keybindings for copy and paste.
                let copy_binding = self.config.keybindings.copy;
                let paste_binding = self.config.keybindings.paste;

                if translated_symbol == copy_binding.symbol && translated_modifiers == copy_binding.modifiers {
                    log::debug!("Copy keybinding matched: {:?}, {:?}", translated_symbol, translated_modifiers);
                    user_input_action = Some(UserInputAction::InitiateCopy);
                } else if translated_symbol == paste_binding.symbol && translated_modifiers == paste_binding.modifiers {
                    log::debug!("Paste keybinding matched: {:?}, {:?}", translated_symbol, translated_modifiers);
                    user_input_action = Some(UserInputAction::InitiatePaste);
                } else {
                    // If not a copy/paste binding, treat as regular key input.
                    let key_input_text = if text.is_empty() { None } else { Some(text) };
                    user_input_action = Some(UserInputAction::KeyInput {
                        symbol: translated_symbol,
                        modifiers: translated_modifiers,
                        text: key_input_text,
                    });
                }

                // Send the determined UserInputAction to the terminal emulator.
                if let Some(uia) = user_input_action {
                    if let Some(action) = self.term.interpret_input(EmulatorInput::User(uia)) {
                        self.handle_emulator_action(action);
                    }
                }
            }
            BackendEvent::Resize { width_px, height_px, } => {
                let (char_width, char_height) = self.driver.get_font_dimensions();
                if char_width == 0 || char_height == 0 {
                    log::warn!("Orchestrator: Received resize but driver reported zero char dimensions ({}, {}). Ignoring resize.", char_width, char_height);
                    return;
                }

                let new_cols = (width_px as usize / char_width.max(1)).max(1);
                let new_rows = (height_px as usize / char_height.max(1)).max(1);

                log::info!("Orchestrator: Resizing to {}x{} cells ({}x{} px, char_size: {}x{})", new_cols, new_rows, width_px, height_px, char_width, char_height);

                if let Err(e) = self.pty_channel.resize(new_cols as u16, new_rows as u16) {
                    log::warn!("Orchestrator: Failed to resize PTY to {}x{}: {}", new_cols, new_rows, e);
                }

                let resize_event = EmulatorInput::Control(ControlEvent::Resize { cols: new_cols, rows: new_rows });
                if let Some(action) = self.term.interpret_input(resize_event) {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::FocusGained => {
                log::debug!("Orchestrator: FocusGained event.");
                self.driver.set_focus(true); // Inform driver of focus change.
                // Inform terminal emulator about focus change.
                if let Some(action) = self.term.interpret_input(EmulatorInput::User(UserInputAction::FocusGained)) {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::FocusLost => {
                log::debug!("Orchestrator: FocusLost event.");
                self.driver.set_focus(false); // Inform driver of focus change.
                // Inform terminal emulator about focus change.
                if let Some(action) = self.term.interpret_input(EmulatorInput::User(UserInputAction::FocusLost)) {
                    self.handle_emulator_action(action);
                }
            }
            BackendEvent::CloseRequested => {
                // This case should be handled in `process_driver_events` for immediate shutdown.
                // If reached here, it's unexpected.
                log::warn!("Orchestrator: CloseRequested event unexpectedly reached handle_specific_driver_event.");
            }
            BackendEvent::Mouse { event_type, button, x, y, modifiers } => {
                log::debug!("Orchestrator: Received BackendEvent::Mouse: type={:?}, button={:?}, x={}, y={}, mods={:?}", event_type, button, x, y, modifiers);
                // TODO: Translate mouse event to UserInputAction::MouseInput and send to terminal.
                // This will involve converting pixel coordinates (x,y) to cell coordinates.
                // let (char_width, char_height) = self.driver.get_font_dimensions();
                // let cell_x = x / char_width.max(1);
                // let cell_y = y / char_height.max(1);
                // let uia = UserInputAction::MouseInput { event_type, col: cell_x, row: cell_y, button, modifiers };
                // if let Some(action) = self.term.interpret_input(EmulatorInput::User(uia)) {
                //     self.handle_emulator_action(action);
                // }
            }
        }
    }

    /// Handles actions signaled by the `TerminalInterface` implementation.
    /// These actions represent side effects or requests resulting from terminal emulation.
    fn handle_emulator_action(&mut self, action: EmulatorAction) {
        log::debug!("Orchestrator: Handling EmulatorAction: {:?}", action);
        match action {
            EmulatorAction::WritePty(data) => {
                if let Err(e) = self.pty_channel.write_all(&data) {
                    log::error!("Orchestrator: Failed to write_all {} bytes to PTY: {}", data.len(), e);
                } else {
                    log::trace!("Orchestrator: Wrote {} bytes to PTY.", data.len());
                }
            }
            EmulatorAction::SetTitle(title) => self.driver.set_title(&title),
            EmulatorAction::RingBell => self.driver.bell(),
            EmulatorAction::RequestRedraw => {
                // Redraw requests are implicit; the main loop calls render_if_needed.
                log::trace!("Orchestrator: EmulatorAction::RequestRedraw received (handled by main loop).");
            }
            EmulatorAction::SetCursorVisibility(visible) => {
                log::trace!("Orchestrator: Setting driver cursor visibility to: {}", visible);
                self.driver.set_cursor_visibility(visible);
            }
            EmulatorAction::CopyToClipboard(text) => {
                // Attempt to access the system clipboard and set its text.
                match arboard::Clipboard::new() {
                    Ok(mut clipboard) => {
                        if let Err(e) = clipboard.set_text(text) {
                            log::error!("Orchestrator: Failed to copy to clipboard via arboard: {}", e);
                        } else {
                            log::info!("Orchestrator: Copied text to clipboard via arboard.");
                        }
                    }
                    Err(e) => {
                        log::error!("Orchestrator: Failed to initialize arboard clipboard for copy: {}", e);
                    }
                }
            }
            EmulatorAction::RequestClipboardContent => {
                // Attempt to access the system clipboard and get its text.
                // If successful, inject the text back into the terminal as a UserInputAction.
                match arboard::Clipboard::new() {
                    Ok(mut clipboard) => {
                        match clipboard.get_text() {
                            Ok(text) => {
                                log::info!("Orchestrator: Retrieved text from clipboard via arboard for paste.");
                                let paste_input = EmulatorInput::User(UserInputAction::PasteText(text));
                                // This can result in a recursive call if the terminal produces another action.
                                // Typically, PasteText might lead to WritePty.
                                if let Some(next_action) = self.term.interpret_input(paste_input) {
                                    self.handle_emulator_action(next_action);
                                }
                            }
                            Err(e) => {
                                log::error!("Orchestrator: Failed to get text from clipboard via arboard: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Orchestrator: Failed to initialize arboard clipboard for paste: {}", e);
                    }
                }
            }
        }
    }

    /// Renders the terminal display if needed (e.g., if dirty lines exist or first draw).
    pub fn render_if_needed(&mut self) -> anyhow::Result<()> {
        // The renderer's draw method itself checks for dirty lines internally via the snapshot.
        // The orchestrator simply calls it.
        log::trace!("Orchestrator: Calling renderer.draw().");
        self.renderer.draw(&mut *self.term, &mut *self.driver)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
