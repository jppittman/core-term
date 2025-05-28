// src/term/emulator/input_handler.rs

//! Handles user input actions and control events for the terminal emulator.
//!
//! This module translates `UserInputAction` (like key presses, mouse events) and
//! `ControlEvent` (like resize) into `EmulatorAction`s that the terminal
//! emulator can then process or delegate (e.g., writing to PTY, copying to clipboard).
//! The functions here aim to follow the project's style guide, emphasizing clarity,
//! minimal nesting, and appropriate use of helper functions, including avoidance
//! of boolean arguments in favor of descriptive enums.

use super::{FocusState, TerminalEmulator}; // TerminalEmulator needed for dec_modes and selection
use crate::backends::{MouseButton, MouseEventType};
use crate::keys::{KeySymbol, Modifiers};
use crate::term::{
    ControlEvent,
    Selection,
    action::{EmulatorAction, KeyInput, MouseInput, UserInputAction}, // Use the structured UserInputAction
    snapshot::SelectionMode, // For mouse input selection mode
};
// Import ANSI constants and enums for use in this module
use crate::ansi::{
    CSI,
    AnsiKeySequence,
    commands::C0Control, // Use C0Control from the commands submodule
};

use log::{debug, trace}; // `trace` is used for verbose logging, `debug` for significant events

// --- Mouse Reporting Constants (specific to this module's logic if not general ANSI) ---
// Sourced from XTerm Control Sequences documentation:
// https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Mouse-Tracking

/// Modifier flag for Shift key in mouse reports. Value: 4.
const MOUSE_MODIFIER_SHIFT: u8 = 4;
/// Modifier flag for Alt/Meta key in mouse reports. Value: 8.
const MOUSE_MODIFIER_ALT: u8 = 8;
/// Modifier flag for Control key in mouse reports. Value: 16.
const MOUSE_MODIFIER_CONTROL: u8 = 16;

/// Base button code for Left Mouse Button (press/release). Value: 0.
const MOUSE_BUTTON_LEFT: u8 = 0;
/// Base button code for Middle Mouse Button (press/release). Value: 1.
const MOUSE_BUTTON_MIDDLE: u8 = 1;
/// Base button code for Right Mouse Button (press/release). Value: 2.
const MOUSE_BUTTON_RIGHT: u8 = 2;
/// Button code used in SGR mode for release of an unknown button or for hover events. Value: 3.
const MOUSE_BUTTON_SGR_NONE: u8 = 3;

/// Button code for Scroll Wheel Up. Value: 64.
const MOUSE_BUTTON_SCROLL_UP: u8 = 64;
/// Button code for Scroll Wheel Down. Value: 65.
const MOUSE_BUTTON_SCROLL_DOWN: u8 = 65;
/// Button code for Scroll Wheel Left (less common). Value: 66.
const MOUSE_BUTTON_SCROLL_LEFT: u8 = 66;
/// Button code for Scroll Wheel Right (less common). Value: 67.
const MOUSE_BUTTON_SCROLL_RIGHT: u8 = 67;

// Constants for legacy mouse reporting modes (non-SGR)
/// Offset added to button code for motion events in legacy modes. Value: 32.
const MOUSE_MOTION_CODE_OFFSET_LEGACY: u8 = 32;
/// Base button code for release events (L/M/R) in legacy modes, before modifiers. Value: 3.
const MOUSE_BUTTON_RELEASE_BASE_CODE_LEGACY: u8 = 3;
/// Offset added to coordinates in legacy mouse reports. Value: 32.
const MOUSE_COORDINATE_OFFSET_LEGACY: u8 = 32;
/// Max raw coordinate value in legacy modes (before subtracting offset). Value: 255.
const MOUSE_LEGACY_MAX_COORD_RAW: u8 = 255;
/// Max effective coordinate value in legacy modes (after subtracting offset). Value: 223.
const MOUSE_LEGACY_MAX_COORD_EFFECTIVE: usize =
    (MOUSE_LEGACY_MAX_COORD_RAW - MOUSE_COORDINATE_OFFSET_LEGACY) as usize;

// Constants for SGR mouse reporting mode (1006)
/// A practical high coordinate limit for SGR mouse reports. Value: 9999.
const MOUSE_SGR_MAX_COORD: usize = 9999;

// --- Enums to replace boolean arguments, as per style guide ---

/// Defines the mode for cursor key sequence generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CursorKeyMode {
    /// Standard ANSI sequences for cursor keys.
    Normal,
    /// Application-specific sequences for cursor keys (DECCKM).
    Application,
}

/// Defines the context of a mouse move event for reporting purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MouseMoveContext {
    /// A simple mouse hover or move without an active selection drag.
    Hover,
    /// Mouse movement while a selection is actively being dragged.
    Drag,
}

/// Defines the mode for pasting text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PasteMode {
    /// Paste text directly.
    Direct,
    /// Wrap pasted text with bracketed paste mode sequences.
    Bracketed,
}

/// Top-level dispatcher for processing a `UserInputAction`.
///
/// This function routes the `action` to more specific handlers based on its type.
/// It also handles global state updates like resetting `cursor_wrap_next`.
///
/// # Arguments
/// * `emulator`: A mutable reference to the `TerminalEmulator` state.
/// * `action`: The `UserInputAction` to process, now using the structured `KeyInput` and `MouseInput`.
///
/// # Returns
/// * `Option<EmulatorAction>`: An action for the emulator to perform, or `None`
///   if the input did not result in a direct emulator action.
pub(super) fn process_user_input_action(
    emulator: &mut TerminalEmulator,
    action: UserInputAction, // This now uses your new enum structure
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false; // Reset flag on any user input

    match action {
        UserInputAction::FocusLost => {
            emulator.focus_state = FocusState::Unfocused;
            None
        }
        UserInputAction::FocusGained => {
            emulator.focus_state = FocusState::Focused;
            None
        }
        UserInputAction::KeyInput(key_input_data) => {
            // Destructure KeyInput
            let key_mode = if emulator.dec_modes.cursor_keys_app_mode {
                CursorKeyMode::Application
            } else {
                CursorKeyMode::Normal
            };
            handle_key_input(key_input_data, key_mode)
        }
        UserInputAction::MouseInput(mouse_input_data) => {
            // Destructure MouseInput
            handle_mouse_input(
                &mut emulator.selection,
                &emulator.dec_modes,
                mouse_input_data, // Pass the whole struct
            )
        }
        UserInputAction::InitiateCopy => handle_initiate_copy(emulator),
        UserInputAction::InitiatePaste => Some(EmulatorAction::RequestClipboardContent),
        UserInputAction::PasteText(text_to_paste) => {
            let paste_mode = if emulator.dec_modes.bracketed_paste_mode {
                PasteMode::Bracketed
            } else {
                PasteMode::Direct
            };
            handle_paste_text(text_to_paste, paste_mode)
        }
    }
}

/// Handles key press inputs and translates them into PTY sequences or other actions.
///
/// # Arguments
/// * `key_input`: The structured `KeyInput` data.
/// * `key_mode`: The `CursorKeyMode` (Normal or Application) for cursor keys.
fn handle_key_input(
    key_input: KeyInput, // Takes the KeyInput struct
    key_mode: CursorKeyMode,
) -> Option<EmulatorAction> {
    let bytes_to_send = generate_pty_sequence_for_key(
        key_input.symbol,
        key_input.modifiers,
        key_input.text, // text is Option<String> within KeyInput
        key_mode,
    );

    if !bytes_to_send.is_empty() {
        Some(EmulatorAction::WritePty(bytes_to_send))
    } else {
        None
    }
}

/// Generates the byte sequence to be sent to the PTY for a given key input.
///
/// # Arguments
/// * `symbol`: The `KeySymbol` from `KeyInput`.
/// * `modifiers`: The `Modifiers` from `KeyInput`.
/// * `text`: The `Option<String>` text from `KeyInput`.
/// * `key_mode`: The `CursorKeyMode` for cursor key sequence generation.
fn generate_pty_sequence_for_key(
    symbol: KeySymbol,
    modifiers: Modifiers,
    text: Option<String>, // This is already Option<String>
    key_mode: CursorKeyMode,
) -> Vec<u8> {
    let mut sequence = Vec::new();

    if modifiers.contains(Modifiers::CONTROL) {
        match symbol {
            KeySymbol::Char(c) if c.is_ascii_alphabetic() => {
                sequence.push((c.to_ascii_lowercase() as u8) - b'a' + (C0Control::SOH as u8));
                return sequence;
            }
            KeySymbol::Char('[') => {
                sequence.push(C0Control::ESC as u8);
                return sequence;
            }
            KeySymbol::Char('\\') => {
                sequence.push(C0Control::FS as u8);
                return sequence;
            }
            KeySymbol::Char(']') => {
                sequence.push(C0Control::GS as u8);
                return sequence;
            }
            KeySymbol::Char('^') => {
                sequence.push(C0Control::RS as u8);
                return sequence;
            }
            KeySymbol::Char('_') => {
                sequence.push(C0Control::US as u8);
                return sequence;
            }
            KeySymbol::Char(' ') => {
                sequence.push(C0Control::NUL as u8);
                return sequence;
            }
            KeySymbol::Char('?') => {
                sequence.push(C0Control::DEL as u8);
                return sequence;
            }
            _ => {}
        }
    }

    // `.as_deref()` converts Option<String> to Option<&str> for starts_with
    if let Some(txt_val_str) = text.as_deref() {
        if modifiers.contains(Modifiers::ALT)
            && !txt_val_str.starts_with(C0Control::ESC as u8 as char)
        {
            sequence.push(C0Control::ESC as u8);
        }
        sequence.extend(txt_val_str.as_bytes());
        return sequence;
    }

    if modifiers.contains(Modifiers::ALT) {
        sequence.push(C0Control::ESC as u8);
    }

    match symbol {
        KeySymbol::Enter | KeySymbol::KeypadEnter => sequence.push(C0Control::CR as u8),
        KeySymbol::Backspace => sequence.push(C0Control::BS as u8),
        KeySymbol::Tab => {
            if modifiers.contains(Modifiers::SHIFT) {
                sequence.extend_from_slice(AnsiKeySequence::Backtab.as_bytes());
            } else {
                sequence.push(C0Control::HT as u8);
            }
        }
        KeySymbol::Escape => sequence.push(C0Control::ESC as u8),

        KeySymbol::Up => sequence.extend_from_slice(
            match key_mode {
                CursorKeyMode::Application => AnsiKeySequence::CursorUpApp,
                CursorKeyMode::Normal => AnsiKeySequence::CursorUpNormal,
            }
            .as_bytes(),
        ),
        KeySymbol::Down => sequence.extend_from_slice(
            match key_mode {
                CursorKeyMode::Application => AnsiKeySequence::CursorDownApp,
                CursorKeyMode::Normal => AnsiKeySequence::CursorDownNormal,
            }
            .as_bytes(),
        ),
        KeySymbol::Right => sequence.extend_from_slice(
            match key_mode {
                CursorKeyMode::Application => AnsiKeySequence::CursorRightApp,
                CursorKeyMode::Normal => AnsiKeySequence::CursorRightNormal,
            }
            .as_bytes(),
        ),
        KeySymbol::Left => sequence.extend_from_slice(
            match key_mode {
                CursorKeyMode::Application => AnsiKeySequence::CursorLeftApp,
                CursorKeyMode::Normal => AnsiKeySequence::CursorLeftNormal,
            }
            .as_bytes(),
        ),

        KeySymbol::Home => sequence.extend_from_slice(AnsiKeySequence::Home.as_bytes()),
        KeySymbol::End => sequence.extend_from_slice(AnsiKeySequence::End.as_bytes()),
        KeySymbol::PageUp => sequence.extend_from_slice(AnsiKeySequence::PageUp.as_bytes()),
        KeySymbol::PageDown => sequence.extend_from_slice(AnsiKeySequence::PageDown.as_bytes()),
        KeySymbol::Insert => sequence.extend_from_slice(AnsiKeySequence::Insert.as_bytes()),
        KeySymbol::Delete => sequence.extend_from_slice(AnsiKeySequence::Delete.as_bytes()),

        KeySymbol::F1 => sequence.extend_from_slice(AnsiKeySequence::F1.as_bytes()),
        KeySymbol::F2 => sequence.extend_from_slice(AnsiKeySequence::F2.as_bytes()),
        KeySymbol::F3 => sequence.extend_from_slice(AnsiKeySequence::F3.as_bytes()),
        KeySymbol::F4 => sequence.extend_from_slice(AnsiKeySequence::F4.as_bytes()),
        KeySymbol::F5 => sequence.extend_from_slice(AnsiKeySequence::F5.as_bytes()),
        KeySymbol::F6 => sequence.extend_from_slice(AnsiKeySequence::F6.as_bytes()),
        KeySymbol::F7 => sequence.extend_from_slice(AnsiKeySequence::F7.as_bytes()),
        KeySymbol::F8 => sequence.extend_from_slice(AnsiKeySequence::F8.as_bytes()),
        KeySymbol::F9 => sequence.extend_from_slice(AnsiKeySequence::F9.as_bytes()),
        KeySymbol::F10 => sequence.extend_from_slice(AnsiKeySequence::F10.as_bytes()),
        KeySymbol::F11 => sequence.extend_from_slice(AnsiKeySequence::F11.as_bytes()),
        KeySymbol::F12 => sequence.extend_from_slice(AnsiKeySequence::F12.as_bytes()),

        KeySymbol::KeypadPlus => sequence.push(b'+'),
        KeySymbol::KeypadMinus => sequence.push(b'-'),
        KeySymbol::KeypadMultiply => sequence.push(b'*'),
        KeySymbol::KeypadDivide => sequence.push(b'/'),
        KeySymbol::KeypadDecimal => sequence.push(b'.'),
        KeySymbol::Keypad0 => sequence.push(b'0'),
        KeySymbol::Keypad1 => sequence.push(b'1'),
        KeySymbol::Keypad2 => sequence.push(b'2'),
        KeySymbol::Keypad3 => sequence.push(b'3'),
        KeySymbol::Keypad4 => sequence.push(b'4'),
        KeySymbol::Keypad5 => sequence.push(b'5'),
        KeySymbol::Keypad6 => sequence.push(b'6'),
        KeySymbol::Keypad7 => sequence.push(b'7'),
        KeySymbol::Keypad8 => sequence.push(b'8'),
        KeySymbol::Keypad9 => sequence.push(b'9'),

        KeySymbol::Char(c) => {
            let mut buf = [0; 4];
            sequence.extend(c.encode_utf8(&mut buf).as_bytes());
        }
        _ => {
            if sequence.as_slice() == [C0Control::ESC as u8] {
                sequence.clear();
            }
            trace!(
                "Unhandled KeySymbol (not text, not Alt-prefixed char): {:?}, Modifiers: {:?}",
                symbol, modifiers
            );
        }
    }
    sequence
}

/// Handles mouse input events, updating selection state and generating PTY sequences if mouse reporting is active.
///
/// # Arguments
/// * `selection_state`: Mutable reference to the terminal's `Selection` state.
/// * `dec_modes`: Reference to the terminal's `DecPrivateModes` for mouse reporting.
/// * `mouse_input`: The structured `MouseInput` data, containing all mouse event details.
fn handle_mouse_input(
    selection_state: &mut Selection,
    dec_modes: &crate::term::modes::DecPrivateModes,
    mouse_input: MouseInput, // Takes the MouseInput struct
) -> Option<EmulatorAction> {
    let move_context = match mouse_input.event_type {
        MouseEventType::Move => {
            if selection_state.is_active {
                // is_active implies a button (likely left) is down for selection
                MouseMoveContext::Drag
            } else {
                MouseMoveContext::Hover
            }
        }
        _ => MouseMoveContext::Hover, // Default context if not a move event for reporting logic
    };

    match mouse_input.event_type {
        MouseEventType::Press => {
            if mouse_input.button == MouseButton::Left {
                let mode = if mouse_input.modifiers.contains(Modifiers::ALT) {
                    SelectionMode::Block
                } else {
                    SelectionMode::Normal
                };
                selection_state.start_selection(mouse_input.col, mouse_input.row, mode);
                debug!(
                    "MousePress: Started selection at ({}, {}) mode {:?}",
                    mouse_input.col, mouse_input.row, mode
                );
            }
        }
        MouseEventType::Move => {
            if selection_state.is_active {
                // This is the primary check for updating selection
                selection_state.update_selection(mouse_input.col, mouse_input.row);
            }
        }
        MouseEventType::Release => {
            if mouse_input.button == MouseButton::Left && selection_state.is_active {
                selection_state.end_selection();
                debug!(
                    "MouseRelease: Ended selection. Start: {:?}, End: {:?}",
                    selection_state.start, selection_state.end
                );
            }
        }
    }

    // Pass a reference to mouse_input to generate_mouse_report_sequence
    let report_bytes = generate_mouse_report_sequence(
        dec_modes,
        &mouse_input, // Pass the MouseInput struct by reference
        move_context,
    );

    report_bytes.map(EmulatorAction::WritePty)
}

/// Generates the byte sequence for a mouse report based on active DEC modes.
///
/// # Arguments
/// * `dec_modes`: Active DEC private modes, determining which protocol to use.
/// * `mouse_input`: A reference to the `MouseInput` struct containing event details.
/// * `move_context`: The derived `MouseMoveContext` (Hover or Drag).
fn generate_mouse_report_sequence(
    dec_modes: &crate::term::modes::DecPrivateModes,
    mouse_input: &MouseInput, // Takes a reference to MouseInput struct
    move_context: MouseMoveContext,
) -> Option<Vec<u8>> {
    if !(dec_modes.mouse_x10_mode
        || dec_modes.mouse_vt200_mode
        || dec_modes.mouse_vt200_highlight_mode
        || dec_modes.mouse_button_event_mode
        || dec_modes.mouse_any_event_mode
        || dec_modes.mouse_sgr_mode)
    {
        return None;
    }

    let mut final_button_code = 0u8;
    let mut is_sgr_release_char = false;

    // Use fields from mouse_input struct
    match mouse_input.button {
        MouseButton::Left => final_button_code = MOUSE_BUTTON_LEFT,
        MouseButton::Middle => final_button_code = MOUSE_BUTTON_MIDDLE,
        MouseButton::Right => final_button_code = MOUSE_BUTTON_RIGHT,
        MouseButton::ScrollUp => final_button_code = MOUSE_BUTTON_SCROLL_UP,
        MouseButton::ScrollDown => final_button_code = MOUSE_BUTTON_SCROLL_DOWN,
        MouseButton::ScrollLeft => final_button_code = MOUSE_BUTTON_SCROLL_LEFT,
        MouseButton::ScrollRight => final_button_code = MOUSE_BUTTON_SCROLL_RIGHT,
        MouseButton::Unknown => {
            if mouse_input.event_type == MouseEventType::Move
                && move_context == MouseMoveContext::Hover
            {
            } else if mouse_input.event_type != MouseEventType::Release {
                trace!(
                    "Unknown mouse button for non-release event: {:?}, type: {:?}",
                    mouse_input.button, mouse_input.event_type
                );
            }
            if mouse_input.event_type == MouseEventType::Release && dec_modes.mouse_sgr_mode {
                final_button_code = MOUSE_BUTTON_SGR_NONE;
            }
        }
    }

    if mouse_input.modifiers.contains(Modifiers::SHIFT) {
        final_button_code += MOUSE_MODIFIER_SHIFT;
    }
    if mouse_input.modifiers.contains(Modifiers::ALT) {
        final_button_code += MOUSE_MODIFIER_ALT;
    }
    if mouse_input.modifiers.contains(Modifiers::CONTROL) {
        final_button_code += MOUSE_MODIFIER_CONTROL;
    }

    let is_drag_event = move_context == MouseMoveContext::Drag;

    if dec_modes.mouse_sgr_mode {
        if mouse_input.event_type == MouseEventType::Release {
            is_sgr_release_char = true;
        } else if mouse_input.event_type == MouseEventType::Move
            && !is_drag_event
            && mouse_input.button == MouseButton::Unknown
        {
            if !dec_modes.mouse_any_event_mode {
                return None;
            }
            final_button_code = MOUSE_BUTTON_SGR_NONE;
        }
    } else {
        if mouse_input.event_type == MouseEventType::Move && is_drag_event {
            final_button_code += MOUSE_MOTION_CODE_OFFSET_LEGACY;
        }
        if mouse_input.event_type == MouseEventType::Release
            && mouse_input.button != MouseButton::ScrollUp
            && mouse_input.button != MouseButton::ScrollDown
            && mouse_input.button != MouseButton::ScrollLeft
            && mouse_input.button != MouseButton::ScrollRight
        {
            final_button_code = MOUSE_BUTTON_RELEASE_BASE_CODE_LEGACY;
            if mouse_input.modifiers.contains(Modifiers::SHIFT) {
                final_button_code += MOUSE_MODIFIER_SHIFT;
            }
            if mouse_input.modifiers.contains(Modifiers::ALT) {
                final_button_code += MOUSE_MODIFIER_ALT;
            }
            if mouse_input.modifiers.contains(Modifiers::CONTROL) {
                final_button_code += MOUSE_MODIFIER_CONTROL;
            }
        }
    }

    let should_report = match mouse_input.event_type {
        MouseEventType::Press => true,
        MouseEventType::Release => {
            dec_modes.mouse_vt200_mode
                || dec_modes.mouse_vt200_highlight_mode
                || dec_modes.mouse_button_event_mode
                || dec_modes.mouse_any_event_mode
                || dec_modes.mouse_sgr_mode
        }
        MouseEventType::Move => {
            if dec_modes.mouse_any_event_mode {
                true
            } else if is_drag_event
                && (dec_modes.mouse_vt200_mode
                    || dec_modes.mouse_vt200_highlight_mode
                    || dec_modes.mouse_button_event_mode
                    || dec_modes.mouse_sgr_mode)
            {
                true
            } else {
                false
            }
        }
    };

    if !should_report {
        return None;
    }

    let report_col = mouse_input.col + 1;
    let report_row = mouse_input.row + 1;

    let max_coord_val = if dec_modes.mouse_sgr_mode {
        MOUSE_SGR_MAX_COORD
    } else {
        MOUSE_LEGACY_MAX_COORD_EFFECTIVE
    };
    let clamped_col = report_col.min(max_coord_val);
    let clamped_row = report_row.min(max_coord_val);

    let mut seq = Vec::new();
    seq.extend_from_slice(CSI);

    if dec_modes.mouse_sgr_mode {
        seq.extend(format!("<{};{};{}", final_button_code, clamped_col, clamped_row).into_bytes());
        seq.push(if is_sgr_release_char { b'm' } else { b'M' });
    } else {
        seq.push(b'M');
        seq.push((final_button_code + MOUSE_COORDINATE_OFFSET_LEGACY) as u8);
        seq.push((clamped_col + MOUSE_COORDINATE_OFFSET_LEGACY as usize) as u8);
        seq.push((clamped_row + MOUSE_COORDINATE_OFFSET_LEGACY as usize) as u8);
    }

    debug!(
        "Sending mouse report sequence: {:?}",
        String::from_utf8_lossy(&seq)
    );
    Some(seq)
}

/// Handles the `InitiateCopy` user action.
fn handle_initiate_copy(emulator: &TerminalEmulator) -> Option<EmulatorAction> {
    let (term_cols, term_rows) = emulator.dimensions();
    let selected_text =
        emulator
            .selection
            .get_selected_text(emulator.screen.active_grid(), term_cols, term_rows);

    if selected_text.is_empty() {
        None
    } else {
        debug!("InitiateCopy: Selected text: '{}'", selected_text);
        Some(EmulatorAction::CopyToClipboard(selected_text))
    }
}

/// Handles the `PasteText` user action.
fn handle_paste_text(text_to_paste: String, paste_mode: PasteMode) -> Option<EmulatorAction> {
    let bytes_to_paste = match paste_mode {
        PasteMode::Bracketed => {
            let mut seq = Vec::new();
            seq.extend_from_slice(AnsiKeySequence::BracketedPasteStart.as_bytes());
            seq.extend_from_slice(text_to_paste.as_bytes());
            seq.extend_from_slice(AnsiKeySequence::BracketedPasteEnd.as_bytes());
            seq
        }
        PasteMode::Direct => text_to_paste.into_bytes(),
    };
    Some(EmulatorAction::WritePty(bytes_to_paste))
}

/// Processes a `ControlEvent` from the system or orchestrator.
pub(super) fn process_control_event(
    emulator: &mut TerminalEmulator,
    event: ControlEvent,
) -> Option<EmulatorAction> {
    emulator.cursor_wrap_next = false;

    match event {
        ControlEvent::FrameRendered => {
            trace!("TerminalEmulator: ControlEvent::FrameRendered received.");
            None
        }
        ControlEvent::Resize { cols, rows } => {
            trace!(
                "TerminalEmulator: ControlEvent::Resize to {}x{} received.",
                cols, rows
            );
            emulator.resize(cols, rows);
            None
        }
    }
}
