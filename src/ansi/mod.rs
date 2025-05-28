// src/ansi/mod.rs

//! Handles ANSI escape sequence parsing.

// Make submodules public so their contents can be used elsewhere
pub mod commands;
mod lexer;
mod parser;
// Re-export necessary items for public API
pub use commands::AnsiCommand;

// Keep internal components private to this module unless needed outside
use lexer::AnsiLexer;
use parser::AnsiParser as ParserImpl;

// --- ANSI Output Key Sequence Definitions ---
// These are byte slice constants representing sequences typically sent by a terminal
// emulator to a host application when specific keys are pressed.
// These are used by the `AnsiKeySequence` enum below.
// References:
// - XTerm Control Sequences: https://invisible-island.net/xterm/ctlseqs/ctlseqs.html
// - ECMA-48: Control Functions for Coded Character Sets

// Note: These are kept private to this module as their primary interface
// for output generation will be through the `AnsiKeySequence` enum.

/// Cursor Up, Application Mode (DECCKM active). Sequence: `ESC O A`
const KEY_SEQ_BYTES_CURSOR_UP_APP: &[u8] = b"\x1bOA";
/// Cursor Up, Normal Mode (DECCKM inactive). Sequence: `ESC [ A`
const KEY_SEQ_BYTES_CURSOR_UP_NORMAL: &[u8] = b"\x1b[A";
/// Cursor Down, Application Mode. Sequence: `ESC O B`
const KEY_SEQ_BYTES_CURSOR_DOWN_APP: &[u8] = b"\x1bOB";
/// Cursor Down, Normal Mode. Sequence: `ESC [ B`
const KEY_SEQ_BYTES_CURSOR_DOWN_NORMAL: &[u8] = b"\x1b[B";
/// Cursor Right, Application Mode. Sequence: `ESC O C`
const KEY_SEQ_BYTES_CURSOR_RIGHT_APP: &[u8] = b"\x1bOC";
/// Cursor Right, Normal Mode. Sequence: `ESC [ C`
const KEY_SEQ_BYTES_CURSOR_RIGHT_NORMAL: &[u8] = b"\x1b[C";
/// Cursor Left, Application Mode. Sequence: `ESC O D`
const KEY_SEQ_BYTES_CURSOR_LEFT_APP: &[u8] = b"\x1bOD";
/// Cursor Left, Normal Mode. Sequence: `ESC [ D`
const KEY_SEQ_BYTES_CURSOR_LEFT_NORMAL: &[u8] = b"\x1b[D";

/// Home key. Common sequence: `ESC [ 1 ~`.
const KEY_SEQ_BYTES_HOME: &[u8] = b"\x1b[1~";
/// End key. Common sequence: `ESC [ 4 ~`.
const KEY_SEQ_BYTES_END: &[u8] = b"\x1b[4~";
/// Page Up key. Sequence: `ESC [ 5 ~`
const KEY_SEQ_BYTES_PAGE_UP: &[u8] = b"\x1b[5~";
/// Page Down key. Sequence: `ESC [ 6 ~`
const KEY_SEQ_BYTES_PAGE_DOWN: &[u8] = b"\x1b[6~";
/// Insert key. Sequence: `ESC [ 2 ~`
const KEY_SEQ_BYTES_INSERT: &[u8] = b"\x1b[2~";
/// Delete key. Sequence: `ESC [ 3 ~`
const KEY_SEQ_BYTES_DELETE: &[u8] = b"\x1b[3~";

/// F1 key. Common XTerm/VT sequence: `ESC O P`.
const KEY_SEQ_BYTES_F1: &[u8] = b"\x1bOP";
/// F2 key. Common XTerm/VT sequence: `ESC O Q`.
const KEY_SEQ_BYTES_F2: &[u8] = b"\x1bOQ";
/// F3 key. Common XTerm/VT sequence: `ESC O R`.
const KEY_SEQ_BYTES_F3: &[u8] = b"\x1bOR";
/// F4 key. Common XTerm/VT sequence: `ESC O S`.
const KEY_SEQ_BYTES_F4: &[u8] = b"\x1bOS";
/// F5 key. Common XTerm/VT sequence: `ESC [ 15 ~`.
const KEY_SEQ_BYTES_F5: &[u8] = b"\x1b[15~";
/// F6 key. Sequence: `ESC [ 17 ~`
const KEY_SEQ_BYTES_F6: &[u8] = b"\x1b[17~";
/// F7 key. Sequence: `ESC [ 18 ~`
const KEY_SEQ_BYTES_F7: &[u8] = b"\x1b[18~";
/// F8 key. Sequence: `ESC [ 19 ~`
const KEY_SEQ_BYTES_F8: &[u8] = b"\x1b[19~";
/// F9 key. Sequence: `ESC [ 20 ~`
const KEY_SEQ_BYTES_F9: &[u8] = b"\x1b[20~";
/// F10 key. Sequence: `ESC [ 21 ~`
const KEY_SEQ_BYTES_F10: &[u8] = b"\x1b[21~";
/// F11 key. Sequence: `ESC [ 23 ~`
const KEY_SEQ_BYTES_F11: &[u8] = b"\x1b[23~";
/// F12 key. Sequence: `ESC [ 24 ~`
const KEY_SEQ_BYTES_F12: &[u8] = b"\x1b[24~";

/// Shift+Tab (Backtab). Sequence: `ESC [ Z`
const KEY_SEQ_BYTES_BACKTAB: &[u8] = b"\x1b[Z";

/// Sequence to start bracketed paste mode. `ESC [ 200 ~`
const KEY_SEQ_BYTES_BRACKETED_PASTE_START: &[u8] = b"\x1b[200~";
/// Sequence to end bracketed paste mode. `ESC [ 201 ~`
const KEY_SEQ_BYTES_BRACKETED_PASTE_END: &[u8] = b"\x1b[201~";


/// Represents logical key actions and other UI-generated sequences that map to
/// specific ANSI output byte sequences.
/// This provides a type-safe way to refer to these sequences for output generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnsiKeySequence {
    /// Cursor Up in Application Mode (DECCKM).
    CursorUpApp,
    /// Cursor Up in Normal Mode.
    CursorUpNormal,
    /// Cursor Down in Application Mode.
    CursorDownApp,
    /// Cursor Down in Normal Mode.
    CursorDownNormal,
    /// Cursor Right in Application Mode.
    CursorRightApp,
    /// Cursor Right in Normal Mode.
    CursorRightNormal,
    /// Cursor Left in Application Mode.
    CursorLeftApp,
    /// Cursor Left in Normal Mode.
    CursorLeftNormal,
    /// Home key.
    Home,
    /// End key.
    End,
    /// Page Up key.
    PageUp,
    /// Page Down key.
    PageDown,
    /// Insert key.
    Insert,
    /// Delete key.
    Delete,
    /// Function key F1.
    F1,
    /// Function key F2.
    F2,
    /// Function key F3.
    F3,
    /// Function key F4.
    F4,
    /// Function key F5.
    F5,
    /// Function key F6.
    F6,
    /// Function key F7.
    F7,
    /// Function key F8.
    F8,
    /// Function key F9.
    F9,
    /// Function key F10.
    F10,
    /// Function key F11.
    F11,
    /// Function key F12.
    F12,
    /// Shift+Tab (Backtab).
    Backtab,
    /// Sequence to start bracketed paste mode.
    BracketedPasteStart,
    /// Sequence to end bracketed paste mode.
    BracketedPasteEnd,
}

impl AnsiKeySequence {
    /// Returns the byte slice representing the ANSI sequence for this logical key/action.
    pub fn as_bytes(self) -> &'static [u8] {
        match self {
            AnsiKeySequence::CursorUpApp => KEY_SEQ_BYTES_CURSOR_UP_APP,
            AnsiKeySequence::CursorUpNormal => KEY_SEQ_BYTES_CURSOR_UP_NORMAL,
            AnsiKeySequence::CursorDownApp => KEY_SEQ_BYTES_CURSOR_DOWN_APP,
            AnsiKeySequence::CursorDownNormal => KEY_SEQ_BYTES_CURSOR_DOWN_NORMAL,
            AnsiKeySequence::CursorRightApp => KEY_SEQ_BYTES_CURSOR_RIGHT_APP,
            AnsiKeySequence::CursorRightNormal => KEY_SEQ_BYTES_CURSOR_RIGHT_NORMAL,
            AnsiKeySequence::CursorLeftApp => KEY_SEQ_BYTES_CURSOR_LEFT_APP,
            AnsiKeySequence::CursorLeftNormal => KEY_SEQ_BYTES_CURSOR_LEFT_NORMAL,
            AnsiKeySequence::Home => KEY_SEQ_BYTES_HOME,
            AnsiKeySequence::End => KEY_SEQ_BYTES_END,
            AnsiKeySequence::PageUp => KEY_SEQ_BYTES_PAGE_UP,
            AnsiKeySequence::PageDown => KEY_SEQ_BYTES_PAGE_DOWN,
            AnsiKeySequence::Insert => KEY_SEQ_BYTES_INSERT,
            AnsiKeySequence::Delete => KEY_SEQ_BYTES_DELETE,
            AnsiKeySequence::F1 => KEY_SEQ_BYTES_F1,
            AnsiKeySequence::F2 => KEY_SEQ_BYTES_F2,
            AnsiKeySequence::F3 => KEY_SEQ_BYTES_F3,
            AnsiKeySequence::F4 => KEY_SEQ_BYTES_F4,
            AnsiKeySequence::F5 => KEY_SEQ_BYTES_F5,
            AnsiKeySequence::F6 => KEY_SEQ_BYTES_F6,
            AnsiKeySequence::F7 => KEY_SEQ_BYTES_F7,
            AnsiKeySequence::F8 => KEY_SEQ_BYTES_F8,
            AnsiKeySequence::F9 => KEY_SEQ_BYTES_F9,
            AnsiKeySequence::F10 => KEY_SEQ_BYTES_F10,
            AnsiKeySequence::F11 => KEY_SEQ_BYTES_F11,
            AnsiKeySequence::F12 => KEY_SEQ_BYTES_F12,
            AnsiKeySequence::Backtab => KEY_SEQ_BYTES_BACKTAB,
            AnsiKeySequence::BracketedPasteStart => KEY_SEQ_BYTES_BRACKETED_PASTE_START,
            AnsiKeySequence::BracketedPasteEnd => KEY_SEQ_BYTES_BRACKETED_PASTE_END,
        }
    }
}

// --- Common ANSI Sequence Introducers/Terminators ---
// These are fundamental parts of many ANSI sequences.
// Reference: ECMA-48 standard.

/// CSI (Control Sequence Introducer) `ESC [`
pub const CSI: &[u8] = b"\x1b[";
/// OSC (Operating System Command) `ESC ]`
pub const OSC: &[u8] = b"\x1b]";
/// ST (String Terminator) for OSC and other sequences, can be `ESC \` (ST) or BEL.
pub const ST_ESC_BACKSLASH: &[u8] = b"\x1b\\";

pub trait AnsiParser {
    fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand>;
}

/// The main processor that combines the lexer and parser.
/// It takes byte slices as input and provides parsed commands.
#[derive(Debug, Default)]
pub struct AnsiProcessor {
    pub(super) lexer: AnsiLexer,
    pub(super) parser: ParserImpl,
}

impl AnsiProcessor {
    /// Creates a new `AnsiProcessor`.
    pub fn new() -> Self {
        AnsiProcessor {
            lexer: AnsiLexer::new(),
            parser: ParserImpl::new(),
        }
    }
}

impl AnsiParser for AnsiProcessor {
    /// Processes a slice of bytes.
    ///
    /// Bytes are lexed into tokens, and tokens are processed by the parser.
    /// Call `take_commands` on the `parser` field to retrieve results.
    fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand> {
        for byte in bytes {
            self.lexer.process_byte(*byte);
        }
        // Finalize any pending UTF-8 sequence in the lexer.
        self.lexer.finalize();

        // Now take all tokens, including any finalization token.
        let tokens = self.lexer.take_tokens();
        for token in tokens {
            self.parser.process_token(token);
        }
        self.parser.take_commands()
    }
}

// Include tests module if defined in this file
#[cfg(test)]
mod tests; // Assuming tests are in ansi/tests.rs
