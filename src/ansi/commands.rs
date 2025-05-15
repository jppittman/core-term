// src/ansi/commands.rs

//! Defines the `AnsiCommand` enum representing parsed ANSI escape sequences
//! and related helper enums/structs.

use std::fmt;
use std::iter::Peekable;
use std::slice::Iter;
use log::warn;

// --- Color Definitions ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color { // Made pub
    Black, Red, Green, Yellow, Blue, Magenta, Cyan, White,
    BrightBlack, BrightRed, BrightGreen, BrightYellow, BrightBlue, BrightMagenta, BrightCyan, BrightWhite,
    Indexed(u8),
    Rgb(u8, u8, u8),
    Default,
}

// --- SGR Attributes ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attribute { // Made pub
    Reset, Bold, Faint, Italic, Underline, BlinkSlow, BlinkRapid, Reverse, Conceal, Strikethrough,
    UnderlineDouble, NoBold, NoItalic, NoUnderline, NoBlink, NoReverse, NoConceal, NoStrikethrough,
    Foreground(Color), Background(Color), Overlined, UnderlineColor(Color),
}

// --- C0 Control Enum ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum C0Control { // Made pub
    NUL = 0x00, SOH = 0x01, STX = 0x02, ETX = 0x03, EOT = 0x04, ENQ = 0x05, ACK = 0x06, BEL = 0x07,
    BS  = 0x08, HT  = 0x09, LF  = 0x0A, VT  = 0x0B, FF  = 0x0C, CR  = 0x0D, SO  = 0x0E, SI  = 0x0F,
    DLE = 0x10, DC1 = 0x11, DC2 = 0x12, DC3 = 0x13, DC4 = 0x14, NAK = 0x15, SYN = 0x16, ETB = 0x17,
    CAN = 0x18, EM  = 0x19, SUB = 0x1A, ESC = 0x1B, FS  = 0x1C, GS  = 0x1D, RS  = 0x1E, US  = 0x1F,
    DEL = 0x7F,
}

impl C0Control {
    pub fn from_byte(byte: u8) -> Option<Self> {
        if byte <= 0x1F || byte == 0x7F {
            Some(unsafe { std::mem::transmute(byte) })
        } else {
            None
        }
    }
}

// --- CSI Command Enum ---
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsiCommand { // Made pub
    CursorUp(u16), CursorDown(u16), CursorForward(u16), CursorBackward(u16),
    CursorNextLine(u16), CursorPrevLine(u16), CursorCharacterAbsolute(u16),
    CursorPosition(u16, u16),
    EraseInDisplay(u16), EraseInLine(u16), EraseCharacter(u16),
    InsertCharacter(u16), InsertLine(u16), DeleteCharacter(u16), DeleteLine(u16),
    ScrollUp(u16), ScrollDown(u16),
    SetTabStop, // Added missing command variant if needed by parser
    ClearTabStops(u16),
    SetGraphicsRendition(Vec<Attribute>),
    SetMode(u16), ResetMode(u16), SetModePrivate(u16), ResetModePrivate(u16),
    DeviceStatusReport(u16),
    SaveCursorAnsi, RestoreCursorAnsi, SaveCursor, RestoreCursor, // Kept both SCO and DEC variants if parser distinguishes
    Reset, // Added missing command variant if needed by parser
    SetScrollingRegion { top: u16, bottom: u16 }, // Added for DECSTBM (CSI r)
    Unsupported(Vec<u8>, Option<u8>), // Kept for debugging/completeness
}

// --- ESC Command Enum ---
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscCommand { // Made pub
    SetTabStop, Index, NextLine, ReverseIndex, SaveCursor, RestoreCursor,
    ResetToInitialState,
    SelectCharacterSet(char, char),
    SingleShift2, SingleShift3,
}

// --- Main AnsiCommand Enum ---
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnsiCommand { // Made pub
    Print(char),
    C0Control(C0Control),
    C1Control(u8),
    Csi(CsiCommand),
    Esc(EscCommand),
    Osc(Vec<u8>), Dcs(Vec<u8>), Pm(Vec<u8>), Apc(Vec<u8>),
    StringTerminator,
    Ignore(u8), Error(u8),
}

// --- Implementation details (kept private unless needed elsewhere) ---

impl AnsiCommand {
    // These helpers can remain private if only used within this module's logic
    pub(crate) fn from_c0(byte: u8) -> Option<Self> {
        C0Control::from_byte(byte).map(AnsiCommand::C0Control)
    }

    pub(crate) fn from_c1(byte: u8) -> Option<Self> {
         match byte {
             0x84 => Some(AnsiCommand::Esc(EscCommand::Index)),
             0x85 => Some(AnsiCommand::Esc(EscCommand::NextLine)),
             0x88 => Some(AnsiCommand::Esc(EscCommand::SetTabStop)),
             0x8D => Some(AnsiCommand::Esc(EscCommand::ReverseIndex)),
             0x8E => Some(AnsiCommand::Esc(EscCommand::SingleShift2)),
             0x8F => Some(AnsiCommand::Esc(EscCommand::SingleShift3)),
             0x90 | 0x9B | 0x9C | 0x9D | 0x9E | 0x9F => None, // Handled by parser state transitions
             _ => Some(AnsiCommand::C1Control(byte)), // Treat others as generic C1
         }
    }

    pub(crate) fn from_esc(final_char: char) -> Option<Self> {
        match final_char {
            'D' => Some(AnsiCommand::Esc(EscCommand::Index)),
            'E' => Some(AnsiCommand::Esc(EscCommand::NextLine)),
            'H' => Some(AnsiCommand::Esc(EscCommand::SetTabStop)),
            'M' => Some(AnsiCommand::Esc(EscCommand::ReverseIndex)),
            '7' => Some(AnsiCommand::Esc(EscCommand::SaveCursor)),
            '8' => Some(AnsiCommand::Esc(EscCommand::RestoreCursor)),
            'c' => Some(AnsiCommand::Esc(EscCommand::ResetToInitialState)),
            'N' => Some(AnsiCommand::Esc(EscCommand::SingleShift2)),
            'O' => Some(AnsiCommand::Esc(EscCommand::SingleShift3)),
            _ => None,
        }
    }

     pub(crate) fn from_esc_intermediate(intermediate: char, final_char: char) -> Option<Self> {
         if ['(', ')', '*', '+'].contains(&intermediate) {
             if final_char.is_ascii_uppercase() || final_char.is_ascii_digit() {
                 Some(AnsiCommand::Esc(EscCommand::SelectCharacterSet(intermediate, final_char)))
             } else { None }
         } else { None }
     }

    fn parse_sgr(params: Vec<u16>) -> Vec<Attribute> {
        let mut attrs = Vec::new();
        if params.is_empty() {
            // Treat CSI m as CSI 0 m (Reset)
            attrs.push(Attribute::Reset);
            return attrs;
        }
        let mut iter = params.iter().peekable();
        while let Some(&param) = iter.next() {
            match param {
                0 => attrs.push(Attribute::Reset), 1 => attrs.push(Attribute::Bold),
                2 => attrs.push(Attribute::Faint), 3 => attrs.push(Attribute::Italic),
                4 => attrs.push(Attribute::Underline), 5 => attrs.push(Attribute::BlinkSlow),
                6 => attrs.push(Attribute::BlinkRapid), 7 => attrs.push(Attribute::Reverse),
                8 => attrs.push(Attribute::Conceal), 9 => attrs.push(Attribute::Strikethrough),
                21 => attrs.push(Attribute::UnderlineDouble),
                22 => attrs.push(Attribute::NoBold), // Also turns off Faint
                23 => attrs.push(Attribute::NoItalic), 24 => attrs.push(Attribute::NoUnderline),
                25 => attrs.push(Attribute::NoBlink), 27 => attrs.push(Attribute::NoReverse),
                28 => attrs.push(Attribute::NoConceal), 29 => attrs.push(Attribute::NoStrikethrough),
                30..=37 => attrs.push(Attribute::Foreground(Self::map_basic_color(param - 30, false))),
                39 => attrs.push(Attribute::Foreground(Color::Default)),
                40..=47 => attrs.push(Attribute::Background(Self::map_basic_color(param - 40, false))),
                49 => attrs.push(Attribute::Background(Color::Default)),
                53 => attrs.push(Attribute::Overlined),
                55 => attrs.push(Attribute::NoUnderline), // Standard says NoOverlined, but often treated as NoUnderline? Check behavior. Let's assume NoUnderline for now.
                58 => { if let Some(color) = Self::parse_extended_color(&mut iter) { attrs.push(Attribute::UnderlineColor(color)); } }
                59 => attrs.push(Attribute::UnderlineColor(Color::Default)),
                90..=97 => attrs.push(Attribute::Foreground(Self::map_basic_color(param - 90, true))),
                100..=107 => attrs.push(Attribute::Background(Self::map_basic_color(param - 100, true))),
                38 => { if let Some(color) = Self::parse_extended_color(&mut iter) { attrs.push(Attribute::Foreground(color)); } }
                48 => { if let Some(color) = Self::parse_extended_color(&mut iter) { attrs.push(Attribute::Background(color)); } }
                _ => { /* Ignore unknown SGR codes */ }
            }
        }
        // If only Reset was provided, return just that.
        if attrs.len() > 1 && attrs[0] == Attribute::Reset {
            attrs.remove(0);
        }
        // If the list ended up empty after processing (e.g., CSI;m), treat as Reset.
        if attrs.is_empty() {
             attrs.push(Attribute::Reset);
        }
        attrs
    }

    fn map_basic_color(code: u16, bright: bool) -> Color {
        match (bright, code) {
            (false, 0) => Color::Black, (false, 1) => Color::Red, (false, 2) => Color::Green, (false, 3) => Color::Yellow,
            (false, 4) => Color::Blue, (false, 5) => Color::Magenta, (false, 6) => Color::Cyan, (false, 7) => Color::White,
            (true, 0) => Color::BrightBlack, (true, 1) => Color::BrightRed, (true, 2) => Color::BrightGreen, (true, 3) => Color::BrightYellow,
            (true, 4) => Color::BrightBlue, (true, 5) => Color::BrightMagenta, (true, 6) => Color::BrightCyan, (true, 7) => Color::BrightWhite,
            _ => Color::Default, // Should not happen with valid code range
        }
    }

    fn parse_extended_color(iter: &mut Peekable<Iter<u16>>) -> Option<Color> {
        match iter.next() {
            Some(&5) => { // 256-color mode
                iter.next().and_then(|&idx| {
                    if idx <= 255 { Some(Color::Indexed(idx as u8)) } else { None }
                })
            }
            Some(&2) => { // RGB mode
                let r = iter.next().map(|&v| v as u8);
                let g = iter.next().map(|&v| v as u8);
                let b = iter.next().map(|&v| v as u8);
                match (r, g, b) {
                    (Some(r_val), Some(g_val), Some(b_val)) => Some(Color::Rgb(r_val, g_val, b_val)),
                    _ => None, // Not enough parameters for RGB
                }
            }
            _ => None, // Invalid or unsupported color mode specifier
        }
    }

    pub(crate) fn from_csi(
        params: Vec<u16>,
        intermediates: Vec<u8>,
        is_private: bool, // Flag set by parser if '?' etc. is seen
        final_byte: u8,
    ) -> Option<Self> {
        let param_or = |idx: usize, default: u16| params.get(idx).copied().unwrap_or(default);
        // Helper for parameters defaulting to 1 (like cursor movements, counts)
        let param_or_1 = |idx: usize| param_or(idx, 1).max(1);

        // Check for private marker intermediate explicitly if needed
        let has_private_intermediate = intermediates.contains(&b'?') || intermediates.contains(&b'>') || intermediates.contains(&b'!'); // Add others if relevant

        match (is_private || has_private_intermediate, intermediates.as_slice(), final_byte) {
            // Mode Setting - Check private flag OR intermediate marker
            (true, _, b'h') => Some(AnsiCommand::Csi(CsiCommand::SetModePrivate(param_or(0, 0)))),
            (false, b"", b'h') => Some(AnsiCommand::Csi(CsiCommand::SetMode(param_or(0, 0)))), // Standard mode set
            (true, _, b'l') => Some(AnsiCommand::Csi(CsiCommand::ResetModePrivate(param_or(0, 0)))),
            (false, b"", b'l') => Some(AnsiCommand::Csi(CsiCommand::ResetMode(param_or(0, 0)))), // Standard mode reset

            // Cursor Movement (Should not be private)
            (false, b"", b'A') => Some(AnsiCommand::Csi(CsiCommand::CursorUp(param_or_1(0)))),
            (false, b"", b'B') => Some(AnsiCommand::Csi(CsiCommand::CursorDown(param_or_1(0)))),
            (false, b"", b'C') => Some(AnsiCommand::Csi(CsiCommand::CursorForward(param_or_1(0)))),
            (false, b"", b'D') => Some(AnsiCommand::Csi(CsiCommand::CursorBackward(param_or_1(0)))),
            (false, b"", b'E') => Some(AnsiCommand::Csi(CsiCommand::CursorNextLine(param_or_1(0)))),
            (false, b"", b'F') => Some(AnsiCommand::Csi(CsiCommand::CursorPrevLine(param_or_1(0)))),
            (false, b"", b'G') => Some(AnsiCommand::Csi(CsiCommand::CursorCharacterAbsolute(param_or_1(0)))),
            (false, b"", b'H') | (false, b"", b'f') => { // CUP / HVP
                 let row = param_or_1(0); let col = param_or_1(1);
                 Some(AnsiCommand::Csi(CsiCommand::CursorPosition(row, col)))
            }
            (false, b"", b'd') => Some(AnsiCommand::Csi(CsiCommand::CursorPosition(param_or_1(0), 1))), // VPA - row only, col 1

             // Erasing
            (false, b"", b'J') => Some(AnsiCommand::Csi(CsiCommand::EraseInDisplay(param_or(0, 0)))), // ED - param 0 is default
            (false, b"", b'K') => Some(AnsiCommand::Csi(CsiCommand::EraseInLine(param_or(0, 0)))), // EL - param 0 is default
            (false, b"", b'X') => Some(AnsiCommand::Csi(CsiCommand::EraseCharacter(param_or_1(0)))), // ECH - defaults to 1
            // Inserting/Deleting Chars/Lines
            (false, b"", b'@') => Some(AnsiCommand::Csi(CsiCommand::InsertCharacter(param_or_1(0)))), // ICH
            (false, b"", b'L') => Some(AnsiCommand::Csi(CsiCommand::InsertLine(param_or_1(0)))), // IL
            (false, b"", b'P') => Some(AnsiCommand::Csi(CsiCommand::DeleteCharacter(param_or_1(0)))), // DCH
            (false, b"", b'M') => Some(AnsiCommand::Csi(CsiCommand::DeleteLine(param_or_1(0)))), // DL
             // Scrolling
            (false, b"", b'S') => Some(AnsiCommand::Csi(CsiCommand::ScrollUp(param_or_1(0)))), // SU
            (false, b"", b'T') => Some(AnsiCommand::Csi(CsiCommand::ScrollDown(param_or_1(0)))), // SD
             // Tabulation
            (false, b"", b'g') => Some(AnsiCommand::Csi(CsiCommand::ClearTabStops(param_or(0, 0)))), // TBC - param 0 default
             // Graphics Rendition
            (false, b"", b'm') => Some(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(Self::parse_sgr(params)))), // SGR
             // Status Reports
            (false, b"", b'n') => Some(AnsiCommand::Csi(CsiCommand::DeviceStatusReport(param_or(0, 0)))), // DSR - param 0 default
             // Cursor Saving/Restoring (SCO variants)
            (false, b"", b's') => Some(AnsiCommand::Csi(CsiCommand::SaveCursor)),
            (false, b"", b'u') => Some(AnsiCommand::Csi(CsiCommand::RestoreCursor)),
            // DECSTBM - Set Scrolling Region
            (false, b"", b'r') => {
                let top = param_or(0, 1); // Default top is 1
                let bottom = param_or(1, 0); // Default bottom is 0 (often means last line of screen)
                Some(AnsiCommand::Csi(CsiCommand::SetScrollingRegion { top, bottom }))
            }
            // Default: Unsupported or Error
            _ => {
                // Log the unsupported sequence details before returning None
                 warn!(
                     "Unsupported CSI sequence: Private={}, Intermediates={:?}, Final={}({}) Params={:?}",
                     is_private || has_private_intermediate,
                     intermediates,
                     final_byte as char, final_byte,
                     params
                 );
                 None
            }
        }
    }
}

impl fmt::Display for C0Control {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

