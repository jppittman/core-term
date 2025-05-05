// src/ansi/commands.rs

//! Defines the `AnsiCommand` enum representing parsed ANSI escape sequences
//! and related helper enums/structs.

use std::fmt;
use std::iter::Peekable;
use std::slice::Iter;

// --- Color Definitions ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black, Red, Green, Yellow, Blue, Magenta, Cyan, White,
    BrightBlack, BrightRed, BrightGreen, BrightYellow, BrightBlue, BrightMagenta, BrightCyan, BrightWhite,
    Indexed(u8),
    Rgb(u8, u8, u8),
    Default,
}

// --- SGR Attributes ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attribute {
    Reset, Bold, Faint, Italic, Underline, BlinkSlow, BlinkRapid, Reverse, Conceal, Strikethrough,
    UnderlineDouble, NoBold, NoItalic, NoUnderline, NoBlink, NoReverse, NoConceal, NoStrikethrough,
    Foreground(Color), Background(Color), Overlined, UnderlineColor(Color),
}

// --- C0 Control Enum ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum C0Control {
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
pub enum CsiCommand {
    CursorUp(u16), CursorDown(u16), CursorForward(u16), CursorBackward(u16),
    CursorNextLine(u16), CursorPrevLine(u16), CursorCharacterAbsolute(u16),
    CursorPosition(u16, u16),
    EraseInDisplay(u16), EraseInLine(u16), EraseCharacter(u16),
    InsertCharacter(u16), InsertLine(u16), DeleteCharacter(u16), DeleteLine(u16),
    ScrollUp(u16), ScrollDown(u16),
    SetTabStop,
    ClearTabStops(u16),
    SetGraphicsRendition(Vec<Attribute>),
    SetMode(u16), ResetMode(u16), SetModePrivate(u16), ResetModePrivate(u16),
    DeviceStatusReport(u16),
    SaveCursorAnsi, RestoreCursorAnsi, SaveCursor, RestoreCursor,
    Reset,
    Unsupported(Vec<u8>, Option<u8>),
}

// --- ESC Command Enum ---
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscCommand {
    SetTabStop, Index, NextLine, ReverseIndex, SaveCursor, RestoreCursor,
    ResetToInitialState,
    SelectCharacterSet(char, char),
    SingleShift2, SingleShift3,
}

// --- Main AnsiCommand Enum ---
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnsiCommand {
    Print(char),
    C0Control(C0Control),
    C1Control(u8),
    Csi(CsiCommand),
    Esc(EscCommand),
    Osc(Vec<u8>), Dcs(Vec<u8>), Pm(Vec<u8>), Apc(Vec<u8>),
    StringTerminator,
    Ignore(u8), Error(u8),
}

impl AnsiCommand {
    pub fn from_c0(byte: u8) -> Option<Self> {
        C0Control::from_byte(byte).map(AnsiCommand::C0Control)
    }

    pub fn from_c1(byte: u8) -> Option<Self> {
         match byte {
             0x84 => Some(AnsiCommand::Esc(EscCommand::Index)),
             0x85 => Some(AnsiCommand::Esc(EscCommand::NextLine)),
             0x88 => Some(AnsiCommand::Esc(EscCommand::SetTabStop)),
             0x8D => Some(AnsiCommand::Esc(EscCommand::ReverseIndex)),
             0x8E => Some(AnsiCommand::Esc(EscCommand::SingleShift2)),
             0x8F => Some(AnsiCommand::Esc(EscCommand::SingleShift3)),
             0x90 | 0x9B | 0x9C | 0x9D | 0x9E | 0x9F => None,
             _ => Some(AnsiCommand::C1Control(byte)),
         }
    }

    pub fn from_esc(final_char: char) -> Option<Self> {
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

    /// Constructor for ESC sequences with an intermediate byte (like SCS).
    /// Moved here from parser.rs
    pub fn from_esc_intermediate(intermediate: char, final_char: char) -> Option<Self> {
         if ['(', ')', '*', '+'].contains(&intermediate) {
             if final_char.is_ascii_uppercase() || final_char.is_ascii_digit() {
                 Some(AnsiCommand::Esc(EscCommand::SelectCharacterSet(intermediate, final_char)))
             } else { None }
         } else { None }
     }

    fn parse_sgr(params: Vec<u16>) -> Vec<Attribute> {
        let mut attrs = Vec::new();
        if params.is_empty() || params == [0] {
            attrs.push(Attribute::Reset);
            return attrs;
        }
        let mut iter = params.iter().peekable();
        while let Some(param) = iter.next() {
            match param {
                0 => attrs.push(Attribute::Reset), 1 => attrs.push(Attribute::Bold),
                2 => attrs.push(Attribute::Faint), 3 => attrs.push(Attribute::Italic),
                4 => attrs.push(Attribute::Underline), 5 => attrs.push(Attribute::BlinkSlow),
                6 => attrs.push(Attribute::BlinkRapid), 7 => attrs.push(Attribute::Reverse),
                8 => attrs.push(Attribute::Conceal), 9 => attrs.push(Attribute::Strikethrough),
                21 => attrs.push(Attribute::UnderlineDouble), 22 => attrs.push(Attribute::NoBold),
                23 => attrs.push(Attribute::NoItalic), 24 => attrs.push(Attribute::NoUnderline),
                25 => attrs.push(Attribute::NoBlink), 27 => attrs.push(Attribute::NoReverse),
                28 => attrs.push(Attribute::NoConceal), 29 => attrs.push(Attribute::NoStrikethrough),
                30..=37 => attrs.push(Attribute::Foreground(Self::map_basic_color(*param - 30, false))),
                39 => attrs.push(Attribute::Foreground(Color::Default)),
                40..=47 => attrs.push(Attribute::Background(Self::map_basic_color(*param - 40, false))),
                49 => attrs.push(Attribute::Background(Color::Default)),
                53 => attrs.push(Attribute::Overlined), 55 => attrs.push(Attribute::NoUnderline), // Should be NoOverlined?
                58 => { if let Some(color) = Self::parse_extended_color(&mut iter) { attrs.push(Attribute::UnderlineColor(color)); } }
                59 => attrs.push(Attribute::UnderlineColor(Color::Default)),
                90..=97 => attrs.push(Attribute::Foreground(Self::map_basic_color(*param - 90, true))),
                100..=107 => attrs.push(Attribute::Background(Self::map_basic_color(*param - 100, true))),
                38 => { if let Some(color) = Self::parse_extended_color(&mut iter) { attrs.push(Attribute::Foreground(color)); } }
                48 => { if let Some(color) = Self::parse_extended_color(&mut iter) { attrs.push(Attribute::Background(color)); } }
                _ => {}
            }
        }
        attrs
    }

    fn map_basic_color(code: u16, bright: bool) -> Color {
        match (bright, code) {
            (false, 0) => Color::Black, (false, 1) => Color::Red, (false, 2) => Color::Green, (false, 3) => Color::Yellow,
            (false, 4) => Color::Blue, (false, 5) => Color::Magenta, (false, 6) => Color::Cyan, (false, 7) => Color::White,
            (true, 0) => Color::BrightBlack, (true, 1) => Color::BrightRed, (true, 2) => Color::BrightGreen, (true, 3) => Color::BrightYellow,
            (true, 4) => Color::BrightBlue, (true, 5) => Color::BrightMagenta, (true, 6) => Color::BrightCyan, (true, 7) => Color::BrightWhite,
            _ => Color::Default,
        }
    }

    fn parse_extended_color(iter: &mut Peekable<Iter<u16>>) -> Option<Color> {
        match iter.next() {
            Some(5) => iter.next().map(|&idx| Color::Indexed(idx as u8)),
            Some(2) => {
                let r = iter.next().map(|&v| v as u8);
                let g = iter.next().map(|&v| v as u8);
                let b = iter.next().map(|&v| v as u8);
                match (r, g, b) {
                    (Some(r_val), Some(g_val), Some(b_val)) => Some(Color::Rgb(r_val, g_val, b_val)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub fn from_csi(
        params: Vec<u16>,
        intermediates: Vec<u8>,
        is_private: bool,
        final_byte: u8,
    ) -> Option<Self> {
        let param_or = |idx: usize, default: u16| params.get(idx).copied().unwrap_or(default);
        let param_or_min = |idx: usize, default: u16, min: u16| param_or(idx, default).max(min);

        match (is_private, intermediates.as_slice(), final_byte) {
            (true, b"", b'h') => Some(AnsiCommand::Csi(CsiCommand::SetModePrivate(param_or(0, 0)))),
            (false, b"", b'h') => Some(AnsiCommand::Csi(CsiCommand::SetMode(param_or(0, 0)))),
            (true, b"", b'l') => Some(AnsiCommand::Csi(CsiCommand::ResetModePrivate(param_or(0, 0)))),
            (false, b"", b'l') => Some(AnsiCommand::Csi(CsiCommand::ResetMode(param_or(0, 0)))),
            (false, b"", b'A') => Some(AnsiCommand::Csi(CsiCommand::CursorUp(param_or_min(0, 1, 1)))),
            (false, b"", b'B') => Some(AnsiCommand::Csi(CsiCommand::CursorDown(param_or_min(0, 1, 1)))),
            (false, b"", b'C') => Some(AnsiCommand::Csi(CsiCommand::CursorForward(param_or_min(0, 1, 1)))),
            (false, b"", b'D') => Some(AnsiCommand::Csi(CsiCommand::CursorBackward(param_or_min(0, 1, 1)))),
            (false, b"", b'E') => Some(AnsiCommand::Csi(CsiCommand::CursorNextLine(param_or_min(0, 1, 1)))),
            (false, b"", b'F') => Some(AnsiCommand::Csi(CsiCommand::CursorPrevLine(param_or_min(0, 1, 1)))),
            (false, b"", b'G') => Some(AnsiCommand::Csi(CsiCommand::CursorCharacterAbsolute(param_or_min(0, 1, 1)))),
            (false, b"", b'H') | (false, b"", b'f') => {
                 let row = param_or_min(0, 1, 1); let col = param_or_min(1, 1, 1);
                 Some(AnsiCommand::Csi(CsiCommand::CursorPosition(row, col)))
            }
            (false, b"", b'J') => Some(AnsiCommand::Csi(CsiCommand::EraseInDisplay(param_or(0, 0)))),
            (false, b"", b'K') => Some(AnsiCommand::Csi(CsiCommand::EraseInLine(param_or(0, 0)))),
            (false, b"", b'X') => Some(AnsiCommand::Csi(CsiCommand::EraseCharacter(param_or_min(0, 1, 1)))),
            (false, b"", b'@') => Some(AnsiCommand::Csi(CsiCommand::InsertCharacter(param_or_min(0, 1, 1)))),
            (false, b"", b'L') => Some(AnsiCommand::Csi(CsiCommand::InsertLine(param_or_min(0, 1, 1)))),
            (false, b"", b'P') => Some(AnsiCommand::Csi(CsiCommand::DeleteCharacter(param_or_min(0, 1, 1)))),
            (false, b"", b'M') => Some(AnsiCommand::Csi(CsiCommand::DeleteLine(param_or_min(0, 1, 1)))),
            (false, b"", b'S') => Some(AnsiCommand::Csi(CsiCommand::ScrollUp(param_or_min(0, 1, 1)))),
            (false, b"", b'T') => Some(AnsiCommand::Csi(CsiCommand::ScrollDown(param_or_min(0, 1, 1)))),
            (false, b"", b'g') => Some(AnsiCommand::Csi(CsiCommand::ClearTabStops(param_or(0, 0)))),
            (false, b"", b'm') => Some(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(Self::parse_sgr(params)))),
            (false, b"", b'n') => Some(AnsiCommand::Csi(CsiCommand::DeviceStatusReport(param_or(0, 0)))),
            (false, b"", b's') => Some(AnsiCommand::Csi(CsiCommand::SaveCursor)),
            (false, b"", b'u') => Some(AnsiCommand::Csi(CsiCommand::RestoreCursor)),
            _ => None,
        }
    }
}

impl fmt::Display for C0Control {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

