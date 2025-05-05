// src/ansi/commands.rs

// Assuming Color is defined in glyph.rs - keeping the import for now, though it's unused in this snippet.
// use crate::glyph::Color;

/// Represents a parsed ANSI command.
#[derive(Debug, PartialEq, Clone)]
pub enum AnsiCommand {
    /// A printable character.
    Print(char),
    /// A C0 control character.
    C0Control(C0Control),
    /// A C1 control character (or placeholder for specific ESC sequences).
    C1Control(u8),
    /// A Control Sequence Introducer (CSI) command.
    Csi(CsiCommand),
    /// A Device Control String (DCS) command. Contains the raw bytes of the string.
    Dcs(Vec<u8>),
    /// An Operating System Command (OSC). Contains the raw bytes of the string.
    Osc(Vec<u8>),
    /// A Privacy Message (PM). Contains the raw bytes of the string.
    Pm(Vec<u8>),
    /// An Application Program Command (APC). Contains the raw bytes of the string.
    Apc(Vec<u8>),
    /// An unsupported or unrecognized sequence introducer byte.
    #[allow(dead_code)] // Keep variant for potential future use
    UnsupportedSequenceIntroducer(u8),
    /// A byte that was ignored by the lexer/parser.
    Ignore(u8),
    /// The String Terminator.
    StringTerminator,
    /// An error occurred during parsing.
    Error(u8),
}

/// Represents a C0 control character.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum C0Control {
    NUL, // Null
    SOH, // Start of Heading
    STX, // Start of Text
    ETX, // End of Text
    EOT, // End of Transmission
    ENQ, // Enquiry
    ACK, // Acknowledge
    BEL, // Bell
    BS,  // Backspace
    HT,  // Horizontal Tab
    LF,  // Line Feed
    VT,  // Vertical Tab
    FF,  // Form Feed
    CR,  // Carriage Return
    SO,  // Shift Out
    SI,  // Shift In
    DLE, // Data Link Escape
    DC1, // Device Control 1
    DC2, // Device Control 2
    DC3, // Device Control 3
    DC4, // Device Control 4
    NAK, // Negative Acknowledge
    SYN, // Synchronous Idle
    ETB, // End of Transmission Block
    CAN, // Cancel
    EM,  // End of Medium
    SUB, // Substitute
    ESC, // Escape
    FS,  // File Separator
    GS,  // Group Separator
    RS,  // Record Separator
    US,  // Unit Separator
    DEL, // Delete (0x7F)
}

impl C0Control {
    /// Converts a byte to a `C0Control` enum variant.
    /// Returns `None` if the byte is not a valid C0 control character.
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(C0Control::NUL),
            0x01 => Some(C0Control::SOH),
            0x02 => Some(C0Control::STX),
            0x03 => Some(C0Control::ETX),
            0x04 => Some(C0Control::EOT),
            0x05 => Some(C0Control::ENQ),
            0x06 => Some(C0Control::ACK),
            0x07 => Some(C0Control::BEL),
            0x08 => Some(C0Control::BS),
            0x09 => Some(C0Control::HT),
            0x0A => Some(C0Control::LF),
            0x0B => Some(C0Control::VT),
            0x0C => Some(C0Control::FF),
            0x0D => Some(C0Control::CR),
            0x0E => Some(C0Control::SO),
            0x0F => Some(C0Control::SI),
            0x10 => Some(C0Control::DLE),
            0x11 => Some(C0Control::DC1),
            0x12 => Some(C0Control::DC2),
            0x13 => Some(C0Control::DC3),
            0x14 => Some(C0Control::DC4),
            0x15 => Some(C0Control::NAK),
            0x16 => Some(C0Control::SYN),
            0x17 => Some(C0Control::ETB),
            0x18 => Some(C0Control::CAN),
            0x19 => Some(C0Control::EM),
            0x1A => Some(C0Control::SUB),
            0x1B => Some(C0Control::ESC),
            0x1C => Some(C0Control::FS),
            0x1D => Some(C0Control::GS),
            0x1E => Some(C0Control::RS),
            0x1F => Some(C0Control::US),
            0x7F => Some(C0Control::DEL), // DEL is also a C0 control
            _ => None, // Not a C0 control byte
        }
    }
}


/// Represents a parsed CSI (Control Sequence Introducer) command.
#[derive(Debug, PartialEq, Clone)]
pub enum CsiCommand {
    /// Cursor Up Ps Times (default = 1) (CUU).
    CursorUp(usize),
    /// Cursor Down Ps Times (default = 1) (CUD).
    CursorDown(usize),
    /// Cursor Forward Ps Times (default = 1) (CUF).
    CursorForward(usize),
    /// Cursor Backward Ps Times (default = 1) (CUB).
    CursorBackward(usize),
    /// Cursor Next Line Ps Times (default = 1) (CNL).
    CursorNextLine(usize),
    /// Cursor Prev Line Ps Times (default = 1) (CPL).
    CursorPrevLine(usize),
    /// Cursor Character Absolute P s Times (default = 1) (CHA).
    CursorCharacterAbsolute(usize),
    /// Cursor Position [Pn; Pm] (default = 1; 1) (CUP).
    CursorPosition(usize, usize),
    /// Erase in Display Ps Times (default = 0) (ED).
    EraseInDisplay(usize),
    /// Erase in Line Ps Times (default = 0) (EL).
    EraseInLine(usize),
    /// Insert Ps Lines (default = 1) (IL).
    InsertLine(usize),
    /// Delete Ps Lines (default = 1) (DL).
    DeleteLine(usize),
    /// Delete Ps Characters (default = 1) (DCH).
    DeleteCharacter(usize),
    /// Erase Ps Characters (default = 1) (ECH).
    EraseCharacter(usize),
    /// Insert Ps Characters (default = 1) (ICH).
    InsertCharacter(usize),
    /// Set Graphics Rendition [Pn] (SGR).
    SetGraphicsRendition(Vec<usize>),
    /// Save Cursor Position (SCO).
    SaveCursor,
    /// Restore Cursor Position (SCO).
    RestoreCursor,
    /// Set mode (SM).
    SetMode(Vec<usize>),
    /// Reset mode (RM).
    ResetMode(Vec<usize>),
    /// Set private mode (DECSET).
    SetModePrivate(Vec<usize>), // Added for CSI ? Pn h
    /// Reset private mode (DECRST).
    ResetModePrivate(Vec<usize>), // Added for CSI ? Pn l
    /// Device Status Report (DSR).
    DeviceStatusReport(usize),
    /// Save Cursor (DECSC). - Handled via C1Control(ESC_7) in parser
    SaveCursorAnsi,
    /// Restore Cursor (DECRC). - Handled via C1Control(ESC_8) in parser
    RestoreCursorAnsi,
    /// Set Tab Stop (HTS).
    SetTabStop,
    /// Clear Tab Stops (TBC).
    ClearTabStops(usize),
    /// Set/Reset Scrolling Region (DECSTBM). Params: top, bottom (0 means default/extent).
    ScrollingRegion(usize, usize),
    /// Request Terminal Parameters (DECRQPSR).
    RequestTerminalParameters(usize),
    /// Identify Terminal (DA).
    IdentifyTerminal,
    /// Set Cursor Style (DECSCUSR).
    SetCursorStyle(usize),
    /// Select Character Set (SCS). (Handled by 2-byte escapes, but might be represented here)
    SelectCharacterSet(char, char), // G-set, character set
    /// Reset to Initial State (RIS). - Handled via C1Control(ESC_C) in parser
    Reset,
    /// Unsupported CSI command. Contains the raw parameters and final byte.
    Unsupported(Vec<usize>, Option<u8>),

    // --- Variants below are likely unused or handled differently, marked dead_code ---
    #[allow(dead_code)] MouseTracking(bool),
    #[allow(dead_code)] KeyboardAction(bool),
    #[allow(dead_code)] AutomaticRepeat(bool),
    #[allow(dead_code)] SendReceive(bool),
    #[allow(dead_code)] TextCursorEnable(bool),
    #[allow(dead_code)] ShowCursor(bool),
    #[allow(dead_code)] BlinkingCursor(bool),
    #[allow(dead_code)] ColumnMode(bool),
    #[allow(dead_code)] ReportCursorPosition,
    #[allow(dead_code)] SetLeftRightMargin(usize, usize),
    #[allow(dead_code)] SetTopBottomMargin(usize, usize),
    #[allow(dead_code)] SetAnsiMode(usize),
    #[allow(dead_code)] DesignateCharacterSet(char, char),
    #[allow(dead_code)] SoftReset,
    #[allow(dead_code)] ReverseAttributesInRectangle(usize, usize, usize, usize, Vec<usize>),
    #[allow(dead_code)] SelectAreaOfDisplay(usize),
    #[allow(dead_code)] RequestChecksumOfArea(usize, usize, usize, usize, usize),
    #[allow(dead_code)] FillRectangle(usize, usize, usize, usize, char),
    #[allow(dead_code)] EraseRectangle(usize, usize, usize, usize),
    #[allow(dead_code)] CopyRectangle(usize, usize, usize, usize, usize, usize),
    #[allow(dead_code)] EnableLocatorReporting(usize),
    #[allow(dead_code)] SetLocatorEvents(usize),
    #[allow(dead_code)] RequestLocatorPosition(usize),
    #[allow(dead_code)] SetColorPallete(usize, String),
    #[allow(dead_code)] SetForegroundColor(String),
    #[allow(dead_code)] SetBackgroundColor(String),
    #[allow(dead_code)] SetTextCursorColor(String),
}
