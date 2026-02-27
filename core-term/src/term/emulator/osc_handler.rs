// src/term/emulator/osc_handler.rs

//! OSC (Operating System Command) handler for the terminal emulator.
//!
//! Handles the following OSC sequences:
//!
//! | OSC Code | Function | Direction |
//! |----------|----------|-----------|
//! | 0, 1, 2  | Set icon name / window title | App → Terminal |
//! | 4        | Query/set 256-color palette entry | Bidirectional |
//! | 7        | Current working directory | App → Terminal |
//! | 10       | Query/set foreground color | Bidirectional |
//! | 11       | Query/set background color | Bidirectional |
//! | 12       | Query/set cursor color | Bidirectional |
//! | 52       | Clipboard manipulation (base64) | Bidirectional |
//! | 104      | Reset palette color | App → Terminal |
//! | 110      | Reset foreground color | App → Terminal |
//! | 111      | Reset background color | App → Terminal |
//! | 112      | Reset cursor color | App → Terminal |
//!
//! Color query responses use the X11 format: `rgb:RRRR/GGGG/BBBB`
//! where each component is scaled to 16-bit (0x00 → 0000, 0xFF → ffff).

use super::TerminalEmulator;
use crate::config::CONFIG;
use crate::term::action::EmulatorAction;
use log::{debug, warn};

// --- Base64 Codec (minimal, no external dependency) ---

/// Decode base64 bytes into raw bytes. Returns None on invalid input.
fn base64_decode(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for &byte in input {
        if byte == b'=' || byte == b'\n' || byte == b'\r' {
            continue;
        }
        let val = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

// --- X11 Color Format ---

/// Format an (r, g, b) tuple as an X11 color string: `rgb:RRRR/GGGG/BBBB`.
/// Each 8-bit component is scaled to 16-bit by duplication (0xAB → 0xABAB).
fn format_x11_color(r: u8, g: u8, b: u8) -> String {
    // Scale 8-bit to 16-bit: 0xAB → 0xABAB
    let r16 = (r as u16) << 8 | r as u16;
    let g16 = (g as u16) << 8 | g as u16;
    let b16 = (b as u16) << 8 | b as u16;
    format!("rgb:{:04x}/{:04x}/{:04x}", r16, g16, b16)
}

/// Parse an X11 color string like `rgb:RRRR/GGGG/BBBB` or `#RRGGBB` into (r, g, b).
fn parse_x11_color(s: &str) -> Option<(u8, u8, u8)> {
    if let Some(hex) = s.strip_prefix("rgb:") {
        let parts: Vec<&str> = hex.split('/').collect();
        if parts.len() == 3 {
            let r = u16::from_str_radix(parts[0], 16).ok()?;
            let g = u16::from_str_radix(parts[1], 16).ok()?;
            let b = u16::from_str_radix(parts[2], 16).ok()?;
            // Scale down from whatever width to 8-bit
            let scale = |v: u16, len: usize| -> u8 {
                match len {
                    1 => (v << 4 | v) as u8,
                    2 => v as u8,
                    3 => (v >> 4) as u8,
                    4 => (v >> 8) as u8,
                    _ => v as u8,
                }
            };
            return Some((
                scale(r, parts[0].len()),
                scale(g, parts[1].len()),
                scale(b, parts[2].len()),
            ));
        }
    } else if let Some(hex) = s.strip_prefix('#') {
        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            return Some((r, g, b));
        }
    }
    None
}

/// Split raw OSC data into (Ps numeric code, content bytes after the semicolon).
///
/// Parses the command prefix as bytes to avoid converting potentially large
/// payloads (e.g. OSC 52 base64 blobs) through `from_utf8_lossy` up front.
fn split_osc_prefix(data: &[u8]) -> Option<(u32, &[u8])> {
    let semi_pos = data.iter().position(|&b| b == b';');
    let (ps_bytes, content) = match semi_pos {
        Some(pos) => (&data[..pos], &data[pos + 1..]),
        None => (data, &[] as &[u8]),
    };
    // Parse the numeric prefix (ASCII digits only)
    let mut ps: u32 = 0;
    for &b in ps_bytes {
        match b {
            b'0'..=b'9' => {
                ps = ps.checked_mul(10)?.checked_add((b - b'0') as u32)?;
            }
            _ => return None,
        }
    }
    Some((ps, content))
}

impl TerminalEmulator {
    pub(super) fn handle_osc(&mut self, data: Vec<u8>) -> Option<EmulatorAction> {
        let (ps, content_bytes) = match split_osc_prefix(&data) {
            Some(pair) => pair,
            None => {
                warn!(
                    "Malformed OSC sequence: {:?}",
                    String::from_utf8_lossy(&data)
                );
                return None;
            }
        };

        // Only convert content to a string for commands that need it.
        // OSC 52 base64 payloads are decoded directly from bytes below.
        match ps {
            // OSC 0 - Set Icon Name and Window Title
            // OSC 1 - Set Icon Name (we treat same as title)
            // OSC 2 - Set Window Title
            0 | 1 | 2 => {
                let content_str = String::from_utf8_lossy(content_bytes);
                Some(EmulatorAction::SetTitle(content_str.into_owned()))
            }

            // OSC 4 - Query/set 256-color palette entry
            // Format: OSC 4 ; index ; ? ST (query)
            // Format: OSC 4 ; index ; color ST (set)
            4 => {
                let content_str = String::from_utf8_lossy(content_bytes);
                self.handle_osc_palette(&content_str)
            }

            // OSC 7 - Current Working Directory
            // Format: OSC 7 ; file://hostname/path ST
            7 => {
                let content_str = String::from_utf8_lossy(content_bytes);
                self.handle_osc_cwd(&content_str)
            }

            // OSC 10 - Query/set foreground color
            10 => {
                let content_str = String::from_utf8_lossy(content_bytes);
                self.handle_osc_color_query(10, &content_str)
            }

            // OSC 11 - Query/set background color
            11 => {
                let content_str = String::from_utf8_lossy(content_bytes);
                self.handle_osc_color_query(11, &content_str)
            }

            // OSC 12 - Query/set cursor color
            12 => {
                let content_str = String::from_utf8_lossy(content_bytes);
                self.handle_osc_color_query(12, &content_str)
            }

            // OSC 52 - Clipboard manipulation
            // Format: OSC 52 ; selection ; base64data ST
            // Parsed directly from bytes to avoid needless UTF-8 conversion
            // of potentially large base64 payloads.
            52 => self.handle_osc_clipboard_bytes(content_bytes),

            // OSC 104 - Reset palette color
            // Format: OSC 104 ; index ST (reset specific) or OSC 104 ST (reset all)
            104 => {
                debug!(
                    "OSC 104: Reset palette color (index='{}')",
                    String::from_utf8_lossy(content_bytes)
                );
                None
            }

            // OSC 110 - Reset foreground color
            110 => {
                debug!("OSC 110: Reset foreground color to default");
                None
            }

            // OSC 111 - Reset background color
            111 => {
                debug!("OSC 111: Reset background color to default");
                None
            }

            // OSC 112 - Reset cursor color
            112 => {
                debug!("OSC 112: Reset cursor color to default");
                None
            }

            _ => {
                debug!(
                    "Unhandled OSC command code: Ps={}, Pt='{}'",
                    ps,
                    String::from_utf8_lossy(content_bytes)
                );
                None
            }
        }
    }

    /// Handle OSC 4: Query/set 256-color palette entries.
    ///
    /// Query format: `index;?`
    /// Response: `\x1b]4;index;rgb:RRRR/GGGG/BBBB\x1b\\`
    fn handle_osc_palette(&self, content: &str) -> Option<EmulatorAction> {
        // Content is "index;?" for query, or "index;color" for set
        let parts: Vec<&str> = content.splitn(2, ';').collect();
        if parts.len() != 2 {
            warn!("OSC 4: Malformed palette request: '{}'", content);
            return None;
        }

        let index = match parts[0].parse::<u16>() {
            Ok(i) if i <= 255 => i as u8,
            _ => {
                warn!("OSC 4: Invalid palette index: '{}'", parts[0]);
                return None;
            }
        };

        if parts[1] == "?" {
            // Query palette color
            let color = crate::color::Color::Indexed(index);
            let (r, g, b) = color.to_rgb_tuple();
            let response = format!(
                "\x1b]4;{};{}\x1b\\",
                index,
                format_x11_color(r, g, b)
            );
            debug!("OSC 4: Palette query for index {} → ({}, {}, {})", index, r, g, b);
            Some(EmulatorAction::WritePty(response.into_bytes()))
        } else if let Some((_r, _g, _b)) = parse_x11_color(parts[1]) {
            // Set palette color (logged but not applied - would need mutable palette)
            debug!("OSC 4: Set palette index {} to '{}'", index, parts[1]);
            None
        } else {
            warn!("OSC 4: Unrecognized color format: '{}'", parts[1]);
            None
        }
    }

    /// Handle OSC 7: Current working directory.
    ///
    /// Format: `file://hostname/path/to/dir`
    fn handle_osc_cwd(&self, content: &str) -> Option<EmulatorAction> {
        // Strip the file:// URI scheme and optional hostname
        let path = if let Some(rest) = content.strip_prefix("file://") {
            // Skip the hostname (everything up to the next '/')
            if let Some(slash_pos) = rest.find('/') {
                &rest[slash_pos..]
            } else {
                // Hostname without path component (e.g. "file://localhost") → root
                "/"
            }
        } else {
            // Some shells send just the path without file:// prefix
            content
        };

        if !path.is_empty() {
            debug!("OSC 7: Working directory → {}", path);
            Some(EmulatorAction::SetWorkingDirectory(path.to_string()))
        } else {
            debug!("OSC 7: Empty working directory");
            None
        }
    }

    /// Handle OSC 10/11/12: Query or set foreground/background/cursor color.
    ///
    /// Query: content is `?`
    /// Set: content is an X11 color spec like `rgb:RRRR/GGGG/BBBB` or `#RRGGBB`
    ///
    /// Response format: `\x1b]Ps;rgb:RRRR/GGGG/BBBB\x1b\\`
    fn handle_osc_color_query(&self, ps: u32, content: &str) -> Option<EmulatorAction> {
        if content == "?" {
            // Query: look up the configured color
            let color = match ps {
                10 => CONFIG.colors.foreground,
                11 => CONFIG.colors.background,
                12 => CONFIG.colors.cursor,
                _ => return None,
            };
            let (r, g, b) = color.to_rgb_tuple();
            let response = format!("\x1b]{};{}\x1b\\", ps, format_x11_color(r, g, b));
            debug!(
                "OSC {}: Color query → ({}, {}, {})",
                ps, r, g, b
            );
            Some(EmulatorAction::WritePty(response.into_bytes()))
        } else if let Some((_r, _g, _b)) = parse_x11_color(content) {
            // Set color (logged but color scheme mutation not implemented)
            debug!("OSC {}: Set color to '{}'", ps, content);
            None
        } else {
            warn!("OSC {}: Unrecognized color format: '{}'", ps, content);
            None
        }
    }

    /// Handle OSC 52: Clipboard manipulation (byte-level).
    ///
    /// Operates on raw bytes to avoid converting potentially large base64
    /// payloads through `from_utf8_lossy`. The selection identifier is always
    /// short ASCII, and the payload is pure base64 (also ASCII), so byte-level
    /// parsing is both correct and efficient.
    ///
    /// Format: `selection;base64data`
    ///
    /// Selection identifiers:
    /// - `c` = clipboard
    /// - `p` = primary (X11)
    /// - `s` = select
    /// - `0`-`7` = cut buffers
    ///
    /// If base64data is `?`, this is a query (respond with clipboard contents).
    /// Otherwise, decode the base64 and set the clipboard.
    fn handle_osc_clipboard_bytes(&self, content: &[u8]) -> Option<EmulatorAction> {
        if !CONFIG.behavior.allow_osc52_clipboard {
            debug!("OSC 52: Clipboard access disabled by configuration");
            return None;
        }

        let semi_pos = content.iter().position(|&b| b == b';');
        let (selection_bytes, payload) = match semi_pos {
            Some(pos) => (&content[..pos], &content[pos + 1..]),
            None => {
                warn!(
                    "OSC 52: Malformed clipboard request: '{}'",
                    String::from_utf8_lossy(content)
                );
                return None;
            }
        };

        // Selection is short ASCII (typically "c", "p", "s", etc.)
        let selection = std::str::from_utf8(selection_bytes).unwrap_or("c");

        if payload == b"?" {
            // Query clipboard - respond with empty (we don't have access to clipboard contents
            // from within the emulator; the orchestrator would need to handle this)
            debug!(
                "OSC 52: Clipboard query for selection '{}' (responding empty)",
                selection
            );
            let response = format!("\x1b]52;{};\x1b\\", selection);
            Some(EmulatorAction::WritePty(response.into_bytes()))
        } else if payload.is_empty() {
            // Empty payload = clear clipboard
            debug!("OSC 52: Clear clipboard for selection '{}'", selection);
            Some(EmulatorAction::CopyToClipboard(String::new()))
        } else {
            // Decode base64 payload directly from bytes and set clipboard
            match base64_decode(payload) {
                Some(decoded_bytes) => match String::from_utf8(decoded_bytes) {
                    Ok(text) => {
                        debug!(
                            "OSC 52: Set clipboard for selection '{}' ({} bytes)",
                            selection,
                            text.len()
                        );
                        Some(EmulatorAction::CopyToClipboard(text))
                    }
                    Err(e) => {
                        warn!("OSC 52: Decoded base64 is not valid UTF-8: {}", e);
                        None
                    }
                },
                None => {
                    warn!("OSC 52: Invalid base64 payload");
                    None
                }
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const B64_ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// Encode raw bytes into base64 (test-only, used for roundtrip validation).
    fn base64_encode(input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity((input.len() + 2) / 3 * 4);
        for chunk in input.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
            let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
            let triple = (b0 << 16) | (b1 << 8) | b2;

            out.push(B64_ALPHABET[((triple >> 18) & 0x3F) as usize]);
            out.push(B64_ALPHABET[((triple >> 12) & 0x3F) as usize]);
            if chunk.len() > 1 {
                out.push(B64_ALPHABET[((triple >> 6) & 0x3F) as usize]);
            } else {
                out.push(b'=');
            }
            if chunk.len() > 2 {
                out.push(B64_ALPHABET[(triple & 0x3F) as usize]);
            } else {
                out.push(b'=');
            }
        }
        out
    }

    // --- Base64 Tests ---

    #[test]
    fn base64_encode_empty() {
        assert_eq!(base64_encode(b""), b"");
    }

    #[test]
    fn base64_encode_single_byte() {
        assert_eq!(base64_encode(b"f"), b"Zg==");
    }

    #[test]
    fn base64_encode_two_bytes() {
        assert_eq!(base64_encode(b"fo"), b"Zm8=");
    }

    #[test]
    fn base64_encode_three_bytes() {
        assert_eq!(base64_encode(b"foo"), b"Zm9v");
    }

    #[test]
    fn base64_encode_hello_world() {
        assert_eq!(base64_encode(b"Hello, World!"), b"SGVsbG8sIFdvcmxkIQ==");
    }

    #[test]
    fn base64_decode_empty() {
        assert_eq!(base64_decode(b""), Some(vec![]));
    }

    #[test]
    fn base64_decode_hello() {
        assert_eq!(
            base64_decode(b"SGVsbG8="),
            Some(b"Hello".to_vec())
        );
    }

    #[test]
    fn base64_decode_hello_world() {
        assert_eq!(
            base64_decode(b"SGVsbG8sIFdvcmxkIQ=="),
            Some(b"Hello, World!".to_vec())
        );
    }

    #[test]
    fn base64_roundtrip() {
        let original = b"The quick brown fox jumps over the lazy dog";
        let encoded = base64_encode(original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn base64_roundtrip_binary() {
        let original: Vec<u8> = (0..=255).collect();
        let encoded = base64_encode(&original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn base64_decode_ignores_newlines() {
        assert_eq!(
            base64_decode(b"SGVs\nbG8="),
            Some(b"Hello".to_vec())
        );
    }

    #[test]
    fn base64_decode_invalid_char() {
        assert_eq!(base64_decode(b"SGVs!G8="), None);
    }

    // --- X11 Color Format Tests ---

    #[test]
    fn x11_color_format_black() {
        assert_eq!(format_x11_color(0, 0, 0), "rgb:0000/0000/0000");
    }

    #[test]
    fn x11_color_format_white() {
        assert_eq!(format_x11_color(255, 255, 255), "rgb:ffff/ffff/ffff");
    }

    #[test]
    fn x11_color_format_red() {
        assert_eq!(format_x11_color(205, 0, 0), "rgb:cdcd/0000/0000");
    }

    #[test]
    fn x11_color_parse_full() {
        assert_eq!(
            parse_x11_color("rgb:cdcd/0000/0000"),
            Some((205, 0, 0))
        );
    }

    #[test]
    fn x11_color_parse_hex() {
        assert_eq!(parse_x11_color("#ff8000"), Some((255, 128, 0)));
    }

    #[test]
    fn x11_color_parse_short_components() {
        // Two-hex-digit components (8-bit)
        assert_eq!(parse_x11_color("rgb:ff/80/00"), Some((255, 128, 0)));
    }

    #[test]
    fn x11_color_roundtrip() {
        let (r, g, b) = (42, 128, 200);
        let formatted = format_x11_color(r, g, b);
        let parsed = parse_x11_color(&formatted);
        assert_eq!(parsed, Some((r, g, b)));
    }

    #[test]
    fn x11_color_parse_invalid() {
        assert_eq!(parse_x11_color("not_a_color"), None);
        assert_eq!(parse_x11_color("rgb:"), None);
        assert_eq!(parse_x11_color("#12345"), None);
    }
}
