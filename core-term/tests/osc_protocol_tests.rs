//! OSC (Operating System Command) Protocol Tests
//!
//! Comprehensive test suite for OSC sequence handling in the terminal emulator.
//! Tests cover the full pipeline: raw bytes → ANSI parser → emulator → action.

mod support;

use core_term::ansi::commands::AnsiCommand;
use core_term::term::action::EmulatorAction;
use support::minimal_test_harness::MinimalTestHarness;

// =============================================================================
// Helper: Parse raw bytes through the ANSI processor and inject into emulator
// =============================================================================

fn process_bytes_and_get_actions(harness: &mut MinimalTestHarness, bytes: &[u8]) -> Vec<EmulatorAction> {
    use core_term::ansi::{AnsiParser, AnsiProcessor};
    let mut processor = AnsiProcessor::new();
    let commands = processor.process_bytes(bytes);
    let mut actions = Vec::new();
    for cmd in commands {
        if let Some(action) = harness.inject_ansi_with_action(cmd) {
            actions.push(action);
        }
    }
    actions
}

// =============================================================================
// OSC 0/1/2 - Set Window Title
// =============================================================================

#[test]
fn osc_0_sets_window_title() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]0;My Terminal Title\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::SetTitle("My Terminal Title".to_string())
    );
}

#[test]
fn osc_2_sets_window_title() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]2;Another Title\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::SetTitle("Another Title".to_string())
    );
}

#[test]
fn osc_1_sets_icon_name() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]1;icon-name\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::SetTitle("icon-name".to_string())
    );
}

#[test]
fn osc_title_with_st_terminator() {
    let mut harness = MinimalTestHarness::new();
    // ST = ESC \ (0x1b 0x5c)
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]0;Title via ST\x1b\\",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::SetTitle("Title via ST".to_string())
    );
}

#[test]
fn osc_title_empty_string() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]0;\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0], EmulatorAction::SetTitle("".to_string()));
}

#[test]
fn osc_title_with_special_chars() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]0;~/projects/core-term (main)\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::SetTitle("~/projects/core-term (main)".to_string())
    );
}

// =============================================================================
// OSC 7 - Current Working Directory
// =============================================================================

#[test]
fn osc_7_sets_working_directory() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]7;file://localhost/home/user/projects\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::SetWorkingDirectory("/home/user/projects".to_string())
    );
}

#[test]
fn osc_7_with_empty_hostname() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]7;file:///tmp/test\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::SetWorkingDirectory("/tmp/test".to_string())
    );
}

#[test]
fn osc_7_bare_path() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]7;/home/user\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::SetWorkingDirectory("/home/user".to_string())
    );
}

// =============================================================================
// OSC 10/11/12 - Color Queries
// =============================================================================

#[test]
fn osc_10_queries_foreground_color() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]10;?\x07",
    );
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        EmulatorAction::WritePty(response) => {
            let response_str = String::from_utf8_lossy(response);
            assert!(
                response_str.starts_with("\x1b]10;rgb:"),
                "Response should start with OSC 10 color: got '{}'",
                response_str
            );
            assert!(
                response_str.ends_with("\x1b\\"),
                "Response should end with ST: got '{}'",
                response_str
            );
        }
        other => panic!("Expected WritePty, got: {:?}", other),
    }
}

#[test]
fn osc_11_queries_background_color() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]11;?\x07",
    );
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        EmulatorAction::WritePty(response) => {
            let response_str = String::from_utf8_lossy(response);
            assert!(
                response_str.starts_with("\x1b]11;rgb:"),
                "Background color response format incorrect: '{}'",
                response_str
            );
        }
        other => panic!("Expected WritePty, got: {:?}", other),
    }
}

#[test]
fn osc_12_queries_cursor_color() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]12;?\x07",
    );
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        EmulatorAction::WritePty(response) => {
            let response_str = String::from_utf8_lossy(response);
            assert!(
                response_str.starts_with("\x1b]12;rgb:"),
                "Cursor color response format incorrect: '{}'",
                response_str
            );
        }
        other => panic!("Expected WritePty, got: {:?}", other),
    }
}

// =============================================================================
// OSC 4 - Palette Color Query
// =============================================================================

#[test]
fn osc_4_queries_palette_color() {
    let mut harness = MinimalTestHarness::new();
    // Query palette index 1 (red)
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]4;1;?\x07",
    );
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        EmulatorAction::WritePty(response) => {
            let response_str = String::from_utf8_lossy(response);
            assert!(
                response_str.starts_with("\x1b]4;1;rgb:"),
                "Palette color response should reference index 1: '{}'",
                response_str
            );
            // Red is (205, 0, 0) → rgb:cdcd/0000/0000
            assert!(
                response_str.contains("cdcd/0000/0000"),
                "Palette index 1 should be red (205,0,0): '{}'",
                response_str
            );
        }
        other => panic!("Expected WritePty, got: {:?}", other),
    }
}

#[test]
fn osc_4_queries_palette_color_index_0() {
    let mut harness = MinimalTestHarness::new();
    // Query palette index 0 (black)
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]4;0;?\x07",
    );
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        EmulatorAction::WritePty(response) => {
            let response_str = String::from_utf8_lossy(response);
            assert!(
                response_str.contains("0000/0000/0000"),
                "Palette index 0 should be black: '{}'",
                response_str
            );
        }
        other => panic!("Expected WritePty, got: {:?}", other),
    }
}

#[test]
fn osc_4_queries_grayscale_palette() {
    let mut harness = MinimalTestHarness::new();
    // Query palette index 232 (first grayscale: gray level 8)
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]4;232;?\x07",
    );
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        EmulatorAction::WritePty(response) => {
            let response_str = String::from_utf8_lossy(response);
            assert!(
                response_str.starts_with("\x1b]4;232;rgb:"),
                "Should respond with palette index 232: '{}'",
                response_str
            );
        }
        other => panic!("Expected WritePty, got: {:?}", other),
    }
}

// =============================================================================
// OSC 52 - Clipboard
// =============================================================================

#[test]
fn osc_52_sets_clipboard_from_base64() {
    let mut harness = MinimalTestHarness::new();
    // "Hello" in base64 is "SGVsbG8="
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]52;c;SGVsbG8=\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::CopyToClipboard("Hello".to_string())
    );
}

#[test]
fn osc_52_sets_clipboard_longer_text() {
    let mut harness = MinimalTestHarness::new();
    // "Hello, World!" in base64 is "SGVsbG8sIFdvcmxkIQ=="
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]52;c;SGVsbG8sIFdvcmxkIQ==\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::CopyToClipboard("Hello, World!".to_string())
    );
}

#[test]
fn osc_52_primary_selection() {
    let mut harness = MinimalTestHarness::new();
    // "test" in base64 is "dGVzdA=="
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]52;p;dGVzdA==\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::CopyToClipboard("test".to_string())
    );
}

#[test]
fn osc_52_empty_payload_clears_clipboard() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]52;c;\x07",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::CopyToClipboard(String::new())
    );
}

#[test]
fn osc_52_query_responds_empty() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]52;c;?\x07",
    );
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        EmulatorAction::WritePty(response) => {
            let response_str = String::from_utf8_lossy(response);
            assert!(
                response_str.starts_with("\x1b]52;c;"),
                "Clipboard query response format incorrect: '{}'",
                response_str
            );
        }
        other => panic!("Expected WritePty for clipboard query, got: {:?}", other),
    }
}

#[test]
fn osc_52_with_st_terminator() {
    let mut harness = MinimalTestHarness::new();
    // Using ESC \ as string terminator instead of BEL
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]52;c;SGVsbG8=\x1b\\",
    );
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0],
        EmulatorAction::CopyToClipboard("Hello".to_string())
    );
}

// =============================================================================
// OSC Reset Commands (104/110/111/112)
// =============================================================================

#[test]
fn osc_104_reset_palette_no_crash() {
    let mut harness = MinimalTestHarness::new();
    // Should not crash, just log
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]104;1\x07",
    );
    // Reset commands currently return None (no action needed)
    assert!(actions.is_empty());
}

#[test]
fn osc_110_reset_foreground_no_crash() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]110\x07",
    );
    assert!(actions.is_empty());
}

#[test]
fn osc_111_reset_background_no_crash() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]111\x07",
    );
    assert!(actions.is_empty());
}

#[test]
fn osc_112_reset_cursor_color_no_crash() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]112\x07",
    );
    assert!(actions.is_empty());
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

#[test]
fn osc_unknown_code_ignored() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]999;some data\x07",
    );
    assert!(actions.is_empty());
}

#[test]
fn osc_52_invalid_base64_ignored() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]52;c;!!!invalid!!!\x07",
    );
    assert!(actions.is_empty(), "Invalid base64 should be silently ignored");
}

#[test]
fn osc_4_invalid_index_ignored() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]4;999;?\x07",
    );
    assert!(actions.is_empty(), "Invalid palette index should be ignored");
}

#[test]
fn osc_interleaved_with_text() {
    let mut harness = MinimalTestHarness::new();
    // Text, then OSC title, then more text
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"Hello\x1b]0;New Title\x07World",
    );
    // Should get exactly one SetTitle action (the Print chars don't produce actions)
    let title_actions: Vec<_> = actions
        .iter()
        .filter(|a| matches!(a, EmulatorAction::SetTitle(_)))
        .collect();
    assert_eq!(title_actions.len(), 1);
    assert_eq!(
        *title_actions[0],
        EmulatorAction::SetTitle("New Title".to_string())
    );
}

#[test]
fn osc_multiple_sequences() {
    let mut harness = MinimalTestHarness::new();
    let actions = process_bytes_and_get_actions(
        &mut harness,
        b"\x1b]0;Title1\x07\x1b]0;Title2\x07",
    );
    let title_actions: Vec<_> = actions
        .iter()
        .filter(|a| matches!(a, EmulatorAction::SetTitle(_)))
        .collect();
    assert_eq!(title_actions.len(), 2);
    assert_eq!(
        *title_actions[0],
        EmulatorAction::SetTitle("Title1".to_string())
    );
    assert_eq!(
        *title_actions[1],
        EmulatorAction::SetTitle("Title2".to_string())
    );
}

#[test]
fn osc_color_query_response_format_valid() {
    let mut harness = MinimalTestHarness::new();
    // Query all three color types and verify response format
    for osc_code in [10, 11, 12] {
        let query = format!("\x1b]{};?\x07", osc_code);
        let actions = process_bytes_and_get_actions(&mut harness, query.as_bytes());
        assert_eq!(actions.len(), 1, "OSC {} should produce one action", osc_code);
        match &actions[0] {
            EmulatorAction::WritePty(response) => {
                let s = String::from_utf8_lossy(response);
                // Verify format: \x1b]PS;rgb:XXXX/XXXX/XXXX\x1b\\
                let prefix = format!("\x1b]{};rgb:", osc_code);
                assert!(s.starts_with(&prefix), "Bad prefix for OSC {}: '{}'", osc_code, s);
                assert!(s.ends_with("\x1b\\"), "Bad suffix for OSC {}: '{}'", osc_code, s);
                // Extract the color part and verify it has 3 components
                let color_part = &s[prefix.len()..s.len() - 2];
                let components: Vec<&str> = color_part.split('/').collect();
                assert_eq!(
                    components.len(), 3,
                    "Color should have 3 components for OSC {}: '{}'",
                    osc_code, color_part
                );
                // Each component should be a valid 4-digit hex
                for comp in &components {
                    assert_eq!(comp.len(), 4, "Component should be 4 hex digits: '{}'", comp);
                    assert!(
                        u16::from_str_radix(comp, 16).is_ok(),
                        "Component should be valid hex: '{}'",
                        comp
                    );
                }
            }
            other => panic!("Expected WritePty for OSC {}, got: {:?}", osc_code, other),
        }
    }
}
