//! This module handles Unicode character width determination using the system's `wcwidth`
//! function via FFI, and locale initialization via a lazily initialized static controller
//! using `std::sync::OnceLock`.

use log::{debug, trace, warn};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint}; // For FFI types
use std::sync::OnceLock; // Standard library for one-time initialization // Logging macros

// --- FFI Declarations ---
unsafe extern "C" {
    fn wcwidth(wc: c_uint) -> c_int;
    fn setlocale(category: c_int, locale: *const c_char) -> *mut c_char;
}

const LC_CTYPE: c_int = 0; // Common value for LC_CTYPE

/// Internal struct to manage the C locale initialization.
/// Its `new` method performs the one-time `setlocale` call.
#[derive(Debug)]
struct LocaleInitializer {
    // This field is a placeholder to signify that initialization logic has run.
    _locale_is_set_guard: (),
}

impl LocaleInitializer {
    /// Performs the one-time C locale initialization for `LC_CTYPE`.
    /// This is called by the `OnceLock` initializer.
    fn new() -> Self {
        // SAFETY: Calling C's setlocale function.
        unsafe {
            let empty_locale_c_str =
                CString::new("").expect("CString::new for empty locale should not fail.");
            if setlocale(LC_CTYPE, empty_locale_c_str.as_ptr()).is_null() {
                warn!(
                    "Failed to set LC_CTYPE locale from environment. Character width calculations might be incorrect or use a default \"C\"/\"POSIX\" locale behavior."
                );
            } else {
                debug!("LC_CTYPE locale set successfully from environment settings for wcwidth.");
            }
        }
        LocaleInitializer {
            _locale_is_set_guard: (),
        }
    }

    /// Internal method to calculate character display width.
    fn char_display_width_internal(&self, c: char) -> usize {
        let wc = c as c_uint;
        // SAFETY: Calling C's wcwidth. Locale is expected to be set by `new()`.
        let width_from_c = unsafe { wcwidth(wc) };

        match width_from_c {
            -1 => {
                if c.is_control() {
                    trace!(
                        "wcwidth returned -1 for control char '{}' (U+{:X}), width is 0.",
                        c,
                        wc
                    );
                    0
                } else {
                    trace!(
                        "wcwidth returned -1 for char '{}' (U+{:X}), defaulting to width 1.",
                        c,
                        wc
                    );
                    1
                }
            }
            0 => {
                trace!("wcwidth returned 0 for char '{}' (U+{:X}).", c, wc);
                0
            }
            1 => 1,
            2 => 2,
            _ => {
                warn!(
                    "wcwidth returned an unexpected positive value: {} for char '{}' (U+{:X}). Defaulting to width 1.",
                    width_from_c, c, wc
                );
                1
            }
        }
    }
}

// Statically allocated OnceLock for our LocaleInitializer.
// The `LocaleInitializer::new()` function will be called to initialize it
// the first time `get_or_init` is invoked.
static GLOBAL_LOCALE_INITIALIZER: OnceLock<LocaleInitializer> = OnceLock::new();

/// Public function to get the display width of a character.
///
/// This function ensures that the C locale (specifically `LC_CTYPE`) has been
/// initialized (on first call) before calling the system's `wcwidth` function.
///
/// # Arguments
/// * `c`: The Rust `char` whose display width is to be determined.
///
/// # Returns
/// * `0` for non-printing characters or characters that do not advance the cursor.
/// * `1` for standard-width printable characters.
/// * `2` for characters that typically occupy two terminal cells.
pub fn get_char_display_width(c: char) -> usize {
    // `get_or_init` ensures that `LocaleInitializer::new` is called only once.
    // It returns a reference to the initialized `LocaleInitializer`.
    let controller = GLOBAL_LOCALE_INITIALIZER.get_or_init(LocaleInitializer::new);
    controller.char_display_width_internal(c)
}

// Add this to the end of jppittman/core-term/core-term-9499c07698dabcac168e7b529a732a831c502d2f/src/term/unicode.rs

#[cfg(test)]
mod tests {
    use super::*; // Imports items from the outer module (unicode.rs)
    use test_log::test; // Ensure test_log is a dev-dependency for log capturing in tests

    // Helper to ensure the global locale initializer is called at least once before tests
    // that depend on get_char_display_width.
    fn ensure_locale_initialized() {
        let _ = get_char_display_width(' '); // Initialize with a benign character
    }

    #[test]
    fn test_ascii_char_width() {
        ensure_locale_initialized();
        assert_eq!(get_char_display_width('A'), 1, "Width of 'A' should be 1");
        assert_eq!(get_char_display_width(' '), 1, "Width of space should be 1");
        assert_eq!(get_char_display_width('~'), 1, "Width of '~' should be 1");
    }

    #[test]
    fn test_box_drawing_char_widths() {
        ensure_locale_initialized();
        // Test a representative set of box-drawing characters
        assert_eq!(
            get_char_display_width('─'),
            1,
            "Width of U+2500 BOX DRAWINGS LIGHT HORIZONTAL"
        );
        assert_eq!(
            get_char_display_width('│'),
            1,
            "Width of U+2502 BOX DRAWINGS LIGHT VERTICAL"
        );
        assert_eq!(
            get_char_display_width('┌'),
            1,
            "Width of U+250C BOX DRAWINGS LIGHT DOWN AND RIGHT"
        );
        assert_eq!(
            get_char_display_width('┐'),
            1,
            "Width of U+2510 BOX DRAWINGS LIGHT DOWN AND LEFT"
        );
        assert_eq!(
            get_char_display_width('└'),
            1,
            "Width of U+2514 BOX DRAWINGS LIGHT UP AND RIGHT"
        );
        assert_eq!(
            get_char_display_width('┘'),
            1,
            "Width of U+2518 BOX DRAWINGS LIGHT UP AND LEFT"
        );
        assert_eq!(
            get_char_display_width('├'),
            1,
            "Width of U+251C BOX DRAWINGS LIGHT VERTICAL AND RIGHT"
        );
        assert_eq!(
            get_char_display_width('┤'),
            1,
            "Width of U+2524 BOX DRAWINGS LIGHT VERTICAL AND LEFT"
        );
        assert_eq!(
            get_char_display_width('┬'),
            1,
            "Width of U+252C BOX DRAWINGS LIGHT DOWN AND HORIZONTAL"
        );
        assert_eq!(
            get_char_display_width('┴'),
            1,
            "Width of U+2534 BOX DRAWINGS LIGHT UP AND HORIZONTAL"
        );
        assert_eq!(
            get_char_display_width('┼'),
            1,
            "Width of U+253C BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL"
        );
        // Add more box characters if you suspect issues with specific ones
    }

    #[test]
    fn test_cjk_wide_char_widths() {
        ensure_locale_initialized();
        assert_eq!(
            get_char_display_width('世'),
            2,
            "Width of '世' (U+4E16) should be 2"
        );
        assert_eq!(
            get_char_display_width('界'),
            2,
            "Width of '界' (U+754C) should be 2"
        );
        assert_eq!(
            get_char_display_width('你'),
            2,
            "Width of '你' (U+4F60) should be 2"
        );
        assert_eq!(
            get_char_display_width('好'),
            2,
            "Width of '好' (U+597D) should be 2"
        );
    }

    #[test]
    fn test_control_char_widths() {
        ensure_locale_initialized();
        // C0 control characters
        assert_eq!(
            get_char_display_width('\u{0000}'),
            0,
            "Width of NUL (U+0000) should be 0"
        );
        assert_eq!(
            get_char_display_width('\u{0007}'),
            0,
            "Width of BEL (U+0007) should be 0"
        );
        assert_eq!(
            get_char_display_width('\u{001B}'),
            0,
            "Width of ESC (U+001B) should be 0"
        );
        // Check a C1 control character (e.g., IND - Index U+0084)
        // Rust's char::is_control should cover C1 too.
        assert_eq!(
            get_char_display_width('\u{0084}'),
            0,
            "Width of IND (U+0084) should be 0"
        );
    }

    #[test]
    fn test_zero_width_chars() {
        ensure_locale_initialized();
        // Example: Zero Width Joiner (U+200D). wcwidth often returns 0.
        assert_eq!(
            get_char_display_width('\u{200D}'),
            0,
            "Width of ZWJ (U+200D) should be 0"
        );
        // Example: Combining Acute Accent (U+0301) when treated as standalone.
        // Note: wcwidth behavior for combining marks can vary; often 0.
        assert_eq!(
            get_char_display_width('\u{0301}'),
            0,
            "Width of Combining Acute Accent (U+0301) should be 0"
        );
    }

    #[test]
    fn test_locale_initializer_called() {
        // This test primarily ensures that the OnceLock mechanism is engaged.
        // It doesn't directly assert the success of setlocale here, as that's
        // better observed by the behavior of get_char_display_width in other tests
        // and by checking logs for the warn/debug messages from LocaleInitializer::new().
        ensure_locale_initialized();
        assert!(
            GLOBAL_LOCALE_INITIALIZER.get().is_some(),
            "LocaleInitializer should be initialized after first call to get_char_display_width."
        );

        // You should also manually check your logs when running tests.
        // If you see "Failed to set LC_CTYPE locale from environment...",
        // it indicates a problem with setlocale. Otherwise, you should see
        // "LC_CTYPE locale set successfully...".
    }
}
