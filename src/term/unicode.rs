//! This module handles Unicode character width determination using the system's `wcwidth`
//! function via FFI, and locale initialization via a lazily initialized static controller
//! using `std::sync::OnceLock`.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint}; // For FFI types
use std::sync::OnceLock; // Standard library for one-time initialization
use log::{debug, warn, trace}; // Logging macros

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
            let empty_locale_c_str = CString::new("").expect("CString::new for empty locale should not fail.");
            if setlocale(LC_CTYPE, empty_locale_c_str.as_ptr()).is_null() {
                warn!("Failed to set LC_CTYPE locale from environment. Character width calculations might be incorrect or use a default \"C\"/\"POSIX\" locale behavior.");
            } else {
                debug!("LC_CTYPE locale set successfully from environment settings for wcwidth.");
            }
        }
        LocaleInitializer { _locale_is_set_guard: () }
    }

    /// Internal method to calculate character display width.
    fn char_display_width_internal(&self, c: char) -> usize {
        let wc = c as c_uint;
        // SAFETY: Calling C's wcwidth. Locale is expected to be set by `new()`.
        let width_from_c = unsafe { wcwidth(wc) };

        match width_from_c {
            -1 => {
                if c.is_control() {
                    trace!("wcwidth returned -1 for control char '{}' (U+{:X}), width is 0.", c, wc);
                    0
                } else {
                    trace!("wcwidth returned -1 for char '{}' (U+{:X}), defaulting to width 1.", c, wc);
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

