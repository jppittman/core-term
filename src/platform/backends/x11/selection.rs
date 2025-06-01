// src/platform/backends/x11/selection.rs

//! Handles X11 selection mechanisms (PRIMARY, CLIPBOARD), primarily by managing
//! the required X Atoms.

#![allow(non_snake_case)] // For X11 types

use super::connection::Connection; // To access XInternAtom
use x11::xlib;
use libc::c_char;
use anyhow::{Result, Context}; // For error handling in `new`

/// Holds commonly used X11 atoms for selection handling.
#[derive(Debug, Clone, Copy)]
pub struct SelectionAtoms {
    // Selection names
    pub primary: xlib::Atom,
    pub clipboard: xlib::Atom,

    // Target types
    pub targets: xlib::Atom,
    pub utf8_string: xlib::Atom,
    pub text: xlib::Atom,        // Older, less specific text type
    pub compound_text: xlib::Atom, // Another text representation

    // Add other atoms as needed, e.g., INCR for incremental transfers
}

impl SelectionAtoms {
    /// Interns the necessary X11 atoms for selection handling.
    ///
    /// # Arguments
    /// * `connection`: A reference to the X11 `Connection` used to intern atoms.
    ///
    /// # Returns
    /// * `Result<Self>`: The struct containing interned atoms, or an error if any atom
    ///   could not be interned.
    pub fn new(connection: &Connection) -> Result<Self> {
        let display = connection.display();

        // Helper closure to intern an atom and wrap errors.
        let intern = |name: &str| -> Result<xlib::Atom> {
            // SAFETY: `XInternAtom` is an FFI call.
            // `display` must be a valid pointer.
            // `name.as_ptr()` provides a C-string pointer from a Rust string literal (null-terminated).
            // `xlib::False` means create the atom if it doesn't exist.
            // The result is an XID (Atom). 0 (None) indicates failure.
            let atom_name_cstr = std::ffi::CString::new(name)
                .with_context(|| format!("Failed to create CString for atom name '{}'", name))?;
            let atom = unsafe {
                xlib::XInternAtom(display, atom_name_cstr.as_ptr() as *const c_char, xlib::False)
            };
            if atom == xlib::NONE { // xlib::NONE is typically 0
                Err(anyhow::anyhow!("Failed to intern X11 atom: {}", name))
            } else {
                Ok(atom)
            }
        };

        Ok(Self {
            primary: intern("PRIMARY")?,
            clipboard: intern("CLIPBOARD")?,
            targets: intern("TARGETS")?,
            utf8_string: intern("UTF8_STRING")?,
            text: intern("TEXT")?,
            compound_text: intern("COMPOUND_TEXT")?,
        })
    }
}
