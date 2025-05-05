// src/term/screen.rs

//! Handles screen manipulation, drawing, and scrolling logic.

// Add imports for command types needed in helper functions
use crate::ansi::commands::{Attribute, Color as AnsiColor}; // Rename Color to avoid conflict
use super::{Cursor, DecModes, Glyph, Term, DEFAULT_TAB_INTERVAL};
use crate::glyph::{AttrFlags, Color}; // Keep original Color import for Glyph
use log::{debug, trace, warn};
use std::cmp::{max, min};

// --- Constants ---
/// Default count parameter value (e.g., for cursor movement, erase).
const DEFAULT_COUNT: u16 = 1;
/// Erase from cursor to end of line/display.
const ERASE_MODE_TO_END: u16 = 0;
/// Erase from start to cursor (inclusive).
const ERASE_MODE_TO_START: u16 = 1;
/// Erase entire line/display.
const ERASE_MODE_ALL: u16 = 2;
/// Erase scrollback buffer (ED only).
const ERASE_MODE_SCROLLBACK: u16 = 3;

// DEC Private Mode constants used in handle_dec_mode_set
const DECCKM: u16 = 1; // Cursor Keys Mode
const DECOM: u16 = 6; // Origin Mode
const DECTCEM: u16 = 25; // Text Cursor Enable Mode
const ALT_SCREEN_BUF_47: u16 = 47; // Legacy alternate screen buffer
const ALT_SCREEN_BUF_1047: u16 = 1047; // Alternate screen buffer
const CURSOR_SAVE_RESTORE_1048: u16 = 1048; // Save/restore cursor with alt screen switch
const ALT_SCREEN_SAVE_RESTORE_1049: u16 = 1049; // Save/restore cursor and switch alt screen

// --- Helper Functions ---

/// Gets the effective top boundary for cursor movement and scrolling (respects origin mode).
/// Returns 0-based row index.
pub(super) fn effective_top(term: &Term) -> usize {
    if term.dec_modes.origin_mode {
        return term.scroll_top;
    }
    0
}

/// Gets the effective bottom boundary for cursor movement and scrolling (respects origin mode).
/// Returns 0-based row index.
pub(super) fn effective_bottom(term: &Term) -> usize {
    if term.dec_modes.origin_mode {
        return term.scroll_bot;
    }
    term.height.saturating_sub(1)
}

/// Fills a range of cells in a given row with spaces using default attributes
/// and the current background color. Used for erase operations.
/// `x_end` is exclusive.
pub(super) fn fill_range(term: &mut Term, y: usize, x_start: usize, x_end: usize) {
    let current_screen = if term.using_alt_screen {
        &mut term.alt_screen
    } else {
        &mut term.screen
    };
    if let Some(row) = current_screen.get_mut(y) {
        // Use default attributes but keep the current background color for erase
        let fill_attr = crate::glyph::Attributes {
            fg: term.default_attributes.fg,
            bg: term.current_attributes.bg, // Use current BG for erase
            flags: term.default_attributes.flags,
        };
        let fill_glyph = Glyph {
            c: ' ',
            attr: fill_attr,
        };

        let start = min(x_start, term.width);
        let end = min(x_end, term.width);

        if start < end {
            row[start..end].fill(fill_glyph);
            term.dirty[y] = 1; // Mark line dirty
        }
    }
}

// --- Cursor Movement ---

/// Moves the cursor relative to its current position, clamping to boundaries.
/// Respects origin mode via effective_top/effective_bottom. Resets wrap_next.
pub(super) fn move_cursor(term: &mut Term, dx: isize, dy: isize) {
    let top = effective_top(term);
    let bot = effective_bottom(term);
    let max_x = term.width.saturating_sub(1);

    // Calculate target Y first, clamping within the effective region
    let target_y = if dy >= 0 {
        term.cursor.y.saturating_add(dy as usize)
    } else {
        term.cursor.y.saturating_sub((-dy) as usize)
    };
    let final_y = target_y.clamp(top, bot);

    // Calculate target X, clamping within screen width
    let target_x = if dx >= 0 {
        term.cursor.x.saturating_add(dx as usize)
    } else {
        term.cursor.x.saturating_sub((-dx) as usize)
    };
    let final_x = min(target_x, max_x);

    term.cursor.x = final_x;
    term.cursor.y = final_y;
    term.wrap_next = false; // Explicit movement resets wrap

    trace!(
        "move_cursor: dx={}, dy={}, new_pos=({}, {})",
        dx,
        dy,
        term.cursor.x,
        term.cursor.y
    );
}

/// Sets the cursor position absolutely (using 1-based coordinates from sequence).
/// Respects origin mode via effective_top/effective_bottom. Resets wrap_next.
pub(super) fn set_cursor_pos(term: &mut Term, col_req: u16, row_req: u16) {
    let top = effective_top(term);
    let bot = effective_bottom(term);
    let max_x = term.width.saturating_sub(1);

    // Convert 1-based request to 0-based target, defaulting to 0 if request is 0
    let target_row_0based = row_req.saturating_sub(1) as usize;
    let target_col_0based = col_req.saturating_sub(1) as usize;

    // Apply origin mode offset if active
    let final_y = if term.dec_modes.origin_mode {
        top + target_row_0based
    } else {
        target_row_0based
    };

    // Clamp final position
    term.cursor.x = min(target_col_0based, max_x);
    term.cursor.y = final_y.clamp(top, bot);
    term.wrap_next = false; // Explicit positioning resets wrap

    trace!(
        "set_cursor_pos: req=(col={}, row={}), target_0based=(col={}, row={}), final=({}, {})",
        col_req,
        row_req,
        target_col_0based,
        target_row_0based,
        term.cursor.x,
        term.cursor.y
    );
}

/// Saves the current cursor position and attributes (DECSC / SCOSC).
pub(super) fn save_cursor(term: &mut Term) {
    trace!("Saving cursor: ({}, {})", term.cursor.x, term.cursor.y);
    let cursor_to_save = term.cursor;
    let attributes_to_save = term.current_attributes;

    if term.using_alt_screen {
        term.saved_cursor_alt = cursor_to_save;
        term.saved_attributes_alt = attributes_to_save;
    } else {
        term.saved_cursor = cursor_to_save;
        term.saved_attributes = attributes_to_save;
    }
}

/// Restores the saved cursor position and attributes (DECRC / SCORC).
/// Resets wrap_next.
pub(super) fn restore_cursor(term: &mut Term) {
    let (cursor_to_restore, attributes_to_restore) = if term.using_alt_screen {
        (term.saved_cursor_alt, term.saved_attributes_alt)
    } else {
        (term.saved_cursor, term.saved_attributes)
    };

    term.cursor = cursor_to_restore;
    term.current_attributes = attributes_to_restore;

    // Clamp restored cursor position to current screen dimensions
    let max_x = term.width.saturating_sub(1);
    let max_y = term.height.saturating_sub(1);
    term.cursor.x = min(term.cursor.x, max_x);
    term.cursor.y = min(term.cursor.y, max_y);
    term.wrap_next = false; // Restore resets wrap
    trace!("Restored cursor: ({}, {})", term.cursor.x, term.cursor.y);
}

// --- Screen Content Manipulation ---

/// Handles a printable character: inserts it at cursor, advances cursor (with wrapping).
pub(super) fn handle_printable(term: &mut Term, c: char) {
    trace!(
        "handle_printable: char='{}', cursor=({}, {}), wrap_next={}",
        c,
        term.cursor.x,
        term.cursor.y,
        term.wrap_next
    );
    // TODO(#1): Implement proper character width calculation.
    // let char_width = unicode_width::UnicodeWidthChar::width(c).unwrap_or(1);
    let char_width = 1; // Assume width 1 for now

    let mut target_x = term.cursor.x;
    let mut target_y = term.cursor.y;

    // Handle wrap_next *before* calculating position if it was set by the previous character
    if term.wrap_next {
        target_x = 0;
        // Call index to potentially scroll and move cursor down
        index(term);
        target_y = term.cursor.y; // Update target_y after index()
        term.wrap_next = false; // Consume the wrap_next flag
        trace!("  -> Applied wrap_next, new target=({}, {})", target_x, target_y);
    }

    // Place the character
    let current_screen = if term.using_alt_screen {
        &mut term.alt_screen
    } else {
        &mut term.screen
    };
    if let Some(row) = current_screen.get_mut(target_y) {
        if let Some(cell) = row.get_mut(target_x) {
            cell.c = c;
            cell.attr = term.current_attributes;
            // Clear wide/dummy flags if we overwrite part of a wide char
            cell.attr.flags &= !(AttrFlags::WIDE | AttrFlags::WIDE_DUMMY);
            term.dirty[target_y] = 1; // Mark line dirty
            trace!("  -> Placed char '{}' at ({}, {})", c, target_x, target_y);
        }
        // TODO(#1): Handle wide characters properly (setting WIDE and WIDE_DUMMY flags)
        // if char_width > 1 { ... }
    }

    // Update cursor position and potentially set wrap_next for the *next* character
    let next_cursor_x = target_x.saturating_add(char_width);
    if next_cursor_x >= term.width {
        // We reached or went past the last column
        term.cursor.x = term.width.saturating_sub(1); // Position cursor *at* the end for wrap_next logic
        term.wrap_next = true;
        trace!("  -> Set wrap_next=true, cursor.x={}", term.cursor.x);
    } else {
        term.cursor.x = next_cursor_x;
        term.wrap_next = false; // Not at the end, so no wrap needed next time
        trace!("  -> Moved cursor to ({}, {})", term.cursor.x, term.cursor.y);
    }
}

/// Moves cursor down one line, scrolling if necessary (IND - Index).
pub(super) fn index(term: &mut Term) {
    trace!("index: cursor=({}, {})", term.cursor.x, term.cursor.y);
    // Scrolling for IND/NEL happens when cursor is at the defined bottom margin,
    // regardless of origin mode. Compare with scroll_bot, not effective_bottom.
    if term.cursor.y == term.scroll_bot {
        scroll_up(term, 1);
    } else {
        // Move down, clamped to screen height (not scroll region)
        term.cursor.y = term.cursor.y.saturating_add(1);
        term.cursor.y = min(term.cursor.y, term.height.saturating_sub(1));
    }
    // Index does NOT reset wrap_next (unlike explicit cursor moves)
}

/// Moves cursor up one line, scrolling if necessary (RI - Reverse Index).
pub(super) fn reverse_index(term: &mut Term) {
    trace!("reverse_index: cursor=({}, {})", term.cursor.x, term.cursor.y);
    // Scrolling for RI happens when cursor is at the defined top margin,
    // regardless of origin mode. Compare with scroll_top, not effective_top.
    if term.cursor.y == term.scroll_top {
        scroll_down(term, 1);
    } else {
        // Move up (cannot go below 0)
        term.cursor.y = term.cursor.y.saturating_sub(1);
    }
    // Reverse Index does NOT reset wrap_next
}

/// Moves cursor to start of next line (like CR + LF) (NEL - Next Line).
pub(super) fn newline(term: &mut Term) {
    trace!("newline: cursor=({}, {})", term.cursor.x, term.cursor.y);
    index(term);
    carriage_return(term); // carriage_return resets wrap_next
}

/// Moves cursor to column 0 (CR - Carriage Return). Resets wrap_next.
pub(super) fn carriage_return(term: &mut Term) {
    trace!("carriage_return: cursor=({}, {})", term.cursor.x, term.cursor.y);
    term.cursor.x = 0;
    term.wrap_next = false; // CR resets wrap
}

/// Moves cursor left one column, stopping at column 0 (BS - Backspace). Resets wrap_next.
pub(super) fn backspace(term: &mut Term) {
    trace!("backspace: cursor=({}, {})", term.cursor.x, term.cursor.y);
    term.cursor.x = term.cursor.x.saturating_sub(1);
    term.wrap_next = false; // BS resets wrap
}

/// Moves cursor to the next tab stop (HT - Horizontal Tabulation). Resets wrap_next.
pub(super) fn tab(term: &mut Term) {
    trace!("tab: cursor=({}, {})", term.cursor.x, term.cursor.y);
    let next_tab_stop = term
        .tabs
        .iter()
        .enumerate()
        .find(|&(i, is_set)| i > term.cursor.x && *is_set);

    if let Some((next_x, _)) = next_tab_stop {
        term.cursor.x = min(next_x, term.width.saturating_sub(1));
    } else {
        // If no more tab stops, move to the last column
        term.cursor.x = term.width.saturating_sub(1);
    }
    term.wrap_next = false; // Tab resets wrap
}

// --- Erasing Functions ---

/// Erases from cursor to end of line (inclusive) (EL 0).
pub(super) fn erase_line_to_end(term: &mut Term) {
    trace!("erase_line_to_end: cursor=({}, {})", term.cursor.x, term.cursor.y);
    fill_range(term, term.cursor.y, term.cursor.x, term.width);
}

/// Erases from start of line to cursor (inclusive) (EL 1).
pub(super) fn erase_line_to_start(term: &mut Term) {
    trace!(
        "erase_line_to_start: cursor=({}, {})",
        term.cursor.x,
        term.cursor.y
    );
    fill_range(term, term.cursor.y, 0, term.cursor.x.saturating_add(1));
}

/// Erases the entire current line (EL 2).
pub(super) fn erase_whole_line(term: &mut Term) {
    trace!(
        "erase_whole_line: cursor=({}, {})",
        term.cursor.x,
        term.cursor.y
    );
    fill_range(term, term.cursor.y, 0, term.width);
}

/// Erases from cursor to end of screen (inclusive) (ED 0).
pub(super) fn erase_display_to_end(term: &mut Term) {
    trace!(
        "erase_display_to_end: cursor=({}, {})",
        term.cursor.x,
        term.cursor.y
    );
    erase_line_to_end(term); // Erase rest of current line
    for y in (term.cursor.y + 1)..term.height {
        fill_range(term, y, 0, term.width); // Erase lines below
    }
}

/// Erases from start of screen to cursor (inclusive) (ED 1).
pub(super) fn erase_display_to_start(term: &mut Term) {
    trace!(
        "erase_display_to_start: cursor=({}, {})",
        term.cursor.x,
        term.cursor.y
    );
    for y in 0..term.cursor.y {
        fill_range(term, y, 0, term.width); // Erase lines above
    }
    erase_line_to_start(term); // Erase start of current line
}

/// Erases the entire screen (ED 2).
pub(super) fn erase_whole_display(term: &mut Term) {
    trace!("erase_whole_display");
    for y in 0..term.height {
        fill_range(term, y, 0, term.width);
    }
}

/// Handles ED (Erase in Display) sequences.
pub(super) fn handle_erase_in_display(term: &mut Term, mode: u16) {
    match mode {
        ERASE_MODE_TO_END => erase_display_to_end(term),
        ERASE_MODE_TO_START => erase_display_to_start(term),
        ERASE_MODE_ALL => erase_whole_display(term),
        ERASE_MODE_SCROLLBACK => {
            debug!("ED 3 (Erase Scrollback) requested - Not Implemented");
            // TODO: Implement scrollback buffer clearing if scrollback is added
        }
        _ => warn!("Unknown ED parameter: {}", mode),
    }
}

/// Handles EL (Erase in Line) sequences.
pub(super) fn handle_erase_in_line(term: &mut Term, mode: u16) {
    match mode {
        ERASE_MODE_TO_END => erase_line_to_end(term),
        ERASE_MODE_TO_START => erase_line_to_start(term),
        ERASE_MODE_ALL => erase_whole_line(term),
        _ => warn!("Unknown EL parameter: {}", mode),
    }
}

// --- Scrolling ---

/// Scrolls the content within the defined scrolling region up by `n` lines.
/// New lines at the bottom are filled with blanks. (SU)
pub(super) fn scroll_up(term: &mut Term, n: usize) {
    trace!(
        "scroll_up: n={}, region=({}-{})",
        n,
        term.scroll_top,
        term.scroll_bot
    );
    let region_height = term.scroll_bot + 1 - term.scroll_top;
    let n = min(n, region_height);
    if n == 0 {
        return;
    }

    let current_screen = if term.using_alt_screen {
        &mut term.alt_screen
    } else {
        &mut term.screen
    };
    let top = term.scroll_top;
    let bot = term.scroll_bot;

    // Rotate lines within the scroll region
    current_screen[top..=bot].rotate_left(n);

    // Fill the new blank lines at the bottom
    // Use default attributes but current background color
    let fill_attr = crate::glyph::Attributes {
        fg: term.default_attributes.fg,
        bg: term.current_attributes.bg,
        flags: term.default_attributes.flags,
    };
    let fill_glyph = Glyph {
        c: ' ',
        attr: fill_attr,
    };
    let fill_start = bot + 1 - n;
    for y in fill_start..=bot {
        if let Some(row) = current_screen.get_mut(y) {
            row.fill(fill_glyph);
            term.dirty[y] = 1; // Mark line dirty
        }
    }

    // Mark scrolled lines dirty as well
    for y in top..fill_start {
        term.dirty[y] = 1;
    }
}

/// Scrolls the content within the defined scrolling region down by `n` lines.
/// New lines at the top are filled with blanks. (SD)
pub(super) fn scroll_down(term: &mut Term, n: usize) {
    trace!(
        "scroll_down: n={}, region=({}-{})",
        n,
        term.scroll_top,
        term.scroll_bot
    );
    let region_height = term.scroll_bot + 1 - term.scroll_top;
    let n = min(n, region_height);
    if n == 0 {
        return;
    }

    let current_screen = if term.using_alt_screen {
        &mut term.alt_screen
    } else {
        &mut term.screen
    };
    let top = term.scroll_top;
    let bot = term.scroll_bot;

    // Rotate lines within the scroll region
    current_screen[top..=bot].rotate_right(n);

    // Fill the new blank lines at the top
    // Use default attributes but current background color
    let fill_attr = crate::glyph::Attributes {
        fg: term.default_attributes.fg,
        bg: term.current_attributes.bg,
        flags: term.default_attributes.flags,
    };
    let fill_glyph = Glyph {
        c: ' ',
        attr: fill_attr,
    };
    let fill_end = top + n;
    for y in top..fill_end {
        if let Some(row) = current_screen.get_mut(y) {
            row.fill(fill_glyph);
            term.dirty[y] = 1; // Mark line dirty
        }
    }

    // Mark scrolled lines dirty as well
    for y in fill_end..=bot {
        term.dirty[y] = 1;
    }
}

// --- Insertion/Deletion ---

/// Inserts `n` blank lines at the cursor's row, within the scroll region. (IL)
pub(super) fn insert_blank_lines(term: &mut Term, n: usize) {
    trace!(
        "insert_blank_lines: n={}, cursor=({}, {})",
        n,
        term.cursor.x,
        term.cursor.y
    );
    // Check if cursor is within scroll region
    if term.cursor.y < term.scroll_top || term.cursor.y > term.scroll_bot {
        return;
    }
    let n = min(n, term.scroll_bot + 1 - term.cursor.y);
    if n == 0 {
        return;
    }

    scroll_down(term, n); // Use scroll_down for the shift and fill

    // Move cursor to start of line after IL, consistent with some terminals.
    term.cursor.x = 0;
    term.wrap_next = false; // Explicit cursor move resets wrap
}

/// Deletes `n` lines starting at the cursor's row, within the scroll region. (DL)
pub(super) fn delete_lines(term: &mut Term, n: usize) {
    trace!(
        "delete_lines: n={}, cursor=({}, {})",
        n,
        term.cursor.x,
        term.cursor.y
    );
    // Check if cursor is within scroll region
    if term.cursor.y < term.scroll_top || term.cursor.y > term.scroll_bot {
        return;
    }
    let n = min(n, term.scroll_bot + 1 - term.cursor.y);
    if n == 0 {
        return;
    }

    scroll_up(term, n); // Use scroll_up for the shift and fill

    // Move cursor to start of line after DL, consistent with some terminals.
    term.cursor.x = 0;
    term.wrap_next = false; // Explicit cursor move resets wrap
}

/// Inserts `n` blank characters at the cursor position. (ICH)
pub(super) fn insert_blank_chars(term: &mut Term, n: usize) {
    trace!(
        "insert_blank_chars: n={}, cursor=({}, {})",
        n,
        term.cursor.x,
        term.cursor.y
    );
    let n = min(n, term.width.saturating_sub(term.cursor.x));
    if n == 0 {
        return;
    }

    let current_screen = if term.using_alt_screen {
        &mut term.alt_screen
    } else {
        &mut term.screen
    };
    if let Some(row) = current_screen.get_mut(term.cursor.y) {
        row[term.cursor.x..].rotate_right(n);

        // Use default attributes but current background color
        let fill_attr = crate::glyph::Attributes {
            fg: term.default_attributes.fg,
            bg: term.current_attributes.bg,
            flags: term.default_attributes.flags,
        };
        let fill_glyph = Glyph {
            c: ' ',
            attr: fill_attr,
        };
        for x in term.cursor.x..(term.cursor.x + n) {
            if let Some(cell) = row.get_mut(x) {
                *cell = fill_glyph;
            }
        }
        term.dirty[term.cursor.y] = 1; // Mark line dirty
    }
    // Inserting chars does not affect wrap_next state
}

/// Deletes `n` characters starting at the cursor position. (DCH)
pub(super) fn delete_chars(term: &mut Term, n: usize) {
    trace!(
        "delete_chars: n={}, cursor=({}, {})",
        n,
        term.cursor.x,
        term.cursor.y
    );
    let n = min(n, term.width.saturating_sub(term.cursor.x));
    if n == 0 {
        return;
    }

    let current_screen = if term.using_alt_screen {
        &mut term.alt_screen
    } else {
        &mut term.screen
    };
    if let Some(row) = current_screen.get_mut(term.cursor.y) {
        row[term.cursor.x..].rotate_left(n);

        // Use default attributes but current background color
        let fill_attr = crate::glyph::Attributes {
            fg: term.default_attributes.fg,
            bg: term.current_attributes.bg,
            flags: term.default_attributes.flags,
        };
        let fill_glyph = Glyph {
            c: ' ',
            attr: fill_attr,
        };
        let fill_start = term.width.saturating_sub(n);
        for x in fill_start..term.width {
            if let Some(cell) = row.get_mut(x) {
                *cell = fill_glyph;
            }
        }
        term.dirty[term.cursor.y] = 1; // Mark line dirty
    }
    // Deleting chars does not affect wrap_next state
}

// --- Mode Setting ---

/// Sets the scrolling region (DECSTBM). Input is 1-based, stored 0-based.
pub(super) fn set_scrolling_region(term: &mut Term, top_req: usize, bot_req: usize) {
    let top = top_req.saturating_sub(1);
    let bot = bot_req.saturating_sub(1);

    // Use max(top, bot) for the upper bound check against height
    if top < bot && max(top, bot) < term.height {
        term.scroll_top = top;
        term.scroll_bot = bot;
        trace!("Set scrolling region: {}-{}", term.scroll_top, term.scroll_bot);
    } else {
        // If invalid region (top >= bot or bot >= height), reset to full height
        term.scroll_top = 0;
        term.scroll_bot = term.height.saturating_sub(1);
        trace!(
            "Invalid scrolling region ({}-{}), reset to full screen",
            top_req,
            bot_req
        );
    }
    // Move cursor to home position (absolute or relative based on origin mode)
    set_cursor_pos(term, 1, 1);
}

/// Enables origin mode (DECOM). Cursor moves to top-left of scroll region.
pub(super) fn enable_origin_mode(term: &mut Term) {
    term.dec_modes.origin_mode = true;
    // Move cursor to home position (which is now relative to scroll region)
    set_cursor_pos(term, 1, 1); // set_cursor_pos handles clamping and wrap_next
    trace!("Enabled origin mode");
}

/// Disables origin mode (DECOM). Cursor moves to absolute (0,0).
pub(super) fn disable_origin_mode(term: &mut Term) {
    term.dec_modes.origin_mode = false;
    // Move cursor to absolute home (0,0)
    term.cursor = Cursor::default();
    term.wrap_next = false; // Reset wrap on mode change
    trace!("Disabled origin mode");
}

/// Switches to the alternate screen buffer.
pub(super) fn enter_alt_screen(term: &mut Term) {
    if term.using_alt_screen {
        return;
    }

    trace!("Entering alt screen");
    term.using_alt_screen = true;
    // Erase the *alternate* screen upon entry (common behavior)
    // Use fill_range directly to avoid cursor side effects of erase_whole_display
    let fill_attr = term.default_attributes;
    let fill_glyph = Glyph { c: ' ', attr: fill_attr };
    for y in 0..term.height {
        if let Some(row) = term.alt_screen.get_mut(y) {
            row.fill(fill_glyph);
        }
        term.dirty[y] = 1; // Mark lines dirty
    }
    // Move cursor to home on alt screen
    term.cursor = Cursor::default();
    term.wrap_next = false;
}

/// Switches back to the main screen buffer.
pub(super) fn exit_alt_screen(term: &mut Term) {
    if !term.using_alt_screen {
        return;
    }

    trace!("Exiting alt screen");
    term.using_alt_screen = false;
    // Cursor position is implicitly restored by switching back to the main screen's state,
    // unless DECSC/DECRC (1048/1049) was used, which is handled in handle_dec_mode_set.
    // Mark all lines dirty on the main screen to ensure it's fully redrawn.
    term.dirty.fill(1);
}

/// Resets the terminal state to defaults. (RIS)
pub(super) fn reset(term: &mut Term) {
    debug!("Resetting terminal state (RIS)");
    let default_attrs = term.default_attributes;
    let fill_glyph = Glyph {
        c: ' ',
        attr: default_attrs,
    };

    term.cursor = Cursor::default();
    term.wrap_next = false;
    term.current_attributes = default_attrs;
    // term.parser_state = ParserState::Ground; // Handled by AnsiProcessor
    // term.csi_params.clear(); // Handled by AnsiProcessor
    // term.csi_intermediates.clear(); // Handled by AnsiProcessor
    // term.osc_string.clear(); // Handled by AnsiProcessor
    // term.utf8_decoder.reset(); // Handled by AnsiProcessor
    term.dec_modes = DecModes::default();
    term.scroll_top = 0;
    term.scroll_bot = term.height.saturating_sub(1);
    term.using_alt_screen = false; // Ensure we are on the main screen

    term.tabs.fill(false);
    for i in (0..term.width).step_by(DEFAULT_TAB_INTERVAL) {
        if let Some(tab) = term.tabs.get_mut(i) {
            *tab = true;
        }
    }

    // Clear the *main* screen
    for row in term.screen.iter_mut() {
        row.fill(fill_glyph);
    }
    term.dirty.fill(1); // Mark all lines dirty
}

// --- Backend Interaction Placeholders ---

/// Placeholder for setting terminal modes via backend.
#[allow(unused_variables)]
pub(super) fn xsetmode(set: bool, mode: u16) {
    trace!("xsetmode called: set={}, mode={}", set, mode);
    // TODO: Implement backend interaction if needed for modes like cursor visibility
}

// --- Screen Resizing ---

/// Resizes the terminal screen buffers and associated state. Called by Term::resize.
pub(super) fn resize(term: &mut Term, new_width: usize, new_height: usize) {
    trace!("Screen resize logic: new_size={}x{}", new_width, new_height);
    let default_glyph = Glyph {
        c: ' ',
        attr: term.default_attributes,
    };

    // Resize screen buffers
    term.screen
        .resize_with(new_height, || vec![default_glyph; new_width]);
    for row in term.screen.iter_mut() {
        row.resize(new_width, default_glyph);
    }
    term.alt_screen
        .resize_with(new_height, || vec![default_glyph; new_width]);
    for row in term.alt_screen.iter_mut() {
        row.resize(new_width, default_glyph);
    }

    // Resize tabs
    term.tabs.resize(new_width, false);
    // Ensure existing tabs remain, reset tabs in the new area
    for i in 0..new_width {
        if i >= term.width || i % DEFAULT_TAB_INTERVAL == 0 {
            term.tabs[i] = i % DEFAULT_TAB_INTERVAL == 0;
        }
        // Keep existing custom tabs if i < term.width
    }

    // Resize dirty flags
    term.dirty.resize(new_height, 1); // New lines are dirty

    // Clamp cursor and saved positions
    let max_x = new_width.saturating_sub(1);
    let max_y = new_height.saturating_sub(1);
    term.cursor.x = min(term.cursor.x, max_x);
    term.cursor.y = min(term.cursor.y, max_y);
    term.saved_cursor.x = min(term.saved_cursor.x, max_x);
    term.saved_cursor.y = min(term.saved_cursor.y, max_y);
    term.saved_cursor_alt.x = min(term.saved_cursor_alt.x, max_x);
    term.saved_cursor_alt.y = min(term.saved_cursor_alt.y, max_y);

    // Scroll region is reset in Term::resize *before* calling this
}

// --- New Helper Functions for Command Dispatch ---

/// Handles SGR (Select Graphic Rendition) commands.
pub(super) fn handle_sgr(term: &mut Term, attrs: &[Attribute]) {
    if attrs.is_empty() {
        // Equivalent to CSI m -> Reset
        term.current_attributes = term.default_attributes;
        return;
    }
    for attr in attrs {
        match attr {
            Attribute::Reset => term.current_attributes = term.default_attributes,
            Attribute::Bold => term.current_attributes.flags |= AttrFlags::BOLD,
            Attribute::Faint => term.current_attributes.flags |= AttrFlags::FAINT,
            Attribute::Italic => term.current_attributes.flags |= AttrFlags::ITALIC,
            Attribute::Underline => term.current_attributes.flags |= AttrFlags::UNDERLINE,
            Attribute::BlinkSlow | Attribute::BlinkRapid => {
                term.current_attributes.flags |= AttrFlags::BLINK
            }
            Attribute::Reverse => term.current_attributes.flags |= AttrFlags::REVERSE,
            Attribute::Conceal => term.current_attributes.flags |= AttrFlags::HIDDEN,
            Attribute::Strikethrough => term.current_attributes.flags |= AttrFlags::STRIKETHROUGH,
            Attribute::NoBold => term.current_attributes.flags &= !(AttrFlags::BOLD | AttrFlags::FAINT), // Turn off Bold and Faint
            Attribute::NoItalic => term.current_attributes.flags &= !AttrFlags::ITALIC,
            Attribute::NoUnderline => term.current_attributes.flags &= !AttrFlags::UNDERLINE,
            Attribute::NoBlink => term.current_attributes.flags &= !AttrFlags::BLINK,
            Attribute::NoReverse => term.current_attributes.flags &= !AttrFlags::REVERSE,
            Attribute::NoConceal => term.current_attributes.flags &= !AttrFlags::HIDDEN,
            Attribute::NoStrikethrough => term.current_attributes.flags &= !AttrFlags::STRIKETHROUGH,
            Attribute::Foreground(c) => term.current_attributes.fg = map_ansi_color_to_glyph_color(*c),
            Attribute::Background(c) => term.current_attributes.bg = map_ansi_color_to_glyph_color(*c),
            // TODO: Implement Overlined, UnderlineColor, UnderlineDouble if needed
            _ => warn!("Unhandled SGR attribute: {:?}", attr),
        }
    }
}

/// Maps `ansi::commands::Color` to `glyph::Color`.
fn map_ansi_color_to_glyph_color(ansi_color: AnsiColor) -> Color {
    match ansi_color {
        AnsiColor::Black => Color::Idx(0),
        AnsiColor::Red => Color::Idx(1),
        AnsiColor::Green => Color::Idx(2),
        AnsiColor::Yellow => Color::Idx(3),
        AnsiColor::Blue => Color::Idx(4),
        AnsiColor::Magenta => Color::Idx(5),
        AnsiColor::Cyan => Color::Idx(6),
        AnsiColor::White => Color::Idx(7),
        AnsiColor::BrightBlack => Color::Idx(8),
        AnsiColor::BrightRed => Color::Idx(9),
        AnsiColor::BrightGreen => Color::Idx(10),
        AnsiColor::BrightYellow => Color::Idx(11),
        AnsiColor::BrightBlue => Color::Idx(12),
        AnsiColor::BrightMagenta => Color::Idx(13),
        AnsiColor::BrightCyan => Color::Idx(14),
        AnsiColor::BrightWhite => Color::Idx(15),
        AnsiColor::Indexed(idx) => Color::Idx(idx),
        AnsiColor::Rgb(r, g, b) => Color::Rgb(r, g, b),
        AnsiColor::Default => Color::Default,
    }
}

/// Handles OSC (Operating System Command) sequences.
pub(super) fn handle_osc(term: &mut Term, data: &[u8]) {
    // Data includes the parameters separated by ';'. e.g., b"0;Title Text"
    if let Some(separator_pos) = data.iter().position(|&b| b == b';') {
        let code_bytes = &data[..separator_pos];
        let arg_bytes = &data[separator_pos + 1..];

        // Attempt to parse the code as a number
        if let Ok(code_str) = std::str::from_utf8(code_bytes) {
            if let Ok(code) = code_str.parse::<u16>() {
                match code {
                    0 | 2 => { // Set window title
                        let title = String::from_utf8_lossy(arg_bytes);
                        debug!("OSC 0/2: Set Title to '{}' (Backend Action Needed)", title);
                        // TODO: Notify backend to set title
                    }
                    1 => { // Set icon name (often same as title)
                        let title = String::from_utf8_lossy(arg_bytes);
                        debug!("OSC 1: Set Icon Name to '{}' (Backend Action Needed)", title);
                         // TODO: Notify backend to set icon name
                    }
                     // Add cases for other OSC codes like 4 (set color palette), 10/11/12 (get colors), 52 (clipboard)
                    _ => warn!("Unhandled OSC code: {}, arg: {:?}", code, String::from_utf8_lossy(arg_bytes)),
                }
            } else {
                warn!("Failed to parse OSC code as number: {:?}", code_str);
            }
        } else {
            warn!("OSC code is not valid UTF-8: {:?}", code_bytes);
        }
    } else {
        // Handle OSC sequences without a ';', if any are relevant (e.g., legacy 'k')
        warn!("Unhandled OSC sequence without ';': {:?}", String::from_utf8_lossy(data));
    }
}

/// Handles setting or resetting DEC Private Modes.
pub(super) fn handle_dec_mode_set(term: &mut Term, mode: u16, enable: bool) {
    trace!(
        "DEC Mode action: {} mode {}",
        if enable { "Set" } else { "Reset" },
        mode
    );
    match mode {
        DECCKM => term.dec_modes.cursor_keys_app_mode = enable,
        DECOM => {
            if enable {
                enable_origin_mode(term);
            } else {
                disable_origin_mode(term);
            }
        }
        DECTCEM => {
            term.dec_modes.cursor_visible = enable;
            debug!(
                "DECTCEM Action: Visible={} (Backend Action Needed)",
                enable
            );
            // TODO: Notify backend about cursor visibility change
        }
        ALT_SCREEN_BUF_47 | ALT_SCREEN_BUF_1047 => {
            // TODO: Check allowaltscreen config setting
            if enable {
                enter_alt_screen(term);
            } else {
                exit_alt_screen(term);
            }
        }
        CURSOR_SAVE_RESTORE_1048 => {
            if enable {
                save_cursor(term);
            } else {
                restore_cursor(term);
            }
        }
        ALT_SCREEN_SAVE_RESTORE_1049 => {
            // TODO: Check allowaltscreen config setting
            if enable {
                save_cursor(term); // Save before switching
                enter_alt_screen(term);
            } else {
                exit_alt_screen(term);
                restore_cursor(term); // Restore after switching back
            }
        }
        // Add other DEC modes as needed
        _ => warn!("Unhandled DEC Private Mode parameter: {}", mode),
    }
}

/// Handles DSR (Device Status Report) requests.
pub(super) fn handle_device_status_report(term: &Term, mode: u16) {
    match mode {
        5 => {
            debug!("DSR 5 (Status Report) received - sending OK");
            // TODO: Need a way to write back to PTY from Term/Backend
            // Example: backend.write_to_pty(b"\x1b[0n")?;
        }
        6 => {
            // CPR - Cursor Position Report
            let report = format!(
                "\x1b[{};{}R",
                term.cursor.y.saturating_add(1), // 1-based row
                term.cursor.x.saturating_add(1) // 1-based col
            );
            debug!("DSR 6 (CPR) received - sending {}", report);
            // TODO: Need a way to write back to PTY from Term/Backend
            // Example: backend.write_to_pty(report.as_bytes())?;
        }
        _ => warn!("Unknown DSR parameter: {}", mode),
    }
}

/// Handles HTS (Horizontal Tabulation Set).
pub(super) fn set_horizontal_tabstop(term: &mut Term) {
     if term.cursor.x < term.width {
         term.tabs[term.cursor.x] = true;
         trace!("Set tab stop at column {}", term.cursor.x);
     }
}

/// Handles TBC (Tabulation Clear).
pub(super) fn handle_clear_tab_stops(term: &mut Term, mode: u16) {
     match mode {
         0 => { // Clear tab stop at current column
             if term.cursor.x < term.width {
                 term.tabs[term.cursor.x] = false;
                 trace!("Cleared tab stop at column {}", term.cursor.x);
             }
         }
         3 => { // Clear all tab stops
             term.tabs.fill(false);
             trace!("Cleared all tab stops");
         }
         _ => warn!("Unknown TBC parameter: {}", mode),
     }
}

/// Handles SCS (Select Character Set).
pub(super) fn handle_select_character_set(term: &mut Term, intermediate: char, final_char: char) {
    // Map ESC ( G, ESC ) G etc. to G0, G1, G2, G3 designation
    let g_set = match intermediate {
        '(' => 0, // G0
        ')' => 1, // G1
        '*' => 2, // G2
        '+' => 3, // G3
        _ => {
            warn!("Invalid SCS intermediate: {}", intermediate);
            return;
        }
    };
    // Map final_char to charset enum if needed
    // For now, just log it
    debug!("SCS: Designate G{} = '{}'", g_set, final_char);
    // TODO: Implement actual charset mapping/translation if needed
}


// Add the module declaration for the tests file
#[cfg(test)]
mod tests;

