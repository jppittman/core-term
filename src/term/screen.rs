// src/term/screen.rs

//! Handles screen manipulation, drawing, and scrolling logic.

use super::{Term, Cursor, Glyph, Attributes, DEFAULT_TAB_INTERVAL, ParserState, DecModes};
use crate::glyph::{Color, AttrFlags, REPLACEMENT_CHARACTER}; // Added REPLACEMENT_CHARACTER back
use std::cmp::{min, max};
use log::{trace, debug, warn};

// --- Constants ---
/// Default count parameter value (e.g., for cursor movement, erase).
const DEFAULT_COUNT: u16 = 1;
// Removed unused ERASE_MODE constants

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

/// Fills a range of cells in a given row with spaces using current attributes.
/// Used for erase operations. Marks the line dirty.
/// `x_end` is exclusive.
pub(super) fn fill_range(term: &mut Term, y: usize, x_start: usize, x_end: usize) {
     let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
     if let Some(row) = current_screen.get_mut(y) {
         // Use current attributes for erase, as per standard terminal behavior
         let fill_attr = term.current_attributes;
         let fill_glyph = Glyph { c: ' ', attr: fill_attr };

         let start = min(x_start, term.width);
         let end = min(x_end, term.width);

         if start < end {
             row[start..end].fill(fill_glyph);
             term.mark_dirty(y); // Mark the modified line as dirty
         }
     }
}


// --- Cursor Movement ---

/// Moves the cursor relative to its current position, clamping to boundaries.
/// Respects origin mode via effective_top/effective_bottom. Resets wrap_next and marks cursor dirty.
pub(super) fn move_cursor(term: &mut Term, dx: isize, dy: isize) {
    let top = effective_top(term);
    let bot = effective_bottom(term);
    let max_x = term.width.saturating_sub(1);

    let current_cursor = term.cursor; // Store current cursor

    let target_x = if dx >= 0 {
        term.cursor.x.saturating_add(dx as usize)
    } else {
        term.cursor.x.saturating_sub((-dx) as usize)
    };

    let target_y = if dy >= 0 {
        term.cursor.y.saturating_add(dy as usize)
    } else {
        term.cursor.y.saturating_sub((-dy) as usize)
    };

    term.cursor.x = min(target_x, max_x);
    term.cursor.y = target_y.clamp(top, bot);
    term.wrap_next = false; // Explicit movement resets wrap

    if term.cursor != current_cursor {
        term.cursor_dirty = true;
    }

    trace!("move_cursor: dx={}, dy={}, new_pos=({}, {})", dx, dy, term.cursor.x, term.cursor.y);
}

/// Sets the cursor position absolutely (using 1-based coordinates from sequence).
/// Respects origin mode via effective_top/effective_bottom. Resets wrap_next and marks cursor dirty.
pub(super) fn set_cursor_pos(term: &mut Term, x_req: u16, y_req: u16) {
    let top = effective_top(term);
    let bot = effective_bottom(term);
    let max_x = term.width.saturating_sub(1);
    let current_cursor = term.cursor; // Store current cursor

    // Parameters are 1-based, default to 1 if 0 or missing.
    let target_y_1based = max(DEFAULT_COUNT, y_req);
    let target_x_1based = max(DEFAULT_COUNT, x_req);

    let target_y_0based = target_y_1based.saturating_sub(1) as usize;
    let target_x_0based = target_x_1based.saturating_sub(1) as usize;

    let final_y = if term.dec_modes.origin_mode {
        top + target_y_0based
    } else {
        target_y_0based
    };

    term.cursor.x = min(target_x_0based, max_x);
    term.cursor.y = final_y.clamp(top, bot);
    term.wrap_next = false; // Explicit positioning resets wrap

    if term.cursor != current_cursor {
        term.cursor_dirty = true;
    }

     trace!("set_cursor_pos: req=({}, {}), target_0based=({}, {}), final=({}, {})",
            x_req, y_req, target_x_0based, target_y_0based, term.cursor.x, term.cursor.y);
}

/// Saves the current cursor position and attributes (DECSC / SCOSC).
pub(super) fn save_cursor(term: &mut Term) {
    trace!("Saving cursor: ({}, {})", term.cursor.x, term.cursor.y);
    let cursor_to_save = term.cursor;
    let attributes_to_save = term.current_attributes;

    if term.using_alt_screen {
        term.saved_cursor_alt = cursor_to_save;
        term.saved_attributes_alt = attributes_to_save;
        return;
    }
    term.saved_cursor = cursor_to_save;
    term.saved_attributes = attributes_to_save;
}

/// Restores the saved cursor position and attributes (DECRC / SCORC). Marks cursor dirty.
pub(super) fn restore_cursor(term: &mut Term) {
    let (cursor_to_restore, attributes_to_restore) = if term.using_alt_screen {
        (term.saved_cursor_alt, term.saved_attributes_alt)
    } else {
        (term.saved_cursor, term.saved_attributes)
    };

    term.cursor = cursor_to_restore;
    term.current_attributes = attributes_to_restore;

     let max_x = term.width.saturating_sub(1);
     let max_y = term.height.saturating_sub(1);
     term.cursor.x = min(term.cursor.x, max_x);
     term.cursor.y = min(term.cursor.y, max_y);
     term.wrap_next = false; // Restore resets wrap
     term.cursor_dirty = true; // Mark cursor as dirty after restore
     trace!("Restored cursor: ({}, {})", term.cursor.x, term.cursor.y);
}


// --- Screen Content Manipulation ---

/// Handles a printable character: inserts it at cursor, advances cursor (with wrapping). Marks line dirty.
pub(super) fn handle_printable(term: &mut Term, c: char) {
    trace!("handle_printable: char='{}', cursor=({}, {}), wrap_next={}", c, term.cursor.x, term.cursor.y, term.wrap_next);

    // TODO(#1): Implement proper character width calculation.
    let char_width = 1; // Assuming width 1 for now

    // Handle line wrap if wrap_next was set by the previous character
    if term.wrap_next {
        term.cursor.x = 0;
        index(term); // Moves cursor down, potentially scrolls, marks cursor dirty
        term.wrap_next = false; // Wrap consumed
    }

    // Place the character
    let y = term.cursor.y; // Store y before potential move
    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    if let Some(row) = current_screen.get_mut(y) {
        if let Some(cell) = row.get_mut(term.cursor.x) {
            cell.c = c;
            cell.attr = term.current_attributes;
            term.mark_dirty(y); // Mark the line dirty
        }
        // TODO(#1): Handle wide char dummy logic if char_width > 1
        // If wide char, also mark cell x+1 and potentially mark line dirty if not already
    }

    // Advance cursor or set wrap_next for the *next* character
    if term.cursor.x + char_width >= term.width {
        if term.dec_modes.autowrap { // Only set wrap_next if autowrap is enabled
            term.wrap_next = true;
        } else {
            // If autowrap is off, stay in the last column
            term.cursor.x = term.width.saturating_sub(1);
            term.cursor_dirty = true; // Cursor position changed (or might have)
        }
    } else {
        term.cursor.x = term.cursor.x.saturating_add(char_width);
        term.cursor_dirty = true; // Cursor position changed
    }
}

/// Moves cursor down one line, scrolling if necessary (IND - Index). Marks cursor dirty.
pub(super) fn index(term: &mut Term) {
    trace!("index: cursor=({}, {})", term.cursor.x, term.cursor.y);
    let current_y = term.cursor.y;
    // Scrolling for IND/NEL happens when cursor is at the defined bottom margin
    if term.cursor.y == term.scroll_bot {
        scroll_up(term, 1); // scroll_up handles marking lines dirty
    } else {
        // Move down, clamped to screen height
        term.cursor.y = term.cursor.y.saturating_add(1);
        term.cursor.y = min(term.cursor.y, term.height.saturating_sub(1));
    }
    if term.cursor.y != current_y {
        term.cursor_dirty = true;
    }
    term.wrap_next = false; // Vertical movement resets wrap
}

/// Moves cursor up one line, scrolling if necessary (RI - Reverse Index). Marks cursor dirty.
pub(super) fn reverse_index(term: &mut Term) {
     trace!("reverse_index: cursor=({}, {})", term.cursor.x, term.cursor.y);
     let current_y = term.cursor.y;
     // Scrolling for RI happens when cursor is at the defined top margin
    if term.cursor.y == term.scroll_top {
        scroll_down(term, 1); // scroll_down handles marking lines dirty
    } else {
        // Move up (cannot go below 0)
        term.cursor.y = term.cursor.y.saturating_sub(1);
    }
    if term.cursor.y != current_y {
        term.cursor_dirty = true;
    }
    term.wrap_next = false; // Vertical movement resets wrap
}

/// Moves cursor to start of next line (like CR + LF) (NEL - Next Line). Marks cursor dirty.
pub(super) fn newline(term: &mut Term) {
    trace!("newline: cursor=({}, {})", term.cursor.x, term.cursor.y);
    let current_cursor = term.cursor;
    index(term);
    carriage_return(term);
    if term.cursor != current_cursor {
        term.cursor_dirty = true;
    }
    // wrap_next is reset by index() and carriage_return()
}

/// Moves cursor to column 0 (CR - Carriage Return). Marks cursor dirty.
pub(super) fn carriage_return(term: &mut Term) {
     trace!("carriage_return: cursor=({}, {})", term.cursor.x, term.cursor.y);
    if term.cursor.x != 0 {
        term.cursor.x = 0;
        term.cursor_dirty = true;
    }
    term.wrap_next = false; // CR resets wrap
}

/// Moves cursor left one column, stopping at column 0 (BS - Backspace). Marks cursor dirty.
pub(super) fn backspace(term: &mut Term) {
     trace!("backspace: cursor=({}, {})", term.cursor.x, term.cursor.y);
     let current_x = term.cursor.x;
    if term.cursor.x > 0 {
        term.cursor.x = term.cursor.x.saturating_sub(1);
    }
    if term.cursor.x != current_x {
        term.cursor_dirty = true;
    }
    term.wrap_next = false; // BS resets wrap
}

/// Moves cursor to the next tab stop (HT - Horizontal Tabulation). Marks cursor dirty.
pub(super) fn tab(term: &mut Term) {
     trace!("tab: cursor=({}, {})", term.cursor.x, term.cursor.y);
     let current_x = term.cursor.x;
     let max_x = term.width.saturating_sub(1);
     let next_tab_stop = term.tabs.iter()
         .enumerate()
         .find(|&(i, is_set)| i > term.cursor.x && *is_set);

     if let Some((next_x, _)) = next_tab_stop {
         term.cursor.x = min(next_x, max_x);
     } else {
         // If no more tabs, move to the last column
         term.cursor.x = max_x;
     }
     if term.cursor.x != current_x {
        term.cursor_dirty = true;
     }
     term.wrap_next = false; // Tab resets wrap
}

// --- Erasing Functions ---

/// Erases from cursor to end of line (inclusive) (EL 0). Marks line dirty.
pub(super) fn erase_line_to_end(term: &mut Term) {
     trace!("erase_line_to_end: cursor=({}, {})", term.cursor.x, term.cursor.y);
    fill_range(term, term.cursor.y, term.cursor.x, term.width); // fill_range marks dirty
}

/// Erases from start of line to cursor (inclusive) (EL 1). Marks line dirty.
pub(super) fn erase_line_to_start(term: &mut Term) {
     trace!("erase_line_to_start: cursor=({}, {})", term.cursor.x, term.cursor.y);
    fill_range(term, term.cursor.y, 0, term.cursor.x.saturating_add(1)); // fill_range marks dirty
}

/// Erases the entire current line (EL 2). Marks line dirty.
pub(super) fn erase_whole_line(term: &mut Term) {
     trace!("erase_whole_line: cursor=({}, {})", term.cursor.x, term.cursor.y);
    fill_range(term, term.cursor.y, 0, term.width); // fill_range marks dirty
}

/// Erases from cursor to end of screen (inclusive) (ED 0). Marks lines dirty.
pub(super) fn erase_display_to_end(term: &mut Term) {
     trace!("erase_display_to_end: cursor=({}, {})", term.cursor.x, term.cursor.y);
    erase_line_to_end(term); // Marks current line dirty
    if term.cursor.y + 1 < term.height {
        term.mark_dirty_range(term.cursor.y + 1, term.height - 1); // Mark range below
        for y in (term.cursor.y + 1)..term.height {
            fill_range(term, y, 0, term.width); // fill_range redundant mark, but ok
        }
    }
}

/// Erases from start of screen to cursor (inclusive) (ED 1). Marks lines dirty.
pub(super) fn erase_display_to_start(term: &mut Term) {
     trace!("erase_display_to_start: cursor=({}, {})", term.cursor.x, term.cursor.y);
     if term.cursor.y > 0 {
         term.mark_dirty_range(0, term.cursor.y - 1); // Mark range above
        for y in 0..term.cursor.y {
            fill_range(term, y, 0, term.width); // fill_range redundant mark, but ok
        }
     }
    erase_line_to_start(term); // Marks current line dirty
}

/// Erases the entire screen (ED 2). Marks all lines dirty.
pub(super) fn erase_whole_display(term: &mut Term) {
     trace!("erase_whole_display");
     term.mark_all_dirty(); // Mark all lines
     for y in 0..term.height {
        fill_range(term, y, 0, term.width); // fill_range redundant mark, but ok
    }
}

// --- Scrolling ---

/// Scrolls the content within the defined scrolling region up by `n` lines.
/// New lines at the bottom are filled with blanks. Marks scrolled region dirty. (SU)
pub(super) fn scroll_up(term: &mut Term, n: usize) {
    trace!("scroll_up: n={}, region=({}-{})", n, term.scroll_top, term.scroll_bot);
    let region_height = term.scroll_bot.saturating_add(1).saturating_sub(term.scroll_top);
    let n = min(n, region_height);
    if n == 0 || region_height == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    let top = term.scroll_top;
    let bot = term.scroll_bot;

    current_screen[top..=bot].rotate_left(n);

    // Fill the newly opened lines at the bottom of the region
    let fill_glyph = Glyph { c: ' ', attr: term.default_attributes }; // Use default for new lines
    let fill_start_y = bot.saturating_add(1).saturating_sub(n);
    for y in fill_start_y..=bot {
        if let Some(row) = current_screen.get_mut(y) {
            row.fill(fill_glyph);
        }
    }
    term.mark_dirty_range(top, bot); // Mark the entire scrolled region dirty
}

/// Scrolls the content within the defined scrolling region down by `n` lines.
/// New lines at the top are filled with blanks. Marks scrolled region dirty. (SD)
pub(super) fn scroll_down(term: &mut Term, n: usize) {
     trace!("scroll_down: n={}, region=({}-{})", n, term.scroll_top, term.scroll_bot);
    let region_height = term.scroll_bot.saturating_add(1).saturating_sub(term.scroll_top);
    let n = min(n, region_height);
    if n == 0 || region_height == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    let top = term.scroll_top;
    let bot = term.scroll_bot;

    current_screen[top..=bot].rotate_right(n);

    // Fill the newly opened lines at the top of the region
    let fill_glyph = Glyph { c: ' ', attr: term.default_attributes }; // Use default for new lines
    for y in top..(top + n) {
        if let Some(row) = current_screen.get_mut(y) {
            row.fill(fill_glyph);
        }
    }
    term.mark_dirty_range(top, bot); // Mark the entire scrolled region dirty
}

// --- Insertion/Deletion ---

/// Inserts `n` blank lines at the cursor's row, within the scroll region. Marks lines dirty. (IL)
pub(super) fn insert_blank_lines(term: &mut Term, n: usize) {
     trace!("insert_blank_lines: n={}, cursor=({}, {})", n, term.cursor.x, term.cursor.y);
     // Check if cursor is within scroll region
     if term.cursor.y < term.scroll_top || term.cursor.y > term.scroll_bot {
        return;
    }
    let n = min(n, term.scroll_bot.saturating_add(1).saturating_sub(term.cursor.y));
    if n == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    let top = term.cursor.y;
    let bot = term.scroll_bot;

    // Manual copy down: Move lines [top..=bot-n] down to [top+n..=bot]
    for i in (top..=bot.saturating_sub(n)).rev() {
        current_screen[i+n] = current_screen[i].clone();
    }

    // Fill the newly opened lines at the top with blanks
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attributes
    for y in top..(top + n) {
         if let Some(row) = current_screen.get_mut(y) {
            row.fill(fill_glyph.clone()); // Use clone if Glyph isn't Copy
         }
    }
    term.mark_dirty_range(top, bot); // Mark affected region dirty
     // Move cursor to start of line after IL
     term.cursor.x = 0;
     term.wrap_next = false;
     term.cursor_dirty = true;
}

/// Deletes `n` lines starting at the cursor's row, within the scroll region. Marks lines dirty. (DL)
pub(super) fn delete_lines(term: &mut Term, n: usize) {
     trace!("delete_lines: n={}, cursor=({}, {})", n, term.cursor.x, term.cursor.y);
     // Check if cursor is within scroll region
     if term.cursor.y < term.scroll_top || term.cursor.y > term.scroll_bot {
        return;
    }
    let n = min(n, term.scroll_bot.saturating_add(1).saturating_sub(term.cursor.y));
    if n == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    let top = term.cursor.y;
    let bot = term.scroll_bot;

    // Manual copy up: Move lines [top+n..=bot] up to [top..=bot-n]
    for i in top..=(bot.saturating_sub(n)) {
        current_screen[i] = current_screen[i+n].clone();
    }

    // Fill the new blank lines at the bottom with blanks
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attributes
    for y in (bot.saturating_add(1).saturating_sub(n))..=bot {
         if let Some(row) = current_screen.get_mut(y) {
            row.fill(fill_glyph.clone()); // Use clone if Glyph isn't Copy
         }
    }
    term.mark_dirty_range(top, bot); // Mark affected region dirty
     // Move cursor to start of line after DL
     term.cursor.x = 0;
     term.wrap_next = false;
     term.cursor_dirty = true;
}

/// Inserts `n` blank characters at the cursor position. Marks line dirty. (ICH)
pub(super) fn insert_blank_chars(term: &mut Term, n: usize) {
    trace!("insert_blank_chars: n={}, cursor=({}, {})", n, term.cursor.x, term.cursor.y);
    let n = min(n, term.width.saturating_sub(term.cursor.x));
    if n == 0 { return; }
    let y = term.cursor.y;

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    if let Some(row) = current_screen.get_mut(y) {
        row[term.cursor.x..].rotate_right(n);

        let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attributes
        for x in term.cursor.x..(term.cursor.x + n) {
             if let Some(cell) = row.get_mut(x) {
                 *cell = fill_glyph;
             }
        }
        term.mark_dirty(y); // Mark line dirty
    }
    term.wrap_next = false; // Insertion resets wrap
    term.cursor_dirty = true; // Cursor position didn't change, but content under it did
}

/// Deletes `n` characters starting at the cursor position. Marks line dirty. (DCH)
pub(super) fn delete_chars(term: &mut Term, n: usize) {
     trace!("delete_chars: n={}, cursor=({}, {})", n, term.cursor.x, term.cursor.y);
    let n = min(n, term.width.saturating_sub(term.cursor.x));
    if n == 0 { return; }
    let y = term.cursor.y;

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    if let Some(row) = current_screen.get_mut(y) {
        row[term.cursor.x..].rotate_left(n);

        let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attributes
        let fill_start = term.width.saturating_sub(n);
        for x in fill_start..term.width {
             if let Some(cell) = row.get_mut(x) {
                 *cell = fill_glyph;
             }
        }
        term.mark_dirty(y); // Mark line dirty
    }
    term.wrap_next = false; // Deletion resets wrap
    term.cursor_dirty = true; // Cursor position didn't change, but content under/after it did
}

// --- Mode Setting ---

/// Sets the scrolling region (DECSTBM). Input is 1-based, stored 0-based. Marks cursor dirty.
pub(super) fn set_scrolling_region(term: &mut Term, top_req: usize, bot_req: usize) {
     let top = top_req.saturating_sub(1);
     let mut bot = bot_req.saturating_sub(1);

     // Ensure bot is within bounds and >= top
     bot = min(bot, term.height.saturating_sub(1));
     if top >= bot {
         // Invalid region, reset to full screen
         term.scroll_top = 0;
         term.scroll_bot = term.height.saturating_sub(1);
         trace!("Invalid scrolling region ({}-{}), reset to full screen", top_req, bot_req);
     } else {
         term.scroll_top = top;
         term.scroll_bot = bot;
         trace!("Set scrolling region: {}-{}", term.scroll_top, term.scroll_bot);
     }
     // Move cursor to home (absolute 0,0 or relative 0,0 depending on origin mode)
     set_cursor_pos(term, 1, 1); // This marks cursor dirty
}

/// Enables origin mode (DECOM). Cursor moves to top-left of scroll region. Marks cursor dirty.
pub(super) fn enable_origin_mode(term: &mut Term) {
    term.dec_modes.origin_mode = true;
    set_cursor_pos(term, 1, 1); // Moves relative to new origin, marks cursor dirty
    trace!("Enabled origin mode");
}

/// Disables origin mode (DECOM). Cursor moves to absolute (0,0). Marks cursor dirty.
pub(super) fn disable_origin_mode(term: &mut Term) {
    term.dec_modes.origin_mode = false;
    term.cursor = Cursor::default(); // Move to absolute 0,0
    term.wrap_next = false;
    term.cursor_dirty = true; // Mark cursor dirty
    trace!("Disabled origin mode");
}

/// Switches to the alternate screen buffer. Marks all lines dirty.
pub(super) fn enter_alt_screen(term: &mut Term) {
    if term.using_alt_screen { return; }

    trace!("Entering alt screen");
    term.using_alt_screen = true;
    erase_whole_display(term); // Erase the *alternate* screen, marks all lines dirty
    term.cursor = Cursor::default(); // Move cursor home on alt screen
    term.wrap_next = false;
    term.cursor_dirty = true;
}

/// Switches back to the main screen buffer. Marks all lines dirty.
pub(super) fn exit_alt_screen(term: &mut Term) {
    if !term.using_alt_screen { return; }

    trace!("Exiting alt screen");
    term.using_alt_screen = false;
    // Mark all lines dirty on the main screen as it's now visible
    term.mark_all_dirty();
    // Cursor position is typically restored by DECRC/SCORC if used with 1049
    // but we mark it dirty just in case.
    term.cursor_dirty = true;
}

/// Resets the terminal state to defaults. (RIS). Marks all lines dirty.
pub(super) fn reset(term: &mut Term) {
     debug!("Resetting terminal state (RIS)");
     let default_attrs = term.default_attributes;
     let fill_glyph = Glyph { c: ' ', attr: default_attrs };

     term.cursor = Cursor::default();
     term.wrap_next = false;
     term.current_attributes = default_attrs;
     term.parser_state = ParserState::Ground;
     term.csi_params.clear();
     term.csi_intermediates.clear();
     term.osc_string.clear();
     term.utf8_decoder.reset();
     term.dec_modes = DecModes { autowrap: true, ..Default::default() }; // Reset modes
     term.scroll_top = 0;
     term.scroll_bot = term.height.saturating_sub(1);
     term.using_alt_screen = false;

     // Reset saved cursors as well
     term.saved_cursor = Cursor::default();
     term.saved_attributes = default_attrs;
     term.saved_cursor_alt = Cursor::default();
     term.saved_attributes_alt = default_attrs;


     term.tabs.fill(false);
     for i in (0..term.width).step_by(DEFAULT_TAB_INTERVAL) {
         if let Some(tab) = term.tabs.get_mut(i) {
             *tab = true;
         }
     }

     for row in term.screen.iter_mut() {
         row.fill(fill_glyph);
     }
     for row in term.alt_screen.iter_mut() {
        row.fill(fill_glyph);
    }
    term.mark_all_dirty(); // Mark all lines dirty after reset
}

// --- Screen Resizing ---

/// Resizes the terminal screen buffers and associated state. Called by Term::resize.
/// Note: Marking dirty is handled by Term::resize itself.
pub(super) fn resize(term: &mut Term, new_width: usize, new_height: usize) {
    trace!("Screen resize logic: new_size={}x{}", new_width, new_height);
    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };

    // Resize screen buffers, filling new cells with default glyph
    term.screen.resize_with(new_height, || vec![default_glyph; new_width]);
    for row in term.screen.iter_mut() {
        row.resize(new_width, default_glyph);
    }
    term.alt_screen.resize_with(new_height, || vec![default_glyph; new_width]);
    for row in term.alt_screen.iter_mut() {
        row.resize(new_width, default_glyph);
    }

    // Resize and reset tabs
    term.tabs.resize(new_width, false);
    for i in 0..new_width {
         term.tabs[i] = i % DEFAULT_TAB_INTERVAL == 0;
    }

    // Clamp cursor and saved cursor positions
    let max_x = new_width.saturating_sub(1);
    let max_y = new_height.saturating_sub(1);
    term.cursor.x = min(term.cursor.x, max_x);
    term.cursor.y = min(term.cursor.y, max_y);
    term.saved_cursor.x = min(term.saved_cursor.x, max_x);
    term.saved_cursor.y = min(term.saved_cursor.y, max_y);
    term.saved_cursor_alt.x = min(term.saved_cursor_alt.x, max_x);
    term.saved_cursor_alt.y = min(term.saved_cursor_alt.y, max_y);

    // Reset wrap flag after resize
    term.wrap_next = false;

    // Note: Scroll region is reset and dirty flags are marked in Term::resize
}


// Add the module declaration for the tests file
#[cfg(test)]
mod tests;
