// src/term/screen.rs

//! Handles screen manipulation, drawing, and scrolling logic.

use super::{Term, Cursor, Glyph, Attributes, DEFAULT_TAB_INTERVAL, ParserState, DecModes};
use crate::glyph::AttrFlags;
use std::cmp::{min, max};
use log::{trace, debug};

// --- Constants ---
/// Default count parameter value (e.g., for cursor movement, erase).
const DEFAULT_COUNT: u16 = 1;
/// Erase from cursor to end of line/display.
#[allow(dead_code)] const ERASE_MODE_TO_END: u16 = 0; // Keep for reference, even if unused now
/// Erase from start to cursor (inclusive).
#[allow(dead_code)] const ERASE_MODE_TO_START: u16 = 1; // Keep for reference, even if unused now
/// Erase entire line/display.
#[allow(dead_code)] const ERASE_MODE_ALL: u16 = 2; // Keep for reference, even if unused now
/// Erase scrollback buffer (ED only).
#[allow(dead_code)] const ERASE_MODE_SCROLLBACK: u16 = 3;

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
     let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
     if let Some(row) = current_screen.get_mut(y) {
         // Use default attributes but keep the current background color for erase
         let fill_attr = Attributes {
             fg: term.default_attributes.fg,
             bg: term.current_attributes.bg, // Use current BG for erase
             flags: term.default_attributes.flags,
         };
         let fill_glyph = Glyph { c: ' ', attr: fill_attr };

         let start = min(x_start, term.width);
         let end = min(x_end, term.width);

         if start < end {
             row[start..end].fill(fill_glyph);
             // TODO: Mark line y dirty
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

    trace!("move_cursor: dx={}, dy={}, new_pos=({}, {})", dx, dy, term.cursor.x, term.cursor.y);
}

/// Sets the cursor position absolutely (using 1-based coordinates from sequence).
/// Respects origin mode via effective_top/effective_bottom. Resets wrap_next.
pub(super) fn set_cursor_pos(term: &mut Term, x_req: u16, y_req: u16) {
    let top = effective_top(term);
    let bot = effective_bottom(term);
    let max_x = term.width.saturating_sub(1);

    let target_y_0based = max(DEFAULT_COUNT, y_req).saturating_sub(1) as usize;
    let target_x_0based = max(DEFAULT_COUNT, x_req).saturating_sub(1) as usize;

    let final_y = if term.dec_modes.origin_mode {
        top + target_y_0based
    } else {
        target_y_0based
    };

    term.cursor.x = min(target_x_0based, max_x);
    term.cursor.y = final_y.clamp(top, bot);
    term.wrap_next = false; // Explicit positioning resets wrap

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
    trace!("handle_printable: char='{}', cursor=({}, {}), wrap_next={}", c, term.cursor.x, term.cursor.y, term.wrap_next);
    // TODO(#1): Implement proper character width calculation.
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
    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    if let Some(row) = current_screen.get_mut(target_y) {
        if let Some(cell) = row.get_mut(target_x) {
            cell.c = c;
            cell.attr = term.current_attributes;
            // Clear wide/dummy flags if we overwrite part of a wide char
            cell.attr.flags &= !(AttrFlags::WIDE | AttrFlags::WIDE_DUMMY);
             trace!("  -> Placed char '{}' at ({}, {})", c, target_x, target_y);
        }
        // TODO(#1): Handle wide characters properly (setting WIDE and WIDE_DUMMY flags)
        // if char_width > 1 { ... }
    }

    // Update cursor position and potentially set wrap_next for the *next* character
    let next_cursor_x = target_x.saturating_add(char_width);
    if next_cursor_x >= term.width {
        // We reached or went past the last column
        term.cursor.x = term.width; // Position cursor *at* the end for wrap_next logic
        term.wrap_next = true;
         trace!("  -> Set wrap_next=true, cursor.x={}", term.cursor.x);
    } else {
        term.cursor.x = next_cursor_x;
        term.wrap_next = false; // Not at the end, so no wrap needed next time
         trace!("  -> Moved cursor to ({}, {})", term.cursor.x, term.cursor.y);
    }

    // TODO(#3): Implement dirty flag logic.
}


/// Moves cursor down one line, scrolling if necessary (IND - Index).
pub(super) fn index(term: &mut Term) {
    trace!("index: cursor=({}, {})", term.cursor.x, term.cursor.y);
    // WHY: Scrolling for IND/NEL happens when cursor is at the defined bottom margin,
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
     // WHY: Scrolling for RI happens when cursor is at the defined top margin,
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
     let next_tab_stop = term.tabs.iter()
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
     trace!("erase_line_to_start: cursor=({}, {})", term.cursor.x, term.cursor.y);
    fill_range(term, term.cursor.y, 0, term.cursor.x.saturating_add(1));
}

/// Erases the entire current line (EL 2).
pub(super) fn erase_whole_line(term: &mut Term) {
     trace!("erase_whole_line: cursor=({}, {})", term.cursor.x, term.cursor.y);
    fill_range(term, term.cursor.y, 0, term.width);
}

/// Erases from cursor to end of screen (inclusive) (ED 0).
pub(super) fn erase_display_to_end(term: &mut Term) {
     trace!("erase_display_to_end: cursor=({}, {})", term.cursor.x, term.cursor.y);
    erase_line_to_end(term);
    for y in (term.cursor.y + 1)..term.height {
        fill_range(term, y, 0, term.width);
    }
}

/// Erases from start of screen to cursor (inclusive) (ED 1).
pub(super) fn erase_display_to_start(term: &mut Term) {
     trace!("erase_display_to_start: cursor=({}, {})", term.cursor.x, term.cursor.y);
    for y in 0..term.cursor.y {
        fill_range(term, y, 0, term.width);
    }
    erase_line_to_start(term);
}

/// Erases the entire screen (ED 2).
pub(super) fn erase_whole_display(term: &mut Term) {
     trace!("erase_whole_display");
    for y in 0..term.height {
        fill_range(term, y, 0, term.width);
    }
}

// --- Scrolling ---

/// Scrolls the content within the defined scrolling region up by `n` lines.
/// New lines at the bottom are filled with blanks. (SU)
pub(super) fn scroll_up(term: &mut Term, n: usize) {
    trace!("scroll_up: n={}, region=({}-{})", n, term.scroll_top, term.scroll_bot);
    let region_height = term.scroll_bot + 1 - term.scroll_top;
    let n = min(n, region_height);
    if n == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    let top = term.scroll_top;
    let bot = term.scroll_bot;

    // Rotate lines within the scroll region
    current_screen[top..=bot].rotate_left(n);

    // Fill the new blank lines at the bottom *only if* n < region_height
    if n < region_height {
        let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attrs
        // The range starts from the position where the last scrolled-out line was,
        // up to the bottom of the region.
        let fill_start = bot + 1 - n;
        for y in fill_start..=bot {
            if let Some(row) = current_screen.get_mut(y) {
                row.fill(fill_glyph);
                // TODO: Mark line y dirty
            }
        }
    } else {
        // If n == region_height, the entire region was effectively cleared by rotation.
        // Mark all lines within the region as dirty.
        for y in top..=bot {
            term.dirty[y] = 1; // Assuming dirty is Vec<u8> or similar
        }
        trace!("scroll_up: Scrolled entire region, marked lines {}-{} dirty", top, bot);
    }
}

/// Scrolls the content within the defined scrolling region down by `n` lines.
/// New lines at the top are filled with blanks. (SD)
pub(super) fn scroll_down(term: &mut Term, n: usize) {
     trace!("scroll_down: n={}, region=({}-{})", n, term.scroll_top, term.scroll_bot);
    let region_height = term.scroll_bot + 1 - term.scroll_top;
    let n = min(n, region_height);
    if n == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    let top = term.scroll_top;
    let bot = term.scroll_bot;

    // Rotate lines within the scroll region
    current_screen[top..=bot].rotate_right(n);

    // Fill the new blank lines at the top *only if* n < region_height
    if n < region_height {
        let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attrs
        for y in top..(top + n) {
             if let Some(row) = current_screen.get_mut(y) {
                row.fill(fill_glyph);
                 // TODO: Mark line y dirty
             }
        }
    } else {
         // If n == region_height, the entire region was effectively cleared by rotation.
         // Mark all lines within the region as dirty.
         for y in top..=bot {
             term.dirty[y] = 1; // Assuming dirty is Vec<u8> or similar
         }
         trace!("scroll_down: Scrolled entire region, marked lines {}-{} dirty", top, bot);
    }
}


// --- Insertion/Deletion ---

/// Inserts `n` blank lines at the cursor's row, within the scroll region. (IL)
/// Replaces rotate_right with manual copy for correctness when n == region_height.
pub(super) fn insert_blank_lines(term: &mut Term, n: usize) {
     trace!("insert_blank_lines: n={}, cursor=({}, {})", n, term.cursor.x, term.cursor.y);
     // Check if cursor is within scroll region
     if term.cursor.y < term.scroll_top || term.cursor.y > term.scroll_bot {
        return;
    }
    let n = min(n, term.scroll_bot + 1 - term.cursor.y);
    if n == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    let top = term.cursor.y;
    let bot = term.scroll_bot;

    // Manual copy down: Move lines [top..=bot-n] down to [top+n..=bot]
    // We iterate downwards to avoid overwriting source lines before they are copied.
    for i in (top..=bot.saturating_sub(n)).rev() {
        // This clone might be inefficient for large screens/regions.
        // A swap-based approach or Vec::spare_capacity_mut could be explored,
        // but clone is simpler and correct for now.
        current_screen[i+n] = current_screen[i].clone();
    }

    // Fill the newly opened lines at the top with blanks
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attrs
    for y in top..(top + n) {
         if let Some(row) = current_screen.get_mut(y) {
            row.fill(fill_glyph.clone()); // Use clone if Glyph isn't Copy
             // TODO: Mark line y dirty
         }
    }
     // WHY: Move cursor to start of line after IL, consistent with some terminals.
     term.cursor.x = 0;
     term.wrap_next = false; // Explicit cursor move resets wrap
}

/// Deletes `n` lines starting at the cursor's row, within the scroll region. (DL)
/// Replaces rotate_left with manual copy for correctness when n == region_height.
pub(super) fn delete_lines(term: &mut Term, n: usize) {
     trace!("delete_lines: n={}, cursor=({}, {})", n, term.cursor.x, term.cursor.y);
     // Check if cursor is within scroll region
     if term.cursor.y < term.scroll_top || term.cursor.y > term.scroll_bot {
        return;
    }
    let n = min(n, term.scroll_bot + 1 - term.cursor.y);
    if n == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    let top = term.cursor.y;
    let bot = term.scroll_bot;

    // Manual copy up: Move lines [top+n..=bot] up to [top..=bot-n]
    // We iterate upwards.
    for i in top..=(bot.saturating_sub(n)) {
        // See clone note in insert_blank_lines
        current_screen[i] = current_screen[i+n].clone();
    }

    // Fill the new blank lines at the bottom with blanks
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attrs
    for y in (bot + 1 - n)..=bot {
         if let Some(row) = current_screen.get_mut(y) {
            row.fill(fill_glyph.clone()); // Use clone if Glyph isn't Copy
             // TODO: Mark line y dirty
         }
    }
     // WHY: Move cursor to start of line after DL, consistent with some terminals.
     term.cursor.x = 0;
     term.wrap_next = false; // Explicit cursor move resets wrap
}

/// Inserts `n` blank characters at the cursor position. (ICH)
pub(super) fn insert_blank_chars(term: &mut Term, n: usize) {
    trace!("insert_blank_chars: n={}, cursor=({}, {})", n, term.cursor.x, term.cursor.y);
    let n = min(n, term.width.saturating_sub(term.cursor.x));
    if n == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    if let Some(row) = current_screen.get_mut(term.cursor.y) {
        row[term.cursor.x..].rotate_right(n);

        let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attrs
        for x in term.cursor.x..(term.cursor.x + n) {
             if let Some(cell) = row.get_mut(x) {
                 *cell = fill_glyph;
             }
        }
         // TODO: Mark line dirty
    }
     // Inserting chars does not affect wrap_next
}

/// Deletes `n` characters starting at the cursor position. (DCH)
pub(super) fn delete_chars(term: &mut Term, n: usize) {
     trace!("delete_chars: n={}, cursor=({}, {})", n, term.cursor.x, term.cursor.y);
    let n = min(n, term.width.saturating_sub(term.cursor.x));
    if n == 0 { return; }

    let current_screen = if term.using_alt_screen { &mut term.alt_screen } else { &mut term.screen };
    if let Some(row) = current_screen.get_mut(term.cursor.y) {
        row[term.cursor.x..].rotate_left(n);

        let fill_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attrs
        let fill_start = term.width.saturating_sub(n);
        for x in fill_start..term.width {
             if let Some(cell) = row.get_mut(x) {
                 *cell = fill_glyph;
             }
        }
         // TODO: Mark line dirty
    }
     // Deleting chars does not affect wrap_next
}

// --- Mode Setting ---

/// Sets the scrolling region (DECSTBM). Input is 1-based, stored 0-based.
pub(super) fn set_scrolling_region(term: &mut Term, top_req: usize, bot_req: usize) {
     let top = top_req.saturating_sub(1);
     let bot = bot_req.saturating_sub(1);

     if top < bot && bot < term.height {
         term.scroll_top = top;
         term.scroll_bot = bot;
         trace!("Set scrolling region: {}-{}", term.scroll_top, term.scroll_bot);
     } else {
         // If invalid region (top >= bot or bot >= height), reset to full height
         term.scroll_top = 0;
         term.scroll_bot = term.height.saturating_sub(1);
         trace!("Invalid scrolling region ({}-{}), reset to full screen", top_req, bot_req);
     }
     // Move cursor to home position (absolute or relative based on origin mode)
     set_cursor_pos(term, 1, 1);
}

/// Enables origin mode (DECOM). Cursor moves to top-left of scroll region.
pub(super) fn enable_origin_mode(term: &mut Term) {
    term.dec_modes.origin_mode = true;
    // Move cursor to home position (which is now relative to scroll region)
    set_cursor_pos(term, 1, 1);
    trace!("Enabled origin mode");
}

/// Disables origin mode (DECOM). Cursor moves to absolute (0,0).
pub(super) fn disable_origin_mode(term: &mut Term) {
    term.dec_modes.origin_mode = false;
    // Move cursor to absolute home
    term.cursor = Cursor::default();
    term.wrap_next = false; // Reset wrap on mode change
    trace!("Disabled origin mode");
}

/// Switches to the alternate screen buffer.
pub(super) fn enter_alt_screen(term: &mut Term) {
    if term.using_alt_screen { return; }

    trace!("Entering alt screen");
    term.using_alt_screen = true;
    erase_whole_display(term); // Erase the *alternate* screen
    term.cursor = Cursor::default(); // Move cursor to home on alt screen
    term.wrap_next = false;
}

/// Switches back to the main screen buffer.
pub(super) fn exit_alt_screen(term: &mut Term) {
    if !term.using_alt_screen { return; }

    trace!("Exiting alt screen");
    term.using_alt_screen = false;
    // Cursor position is implicitly restored by switching back to the main screen's state,
    // unless DECSC/DECRC (1048/1049) was used.
}

/// Resets the terminal state to defaults. (RIS)
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
     // TODO: Mark lines dirty
}

// --- Backend Interaction Placeholders ---

/// Placeholder for setting terminal modes via backend.
#[allow(unused_variables)]
pub(super) fn xsetmode(set: bool, mode: u16) {
     trace!("xsetmode called: set={}, mode={}", set, mode);
}

// --- Screen Resizing ---

/// Resizes the terminal screen buffers and associated state. Called by Term::resize.
pub(super) fn resize(term: &mut Term, new_width: usize, new_height: usize) {
    trace!("Screen resize logic: new_size={}x{}", new_width, new_height);
    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };

    term.screen.resize_with(new_height, || vec![default_glyph; new_width]);
    for row in term.screen.iter_mut() {
        row.resize(new_width, default_glyph);
    }
    term.alt_screen.resize_with(new_height, || vec![default_glyph; new_width]);
    for row in term.alt_screen.iter_mut() {
        row.resize(new_width, default_glyph);
    }

    term.tabs.resize(new_width, false);
    for i in (0..new_width).step_by(DEFAULT_TAB_INTERVAL) {
        if let Some(tab) = term.tabs.get_mut(i) {
            *tab = true;
        }
    }

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


// Add the module declaration for the tests file
#[cfg(test)]
mod tests;
