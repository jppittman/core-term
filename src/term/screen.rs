// src/term/screen.rs

//! Handles screen buffer manipulation, cursor movement, erasing, scrolling, DECOM, DECSTBM, etc.
//! Adheres to the style guide defined in STYLE.md.

// Import necessary items
use super::{Term, Cursor, DecModes}; // Import Term, Cursor, DecModes from parent module (mod.rs)
use crate::glyph::Glyph;
use std::cmp::min;

// --- Constants ---

/// Default top margin for scrolling region (0-based index).
const DEFAULT_SCROLL_TOP: usize = 0;
/// Default parameter value (often 1) for sequences like CUP, CUU, etc.
const DEFAULT_PARAM_VALUE: usize = 1;

// --- Screen Access ---

/// Returns a mutable reference to the currently active screen buffer (main or alternate).
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
///
/// # Returns
///
/// A mutable reference to the `screen` or `alt_screen` vector.
pub(super) fn current_screen_mut(term: &mut Term) -> &mut Vec<Vec<Glyph>> {
    if term.using_alt_screen {
        &mut term.alt_screen
    } else {
        &mut term.screen
    }
}

// --- Resizing ---

/// Resizes the terminal emulator state (both screen buffers).
///
/// Resets the scrolling region to full height and recalculates tab stops.
/// Clamps cursor positions to the new boundaries.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `new_width` - The new terminal width in columns.
/// * `new_height` - The new terminal height in rows.
pub(super) fn resize(term: &mut Term, new_width: usize, new_height: usize) {
    // Guard clause: Exit early if dimensions haven't changed.
    if new_width == term.width && new_height == term.height {
        return;
    }

    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };

    // Helper closure to resize a single screen buffer.
    // Captures `new_width`, `new_height`, and `default_glyph`.
    let resize_buffer = |buffer: &mut Vec<Vec<Glyph>>, old_height: usize| {
        // Ensure buffer has the correct number of rows (lines).
        buffer.resize_with(new_height, || vec![default_glyph; new_width]);
        // Ensure each row has the correct number of columns.
        for y in 0..new_height {
            // Only resize existing rows if they were part of the old height.
            // New rows are already correctly sized by `resize_with`.
            if y < old_height {
                buffer[y].resize(new_width, default_glyph);
            }
        }
    };

    let old_height = term.height;
    resize_buffer(&mut term.screen, old_height);
    resize_buffer(&mut term.alt_screen, old_height);

    // Update terminal dimensions
    term.width = new_width;
    term.height = new_height;

    // Reset scrolling region to full height on resize (standard behavior).
    term.scroll_top = DEFAULT_SCROLL_TOP;
    term.scroll_bot = term.height.saturating_sub(1);

    // Clamp cursor and saved cursor positions to the new bounds.
    term.cursor.x = min(term.cursor.x, term.width.saturating_sub(1));
    term.cursor.y = min(term.cursor.y, term.height.saturating_sub(1));
    term.saved_cursor.x = min(term.saved_cursor.x, term.width.saturating_sub(1));
    term.saved_cursor.y = min(term.saved_cursor.y, term.height.saturating_sub(1));
    term.saved_cursor_alt.x = min(term.saved_cursor_alt.x, term.width.saturating_sub(1));
    term.saved_cursor_alt.y = min(term.saved_cursor_alt.y, term.height.saturating_sub(1));

    // Recalculate tab stops based on the new width.
    term.tabs = (0..term.width).map(|i| i % super::DEFAULT_TAB_INTERVAL == 0).collect();
}

// --- Scrolling Region and Origin Mode ---

/// Sets the scrolling region (DECSTBM - Set Top and Bottom Margins).
///
/// Parameters `top` and `bottom` are 1-based row numbers.
/// If parameters are invalid (e.g., top=0, bottom=0, top >= bottom, bottom > height),
/// the scrolling region is reset to the full height of the screen.
/// After setting the region, the cursor is moved to the absolute home position (0, 0).
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `top` - The 1-based row number for the top margin.
/// * `bottom` - The 1-based row number for the bottom margin.
pub(super) fn set_scrolling_region(term: &mut Term, top: usize, bottom: usize) {
    let height = term.height;
    // Convert 1-based parameters to 0-based indices.
    let mut t = top.saturating_sub(1);
    let mut b = bottom.saturating_sub(1);

    // Validate region parameters. Reset to full height if invalid.
    // Style Guide: Use guard clauses or clear conditions instead of deep nesting.
    let reset_region = top == 0 || bottom == 0 || t >= b || b >= height;
    if reset_region {
        t = DEFAULT_SCROLL_TOP;
        b = height.saturating_sub(1);
    }

    term.scroll_top = t;
    term.scroll_bot = b;

    // Move cursor to absolute home position (0, 0) after setting region.
    // This is standard behavior for DECSTBM.
    set_cursor_pos_absolute(term, 0, 0);
}

/// Enables DEC Origin Mode (DECOM).
///
/// Sets the origin mode flag and moves the cursor to the top-left
/// corner of the current scrolling region. Cursor addressing becomes relative
/// to this new origin.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn enable_origin_mode(term: &mut Term) {
    if !term.dec_modes.origin_mode {
        term.dec_modes.origin_mode = true;
        // Move cursor to top-left of the scrolling region (absolute coordinates).
        set_cursor_pos_absolute(term, 0, term.scroll_top);
    }
}

/// Disables DEC Origin Mode (DECOM).
///
/// Clears the origin mode flag. Cursor addressing becomes absolute relative
/// to the top-left corner (0,0) of the screen. The cursor position itself
/// is not changed when disabling the mode.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn disable_origin_mode(term: &mut Term) {
    term.dec_modes.origin_mode = false;
    // Cursor position remains, but interpretation changes to absolute.
}

/// Sets or resets DEC Origin Mode (DECOM).
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `enable` - `true` to enable origin mode, `false` to disable.
///
/// # Deprecation Note
///
/// This function uses a boolean argument, which is discouraged by the style guide.
/// Prefer using `enable_origin_mode` or `disable_origin_mode` instead.
#[deprecated(note = "Use enable_origin_mode or disable_origin_mode instead")]
pub(super) fn set_origin_mode(term: &mut Term, enable: bool) {
    if enable {
        enable_origin_mode(term);
    } else {
        disable_origin_mode(term);
    }
}


// --- Scrolling ---

/// Scrolls the defined scrolling region (`term.scroll_top` to `term.scroll_bot`) up by `n` lines.
///
/// Lines shifted off the top of the region are lost.
/// Blank lines (using default attributes) are inserted at the bottom of the region.
/// Does nothing if `n` is 0 or the region is invalid.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `n` - The number of lines to scroll up.
pub(super) fn scroll_up(term: &mut Term, n: usize) {
    let top = term.scroll_top;
    let bot = term.scroll_bot;
    let height = term.height;

    // Guard clauses for invalid region or no scroll needed.
    if n == 0 || top >= bot || bot >= height {
        return;
    }

    let num_lines_in_region = bot - top + 1;
    let lines_to_scroll = min(n, num_lines_in_region);

    // Guard clause if effective scroll is 0.
    if lines_to_scroll == 0 {
        return;
    }

    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };
    let screen = current_screen_mut(term);

    // Rotate lines upwards within the slice screen[top..=bot].
    screen[top..=bot].rotate_left(lines_to_scroll);

    // Clear the newly exposed lines at the bottom of the region.
    // Calculate the start index for clearing carefully.
    let clear_start = bot.saturating_sub(lines_to_scroll) + 1;
    for y in clear_start..=bot {
        // Bounds check (should be redundant due to slice logic, but safe).
        if y < height {
            screen[y].fill(default_glyph);
        }
    }
}

/// Scrolls the defined scrolling region (`term.scroll_top` to `term.scroll_bot`) down by `n` lines.
///
/// Lines shifted off the bottom of the region are lost.
/// Blank lines (using default attributes) are inserted at the top of the region.
/// Does nothing if `n` is 0 or the region is invalid.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `n` - The number of lines to scroll down.
pub(super) fn scroll_down(term: &mut Term, n: usize) {
    let top = term.scroll_top;
    let bot = term.scroll_bot;
    let height = term.height;

    // Guard clauses for invalid region or no scroll needed.
    if n == 0 || top >= bot || bot >= height {
        return;
    }

    let num_lines_in_region = bot - top + 1;
    let lines_to_scroll = min(n, num_lines_in_region);

    // Guard clause if effective scroll is 0.
    if lines_to_scroll == 0 {
        return;
    }

    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };
    let screen = current_screen_mut(term);

    // Rotate lines downwards within the slice screen[top..=bot].
    screen[top..=bot].rotate_right(lines_to_scroll);

    // Clear the newly exposed lines at the top of the region.
    // Calculate the end index for clearing carefully (exclusive).
    let clear_end = min(top + lines_to_scroll, bot + 1);
    for y in top..clear_end {
         // Bounds check (should be redundant, but safe).
         if y < height {
             screen[y].fill(default_glyph);
         }
    }
}

// --- Cursor Movement Helpers ---

/// Gets the effective top boundary row index for cursor movement.
/// Respects DEC Origin Mode (DECOM).
#[inline]
pub fn effective_top(term: &Term) -> usize {
    if term.dec_modes.origin_mode { term.scroll_top } else { 0 }
}

/// Gets the effective bottom boundary row index for cursor movement.
/// Respects DEC Origin Mode (DECOM).
#[inline]
pub fn effective_bottom(term: &Term) -> usize {
    if term.dec_modes.origin_mode { term.scroll_bot } else { term.height.saturating_sub(1) }
}

// --- Cursor Movement ---

/// Moves the cursor relative to its current position (`dx`, `dy`).
///
/// Clamps the final position horizontally to the screen width and
/// vertically within the effective boundaries defined by the current
/// origin mode and scrolling region.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `dx` - The change in the horizontal position (columns).
/// * `dy` - The change in the vertical position (rows).
pub(super) fn move_cursor(term: &mut Term, dx: isize, dy: isize) {
    let width = term.width;
    let current_x = term.cursor.x as isize;
    let current_y = term.cursor.y as isize;

    // Calculate new X, clamping horizontally 0 to width-1.
    let new_x = (current_x + dx).clamp(0, width.saturating_sub(1) as isize) as usize;

    // Calculate new Y, clamping vertically within effective boundaries.
    let top_bound = effective_top(term) as isize;
    let bottom_bound = effective_bottom(term) as isize;
    let new_y = (current_y + dy).clamp(top_bound, bottom_bound) as usize;

    term.cursor.x = new_x;
    term.cursor.y = new_y;
}

/// Sets the cursor position (CUP/HVP). Coordinates are 1-based.
///
/// Handles coordinates potentially relative to the scrolling region origin
/// if DEC Origin Mode (DECOM) is active. Clamps the final position horizontally
/// to the screen width and vertically within the effective boundaries.
/// Use `set_cursor_pos_absolute` for explicitly absolute positioning.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `x` - The 1-based target column number.
/// * `y` - The 1-based target row number (potentially relative to origin).
pub(super) fn set_cursor_pos(term: &mut Term, x: usize, y: usize) {
    let width = term.width;
    let top_bound = effective_top(term);
    let bottom_bound = effective_bottom(term);

    // Clamp 1-based X to 0-based index within width.
    let final_x = min(x.saturating_sub(1), width.saturating_sub(1));

    // If origin mode is active, the incoming 1-based 'y' is relative to scroll_top.
    // Convert it to an absolute 0-based coordinate.
    // Otherwise, just convert the 1-based 'y' to 0-based absolute coordinate.
    let absolute_y = if term.dec_modes.origin_mode {
        term.scroll_top + y.saturating_sub(1)
    } else {
        y.saturating_sub(1)
    };

    // Clamp the absolute 0-based Y coordinate within the effective boundaries.
    let final_y = absolute_y.clamp(top_bound, bottom_bound);

    term.cursor.x = final_x;
    term.cursor.y = final_y;
}

/// Sets the cursor to an absolute position (0-based indices), IGNORING origin mode.
///
/// Used internally when absolute positioning is required regardless of DECOM state
/// (e.g., after DECSTBM, during reset). Clamps to screen bounds.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `x` - The 0-based target column index.
/// * `y` - The 0-based target row index.
fn set_cursor_pos_absolute(term: &mut Term, x: usize, y: usize) {
    let width = term.width;
    let height = term.height;
    term.cursor.x = min(x, width.saturating_sub(1));
    term.cursor.y = min(y, height.saturating_sub(1));
}


/// Handles Line Feed (LF), Vertical Tab (VT), Form Feed (FF), or Index (IND, ESC D).
/// Moves cursor down one line. If the cursor is at the bottom margin of the
/// scrolling region, the region scrolls up by one line instead.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn index(term: &mut Term) {
    let y = term.cursor.y;
    let bottom_margin = term.scroll_bot;

    if y == bottom_margin {
        scroll_up(term, DEFAULT_PARAM_VALUE); // Scroll region up by 1 line
    } else {
        // Move cursor down, clamping to ensure it doesn't exceed physical screen height
        // (although logic should prevent this if bottom_margin is correct).
        term.cursor.y = min(y + 1, term.height.saturating_sub(1));
    }
}

/// Handles New Line (NEL, ESC E).
///
/// Performs a Carriage Return (CR) followed by an Index (IND/LF) operation.
/// Moves cursor to the start of the next line, scrolling the region if necessary.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn newline(term: &mut Term) {
    index(term); // Perform index (move down or scroll)
    term.cursor.x = 0; // Move to first column (CR part)
}

/// Handles Reverse Index (RI, ESC M).
///
/// Moves cursor up one line. If the cursor is at the top margin of the
/// scrolling region, the region scrolls down by one line instead.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn reverse_index(term: &mut Term) {
    let y = term.cursor.y;
    let top_margin = term.scroll_top;

    if y == top_margin {
        scroll_down(term, DEFAULT_PARAM_VALUE); // Scroll region down by 1 line
    } else {
        // Move cursor up, saturating subtract ensures it doesn't go below 0.
        term.cursor.y = y.saturating_sub(1);
    }
}


/// Handles Carriage Return (CR). Moves cursor to the beginning (column 0) of the current line.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn carriage_return(term: &mut Term) {
    term.cursor.x = 0;
}

/// Handles Backspace (BS). Moves cursor one column to the left, stopping at the first column (column 0).
/// Does not erase characters.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn backspace(term: &mut Term) {
    // Use saturating_sub for safety, although check is sufficient.
    // if term.cursor.x > 0 { term.cursor.x -= 1; }
    term.cursor.x = term.cursor.x.saturating_sub(1);
}

/// Handles Horizontal Tab (HT). Moves cursor to the next tab stop.
/// If no more tab stops exist on the line, moves to the last column.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn tab(term: &mut Term) {
    let current_x = term.cursor.x;
    let width = term.width;
    let tabs = &term.tabs;

    // Find the index of the next tab stop strictly after the current position.
    // Style Guide: Prefer iterators and functional style where clear.
    let next_tab_index = tabs.iter()
        .enumerate()
        .skip(current_x + 1) // Start searching *after* current_x
        .find(|&(_, &is_stop)| is_stop) // Find the first element where is_stop is true
        .map(|(index, _)| index); // Get the index if found

    match next_tab_index {
        Some(absolute_pos) => {
            // Move to the found tab stop, ensuring it's within bounds (redundant due to find logic, but safe).
            term.cursor.x = min(absolute_pos, width.saturating_sub(1));
        }
        None => {
            // No more tab stops on this line, move to the last column.
            term.cursor.x = width.saturating_sub(1);
        }
    }
}

/// Saves the current cursor position (DECSC / SCOSC).
///
/// This implementation currently only saves the cursor coordinates (x, y).
/// A full implementation might also save character attributes, origin mode state, etc.
/// Saves to the appropriate slot based on whether the alt screen is active.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn save_cursor(term: &mut Term) {
    // TODO: Also save attributes (SGR), origin mode state (DECOM), character set (SCS)
    //       if implementing full DECSC behavior. SCOSC typically only saves position.
    if term.using_alt_screen {
        term.saved_cursor_alt = term.cursor;
    } else {
        term.saved_cursor = term.cursor;
    }
}

/// Restores the saved cursor position (DECRC / SCORC).
///
/// Restores coordinates from the appropriate saved slot (main or alt screen).
/// Ensures the restored cursor position is clamped within the *current* screen
/// dimensions and respects the *current* origin mode setting.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn restore_cursor(term: &mut Term) {
    // TODO: Also restore attributes, origin mode, etc., if saved by `save_cursor`.
    term.cursor = if term.using_alt_screen {
        term.saved_cursor_alt
    } else {
        term.saved_cursor
    };

    // Ensure restored cursor is within current bounds and respects current origin mode.
    let top_bound = effective_top(term);
    let bottom_bound = effective_bottom(term);
    term.cursor.x = min(term.cursor.x, term.width.saturating_sub(1));
    term.cursor.y = term.cursor.y.clamp(top_bound, bottom_bound);
}

// --- Erasing Functions ---

/// Fills a range of cells on a given line with spaces using the current attributes.
/// Internal helper function.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `y` - The 0-based row index to fill.
/// * `x_start` - The 0-based starting column index (inclusive).
/// * `x_end` - The 0-based ending column index (exclusive).
pub fn fill_range(term: &mut Term, y: usize, x_start: usize, x_end: usize) {
    let height = term.height;
    let width = term.width;

    // Guard clauses for invalid ranges or coordinates.
    if y >= height || x_start >= x_end || x_start >= width {
        return;
    }

    // Use current attributes for filling.
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };
    let screen = current_screen_mut(term);

    // Ensure the end index doesn't exceed screen width.
    let clamped_end = min(x_end, width);

    // Fill the specified range on the line.
    // Use slice::fill if Glyph is Copy, otherwise loop.
    // screen[y][x_start..clamped_end].fill(fill_glyph);
     for x in x_start..clamped_end {
         screen[y][x] = fill_glyph;
     }
}

/// Erases from the cursor position to the end of the line (EL 0).
/// Fills with spaces using the current attributes. Cursor position is not changed.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn erase_line_to_end(term: &mut Term) {
    let y = term.cursor.y;
    let x = term.cursor.x;
    let width = term.width;
    fill_range(term, y, x, width);
}

/// Erases from the start of the line to the cursor position (inclusive) (EL 1).
/// Fills with spaces using the current attributes. Cursor position is not changed.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn erase_line_to_start(term: &mut Term) {
    let y = term.cursor.y;
    let x = term.cursor.x;
    // Inclusive erase means up to and including column x.
    fill_range(term, y, 0, x + 1);
}

/// Erases the entire current line (EL 2).
/// Fills with spaces using the current attributes. Cursor position is not changed.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn erase_whole_line(term: &mut Term) {
    let y = term.cursor.y;
    let width = term.width;
    fill_range(term, y, 0, width);
}

/// Erases from the cursor position to the end of the display (ED 0).
/// Includes erasing the rest of the current line and all lines below it.
/// Fills with spaces using the current attributes. Cursor position is not changed.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn erase_display_to_end(term: &mut Term) {
    let cursor_y = term.cursor.y;
    let height = term.height;
    let width = term.width;

    // Erase rest of the current line first.
    erase_line_to_end(term);

    // Fill all lines below the cursor line.
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };
    let screen = current_screen_mut(term);
    for y in (cursor_y + 1)..height {
        // screen[y].fill(fill_glyph); // Use fill if possible
        for x in 0..width {
             screen[y][x] = fill_glyph;
        }
    }
}

/// Erases from the start of the display to the cursor position (inclusive) (ED 1).
/// Includes erasing the beginning of the current line and all lines above it.
/// Fills with spaces using the current attributes. Cursor position is not changed.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn erase_display_to_start(term: &mut Term) {
    let cursor_y = term.cursor.y;
    let height = term.height; // Use height for loop bound check

    // Erase start of the current line (inclusive).
    erase_line_to_start(term);

    // Fill all lines above the cursor line.
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };
    let width = term.width;
    let screen = current_screen_mut(term);
    for y in 0..cursor_y {
         // Bounds check (should be redundant, but safe).
         if y < height {
             // screen[y].fill(fill_glyph); // Use fill if possible
             for x in 0..width {
                  screen[y][x] = fill_glyph;
             }
         }
    }
}

/// Erases the entire display (ED 2).
/// Fills the entire screen with spaces using the current attributes.
/// Cursor position is typically moved to home (0, 0) by most terminals.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn erase_whole_display(term: &mut Term) {
    let height = term.height;
    let width = term.width;
    let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };

    let screen = current_screen_mut(term);
    for y in 0..height {
        // screen[y].fill(fill_glyph); // Use fill if possible
        for x in 0..width {
             screen[y][x] = fill_glyph;
        }
    }

    // Move cursor to home (0,0) after ED 2, respecting origin mode.
    set_cursor_pos(term, 1, 1); // Use 1-based coordinates for set_cursor_pos
}

// --- Insert/Delete Characters and Lines ---

/// Inserts `n` blank characters at the cursor position (ICH).
/// Characters from the cursor position to the right margin are shifted right.
/// Characters shifted off the right margin are lost.
/// Fills inserted space with blanks using current attributes. Cursor does not move.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `n` - The number of blank characters to insert.
pub(super) fn insert_blank_chars(term: &mut Term, n: usize) {
     let y = term.cursor.y;
     let x = term.cursor.x;
     let width = term.width;

     // Guard clauses: Check if insertion is possible/needed.
     if n == 0 || y >= term.height || x >= width {
         return;
     }

     // Calculate actual number of characters to insert (cannot exceed available space).
     let num_to_insert = min(n, width.saturating_sub(x));
     if num_to_insert == 0 {
         return; // No space to insert
     }

     let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };
     let screen = current_screen_mut(term);
     let line = &mut screen[y];

     // Calculate length of the segment to shift right.
     let copy_len = width.saturating_sub(x).saturating_sub(num_to_insert);
     if copy_len > 0 {
         // Shift existing characters to the right using copy_within.
         // Source range: x .. (x + copy_len)
         // Destination start: x + num_to_insert
         line.copy_within(x..(x + copy_len), x + num_to_insert);
     }

     // Fill the gap created at position x with blank characters.
     for i in 0..num_to_insert {
         // Bounds check (should be redundant due to earlier checks, but safe).
         if x + i < width {
             line[x + i] = fill_glyph;
         }
     }
 }

/// Deletes `n` characters starting at the cursor position (DCH).
/// Remaining characters from the right of the deleted section are shifted left.
/// Fills the newly exposed space at the right end of the line with blanks
/// using current attributes. Cursor does not move.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `n` - The number of characters to delete.
pub(super) fn delete_chars(term: &mut Term, n: usize) {
     let y = term.cursor.y;
     let x = term.cursor.x;
     let width = term.width;

     // Guard clauses: Check if deletion is possible/needed.
     if n == 0 || y >= term.height || x >= width {
         return;
     }

     // Calculate actual number of characters to delete.
     let num_to_delete = min(n, width.saturating_sub(x));
     if num_to_delete == 0 {
         return; // Nothing to delete at this position
     }

     let fill_glyph = Glyph { c: ' ', attr: term.current_attributes };
     let screen = current_screen_mut(term);
     let line = &mut screen[y];

     // Calculate where the characters to shift left start.
     let copy_start = x + num_to_delete;
     if copy_start < width {
         // Shift characters left using copy_within.
         // Source range starts after the deleted characters: copy_start..width
         // Destination start: x
         line.copy_within(copy_start.., x);
     }

     // Fill the end of the line (now empty due to the shift) with blanks.
     let fill_start = width.saturating_sub(num_to_delete);
     for i in fill_start..width {
          // Bounds check (redundant but safe).
          if i < width {
              line[i] = fill_glyph;
          }
     }
 }

/// Inserts `n` blank lines at the cursor's row position (IL).
/// Operates within the current scrolling region. Lines within the region
/// starting from the cursor row are shifted down. Lines shifted off the
/// bottom margin are lost. Inserted lines are filled with blanks using
/// current attributes. Cursor does not move.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `n` - The number of blank lines to insert.
pub(super) fn insert_blank_lines(term: &mut Term, n: usize) {
     let y = term.cursor.y;
     let top = term.scroll_top;
     let bot = term.scroll_bot;

     // Guard clauses: IL only works if the cursor is within the scrolling region.
     if n == 0 || y < top || y > bot {
         return;
     }

     // Calculate the effective bottom for insertion (cursor line included).
     let effective_bot = bot;
     // Calculate number of lines to insert, limited by space below cursor in region.
     let num_to_insert = min(n, effective_bot.saturating_sub(y) + 1);

     if num_to_insert > 0 {
         // Scroll down the part of the region from the cursor line to the bottom margin.
         scroll_down_region(term, y, effective_bot, num_to_insert);
     }
 }

/// Deletes `n` lines starting at the cursor's row position (DL).
/// Operates within the current scrolling region. Lines below the deleted lines
/// within the region are shifted up. Blank lines (using current attributes)
/// are inserted at the bottom margin to fill the gap. Cursor does not move.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `n` - The number of lines to delete.
pub(super) fn delete_lines(term: &mut Term, n: usize) {
     let y = term.cursor.y;
     let top = term.scroll_top;
     let bot = term.scroll_bot;

     // Guard clauses: DL only works if the cursor is within the scrolling region.
     if n == 0 || y < top || y > bot {
         return;
     }

     // Calculate the effective bottom for deletion (cursor line included).
     let effective_bot = bot;
     // Calculate number of lines to delete, limited by lines from cursor to bottom.
     let num_to_delete = min(n, effective_bot.saturating_sub(y) + 1);

     if num_to_delete > 0 {
         // Scroll up the part of the region from the cursor line to the bottom margin.
         scroll_up_region(term, y, effective_bot, num_to_delete);
     }
 }

 /// Helper function: Scrolls a specific sub-region *down*. Used by IL.
 /// Internal helper function.
 ///
 /// # Arguments
 /// * `term` - Mutable terminal state.
 /// * `region_top` - 0-based top row of the sub-region to scroll.
 /// * `region_bot` - 0-based bottom row of the sub-region to scroll.
 /// * `n` - Number of lines to scroll down.
 fn scroll_down_region(term: &mut Term, region_top: usize, region_bot: usize, n: usize) {
     let height = term.height;
     let width = term.width;
     let default_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attrs for fill

     // Guard clauses for invalid region or no scroll needed.
     if n == 0 || region_top > region_bot || region_bot >= height {
         return;
     }

     let num_lines_in_region = region_bot - region_top + 1;
     let lines_to_scroll = min(n, num_lines_in_region);
     if lines_to_scroll == 0 {
         return;
     }

     let screen = current_screen_mut(term);
     // Rotate the specified slice downwards.
     screen[region_top..=region_bot].rotate_right(lines_to_scroll);

     // Clear the newly exposed lines at the top of the scrolled sub-region.
     let clear_end = min(region_top + lines_to_scroll, region_bot + 1);
     for y in region_top..clear_end {
         if y < height { // Bounds check
             // screen[y].fill(default_glyph); // Use fill if possible
             for x in 0..width { screen[y][x] = default_glyph; }
         }
     }
 }

 /// Helper function: Scrolls a specific sub-region *up*. Used by DL.
 /// Internal helper function.
 ///
 /// # Arguments
 /// * `term` - Mutable terminal state.
 /// * `region_top` - 0-based top row of the sub-region to scroll.
 /// * `region_bot` - 0-based bottom row of the sub-region to scroll.
 /// * `n` - Number of lines to scroll up.
 fn scroll_up_region(term: &mut Term, region_top: usize, region_bot: usize, n: usize) {
     let height = term.height;
     let width = term.width;
     let default_glyph = Glyph { c: ' ', attr: term.current_attributes }; // Use current attrs for fill

     // Guard clauses for invalid region or no scroll needed.
     if n == 0 || region_top > region_bot || region_bot >= height {
         return;
     }

     let num_lines_in_region = region_bot - region_top + 1;
     let lines_to_scroll = min(n, num_lines_in_region);
     if lines_to_scroll == 0 {
         return;
     }

     let screen = current_screen_mut(term);
     // Rotate the specified slice upwards.
     screen[region_top..=region_bot].rotate_left(lines_to_scroll);

     // Clear the newly exposed lines at the bottom of the scrolled sub-region.
     let clear_start = region_bot.saturating_sub(lines_to_scroll) + 1;
     for y in clear_start..=region_bot {
         if y < height { // Bounds check
             // screen[y].fill(default_glyph); // Use fill if possible
             for x in 0..width { screen[y][x] = default_glyph; }
         }
     }
 }


// --- Character Handling ---

/// Handles a printable character received from the parser.
///
/// Places the character at the current cursor position using the current attributes.
/// Handles line wrapping if the character would exceed the right margin (assumes DECAWM is on).
/// Advances the cursor according to the character's width (currently assumes width 1).
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
/// * `c` - The character to handle.
pub(super) fn handle_printable(term: &mut Term, c: char) {
    // TODO: Implement proper Unicode width calculation (e.g., using unicode-width crate).
    //       Treat NUL explicitly? Terminals might ignore it or display it specially.
    let char_width: usize = if c == '\0' { 0 } else { 1 }; // Basic width assumption.

    // Guard clause: Ignore zero-width characters like NUL for now.
    if char_width == 0 {
        return;
    }

    let width = term.width;
    let height = term.height;
    let current_attributes = term.current_attributes;

    // Check for line wrap.
    // TODO: Check term.dec_modes.autowrap (DECAWM) flag when implemented.
    let needs_wrap = term.cursor.x + char_width > width;
    if needs_wrap { // && term.dec_modes.autowrap {
        newline(term); // Move to start of next line (handles scrolling).
    }

    // Place character if cursor is within screen bounds after potential wrap.
    // Style Guide: Avoid deep nesting - check bounds after wrap.
    if term.cursor.y < height && term.cursor.x < width {
        let y = term.cursor.y;
        let x = term.cursor.x;
        let screen = current_screen_mut(term);
        screen[y][x] = Glyph { c, attr: current_attributes };

        // Advance cursor if character has width and fits on the line.
        if term.cursor.x + char_width <= width {
             term.cursor.x += char_width;
        }
        // If cursor is now exactly at width (e.g., after placing char in last column),
        // behavior depends on terminal (some wrap immediately, some wait for next char).
        // Current logic effectively waits (wrap happens on *next* character).
    }
}

// --- Alt Screen ---

/// Switches to the alternate screen buffer (DECSET 1049).
///
/// Saves the cursor position of the main screen.
/// Activates the alternate screen buffer.
/// Clears the alternate screen.
/// Moves the cursor to the home position (respecting origin mode) on the alt screen.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn enter_alt_screen(term: &mut Term) {
    // Guard clause: Only switch if not already using alt screen.
    if term.using_alt_screen {
        return;
    }

    save_cursor(term); // Save cursor state *before* switching context.
    term.using_alt_screen = true;

    // Per xterm behavior with mode 1049: clear the *newly active* alt screen.
    erase_whole_display(term); // This now operates on term.alt_screen.

    set_cursor_pos_absolute(term, 0, 0);
}

/// Switches back to the main screen buffer (DECRST 1049).
///
/// Activates the main screen buffer.
/// Restores the previously saved cursor position for the main screen.
/// The content of the alternate screen is typically lost.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn exit_alt_screen(term: &mut Term) {
    // Guard clause: Only switch if currently using alt screen.
    if !term.using_alt_screen {
        return;
    }

    term.using_alt_screen = false;
    // Restore cursor position *saved* from the main screen *before* entering alt screen.
    restore_cursor(term); // This now operates on term.screen and restores main screen cursor.
}

// --- Reset ---

/// Resets terminal state to defaults (RIS - Reset to Initial State, ESC c).
///
/// Switches back to the main screen if necessary.
/// Clears the main screen.
/// Resets cursor position, attributes, DEC modes (including origin mode),
/// scrolling region, tab stops, and saved cursor positions.
///
/// # Arguments
///
/// * `term` - The mutable terminal state.
pub(super) fn reset(term: &mut Term) {
    let default_glyph = Glyph { c: ' ', attr: term.default_attributes };

    // Ensure we are on the main screen before resetting state.
    if term.using_alt_screen {
        exit_alt_screen(term);
    }

    // Clear the main screen content.
    let screen = &mut term.screen; // Directly target main screen.
    for row in screen.iter_mut() {
        // row.fill(default_glyph); // Use fill if possible
        for x in 0..term.width { row[x] = default_glyph; }
    }

    // Reset cursor to absolute home.
    term.cursor = Cursor { x: 0, y: 0 };

    // Reset attributes to default.
    term.current_attributes = term.default_attributes;

    // Reset DEC modes to default.
    term.dec_modes = DecModes::default(); // Assumes DecModes::default() exists.

    // Reset scrolling region to full screen.
    term.scroll_top = DEFAULT_SCROLL_TOP;
    term.scroll_bot = term.height.saturating_sub(1);

    // Reset tab stops to default interval.
    term.tabs = (0..term.width).map(|i| i % super::DEFAULT_TAB_INTERVAL == 0).collect();

    // Reset saved cursor positions (RIS affects DECSC/DECRC state).
    term.saved_cursor = Cursor { x: 0, y: 0 };
    term.saved_cursor_alt = Cursor { x: 0, y: 0 };

    // TODO: Reset character sets (SCS) if implemented.
}

#[cfg(test)]
mod tests;
