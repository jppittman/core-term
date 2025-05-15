use std::cmp::min;
use std::collections::VecDeque;

use crate::glyph::{Glyph, Attributes}; // DEFAULT_GLYPH used via fully qualified path
use crate::term::DEFAULT_TAB_INTERVAL;
use log::{trace, warn};

// Define a type alias for a single row in the grid
pub type Row = Vec<Glyph>;
// Define a type alias for the grid itself (for primary and alternate screens)
pub type Grid = Vec<Row>;

/// Defines the modes for clearing tab stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabClearMode {
    /// Clear tab stop at the current cursor column.
    CurrentColumn,
    /// Clear all tab stops.
    All,
    /// Represents an unsupported or unknown mode.
    Unsupported,
}

impl From<u16> for TabClearMode {
    fn from(value: u16) -> Self {
        match value {
            0 => TabClearMode::CurrentColumn,
            2 | 5 => TabClearMode::All, // Mode 5 is often treated as "All"
            _ => TabClearMode::Unsupported,
        }
    }
}


/// Represents the state of the terminal screen, including display grids,
/// cursor, scrollback, and styling attributes.
///
/// This struct is responsible for managing the visual state of the terminal.
/// It handles operations like character insertion, deletion, scrolling, and resizing,
/// while tracking which lines have changed (`dirty` flags) to optimize rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Screen {
    /// The primary screen grid.
    pub grid: Grid,
    /// The alternate screen grid, used by full-screen applications.
    pub alt_grid: Grid,
    /// Scrollback buffer; lines that have scrolled off the primary screen.
    pub scrollback: VecDeque<Row>,
    /// Maximum number of lines to store in the scrollback buffer.
    scrollback_limit: usize,
    /// True if the alternate screen (`alt_grid`) is currently active.
    pub alt_screen_active: bool,
    /// Current cursor state (position and attributes) for the screen.
    /// This cursor's position is always absolute to the current grid.
    /// If `origin_mode` is active, logical cursor operations are translated
    /// relative to scroll margins before updating this screen cursor.
    pub cursor: Cursor,
    /// Saved cursor state for the primary screen (used when switching to/from alt screen).
    saved_cursor_primary: Cursor,
    /// Saved cursor state for the alternate screen.
    saved_cursor_alternate: Cursor,
    /// Screen width in columns.
    pub width: usize,
    /// Screen height in rows.
    pub height: usize,
    /// Top margin of the scrolling region (0-based, inclusive).
    scroll_top: usize,
    /// Bottom margin of the scrolling region (0-based, inclusive).
    scroll_bot: usize,
    /// Tab stops; `tabs[i]` is true if column `i` is a tab stop.
    tabs: Vec<bool>,
    /// Dirty flags for each line; `dirty[y] = 1` means line `y` needs redraw.
    pub dirty: Vec<u8>,
    /// Default attributes for new or cleared glyphs.
    default_attributes: Attributes,
    /// Origin Mode (DECOM); affects how cursor positions are interpreted
    /// relative to the scrolling margins by the `TerminalEmulator`.
    /// `Screen` itself uses absolute coordinates for its `cursor` field,
    /// but this flag informs the `TerminalEmulator`'s coordinate translation.
    pub origin_mode: bool,
}

/// Represents the cursor's state, including position and rendering attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cursor {
    /// Horizontal position (column, 0-based).
    pub x: usize,
    /// Vertical position (row, 0-based).
    /// For `Screen::cursor`, this `y` is always absolute to the grid.
    /// The logical `TerminalEmulator::cursor`'s `y` might be relative to `scroll_top`
    /// if `origin_mode` is active, and then translated to this absolute `y`.
    pub y: usize,
    /// Attributes for new glyphs to be written at the cursor position.
    pub attributes: Attributes,
}

impl Screen {
    /// Creates a new `Screen` with the given dimensions and scrollback limit.
    ///
    /// Initializes primary and alternate grids, the scrollback buffer, cursor,
    /// tab stops, and default attributes. All lines are initially marked dirty.
    ///
    /// # Arguments
    /// * `width` - The width of the screen in columns.
    /// * `height` - The height of the screen in rows.
    /// * `scrollback_limit` - The maximum number of lines to keep in the scrollback buffer.
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let default_attributes = Attributes::default();
        let default_glyph = Glyph { c: ' ', attr: default_attributes };

        let grid = vec![vec![default_glyph.clone(); width]; height];
        let alt_grid = vec![vec![default_glyph.clone(); width]; height];
        let scrollback = VecDeque::with_capacity(scrollback_limit);

        let mut tabs = vec![false; width];
        // Initialize tab stops at regular intervals.
        // Ensure loop variable `i` is `usize` for correct comparison and indexing.
        for i in (DEFAULT_TAB_INTERVAL as usize..width).step_by(DEFAULT_TAB_INTERVAL as usize) {
            if i < tabs.len() { // Defensively check bounds.
                tabs[i] = true;
            }
        }

        Screen {
            grid,
            alt_grid,
            scrollback,
            scrollback_limit,
            alt_screen_active: false,
            cursor: Cursor { x: 0, y: 0, attributes: default_attributes },
            saved_cursor_primary: Cursor { x: 0, y: 0, attributes: default_attributes },
            saved_cursor_alternate: Cursor { x: 0, y: 0, attributes: default_attributes },
            width,
            height,
            scroll_top: 0,
            scroll_bot: height.saturating_sub(1), // Default scroll region is full screen.
            tabs,
            dirty: vec![1; height], // Mark all lines dirty initially.
            default_attributes,
            origin_mode: false,
        }
    }

    /// Returns a mutable reference to the currently active grid (primary or alternate).
    /// This is an internal helper.
    fn active_grid_mut(&mut self) -> &mut Grid {
        if self.alt_screen_active {
            &mut self.alt_grid
        } else {
            &mut self.grid
        }
    }

    /// Returns an immutable reference to the currently active grid.
    pub fn active_grid(&self) -> &Grid {
        if self.alt_screen_active {
            &self.alt_grid
        } else {
            &self.grid
        }
    }
    
    /// Returns the 0-based top row of the scrolling region.
    pub fn scroll_top(&self) -> usize {
        self.scroll_top
    }

    /// Returns the 0-based bottom row of the scrolling region.
    pub fn scroll_bot(&self) -> usize {
        self.scroll_bot
    }
    
    /// Returns the current scrollback limit.
    pub fn scrollback_limit(&self) -> usize {
        self.scrollback_limit
    }


    /// Fills a rectangular region of a single line `y` from `x_start` (inclusive)
    /// to `x_end` (exclusive) with the provided `fill_glyph`.
    /// Marks the line `y` as dirty. Coordinates are clamped to screen dimensions.
    pub fn fill_region(&mut self, y: usize, x_start: usize, x_end: usize, fill_glyph: Glyph) {
        let current_width = self.width; // Cache to avoid borrow issues.
        let current_height = self.height;

        if y >= current_height {
            warn!("fill_region: y coordinate {} is out of bounds (height {})", y, current_height);
            return;
        }

        let row = match self.active_grid_mut().get_mut(y) {
            Some(r) => r,
            None => {
                // This case implies an inconsistency if y < current_height.
                warn!("fill_region: Failed to get row {} despite bounds check.", y);
                return;
            }
        };

        let start = min(x_start, current_width);
        let end = min(x_end, current_width);

        if start < end { // Ensure there's a valid region to fill.
            row[start..end].fill(fill_glyph);
        }

        // It's crucial to mark the line dirty after modification.
        if y < self.dirty.len() { // Should always be true if y < current_height.
            self.dirty[y] = 1;
        }
    }

    /// Scrolls the content of the defined scrolling region up by `n` lines.
    /// New lines appearing at the bottom of the region are filled with `fill_glyph`.
    /// All lines within the affected region are marked dirty.
    pub fn scroll_up_serial(&mut self, n: usize, fill_glyph: Glyph) {
        if self.scroll_top > self.scroll_bot || self.scroll_bot >= self.height {
            warn!("scroll_up_serial: Invalid scroll region top={}, bot={}, height={}", self.scroll_top, self.scroll_bot, self.height);
            return;
        }
        let n_val = n.min(self.scroll_bot - self.scroll_top + 1); // Cannot scroll more than the region's height.
        if n_val == 0 { return; } // No lines to scroll.
        trace!("Scrolling up by {} lines in region ({}, {})", n_val, self.scroll_top, self.scroll_bot);

        let top = self.scroll_top; // Cache for use after mutable borrow.
        let bot = self.scroll_bot;

        self.active_grid_mut()[top..=bot].rotate_left(n_val);

        // Fill the new blank lines at the bottom of the scrolling region.
        for y_idx in (bot.saturating_sub(n_val) + 1)..=bot {
            if let Some(row_to_fill) = self.active_grid_mut().get_mut(y_idx) {
                row_to_fill.fill(fill_glyph.clone()); // Must clone fill_glyph for each row.
            }
        }

        // Mark all lines in the scrolled region as dirty.
        for y_idx in top..=bot {
            if y_idx < self.dirty.len() { // Defensive check.
                 self.dirty[y_idx] = 1;
            }
        }
    }

    /// Scrolls the content of the defined scrolling region down by `n` lines.
    /// New lines appearing at the top of the region are filled with `fill_glyph`.
    /// All lines within the affected region are marked dirty.
    pub fn scroll_down_serial(&mut self, n: usize, fill_glyph: Glyph) {
        if self.scroll_top > self.scroll_bot || self.scroll_bot >= self.height {
             warn!("scroll_down_serial: Invalid scroll region top={}, bot={}, height={}", self.scroll_top, self.scroll_bot, self.height);
            return;
        }
        let n_val = n.min(self.scroll_bot - self.scroll_top + 1);
        if n_val == 0 { return; }
        trace!("Scrolling down by {} lines in region ({}, {})", n_val, self.scroll_top, self.scroll_bot);

        let top = self.scroll_top;
        let bot = self.scroll_bot;

        self.active_grid_mut()[top..=bot].rotate_right(n_val);

        // Fill the new blank lines at the top of the scrolling region.
        for y_idx in top..(top + n_val) {
            if let Some(row_to_fill) = self.active_grid_mut().get_mut(y_idx) {
                row_to_fill.fill(fill_glyph.clone());
            }
        }

        for y_idx in top..=bot {
            if y_idx < self.dirty.len() {
                self.dirty[y_idx] = 1;
            }
        }
    }

    /// Inserts `n` blank characters (using `fill_glyph`) in line `y` at column `x`.
    /// Existing characters from `x` onwards are shifted right; characters shifted off are lost.
    /// The line `y` is marked dirty.
    pub fn insert_blank_chars_in_line(&mut self, y: usize, x: usize, n: usize, fill_glyph: Glyph) {
        if y >= self.height {
            warn!("insert_blank_chars_in_line: y coordinate {} out of bounds.", y);
            return;
        }
        // Mark line dirty early, as it will likely be modified.
        if y < self.dirty.len() { self.dirty[y] = 1; }


        if x >= self.width || n == 0 { // No-op if inserting at/beyond width or inserting zero chars.
            return;
        }

        let current_width = self.width; // Cache for use after mutable borrow.
        let row = match self.active_grid_mut().get_mut(y) {
            Some(r) => r,
            None => { warn!("insert_blank_chars_in_line: Failed to get row {} for insertion.", y); return; }
        };

        // Number of blanks to insert, capped by how many chars can be shifted right.
        let count = n.min(current_width - x);
        if count == 0 { // Should be caught by x >= current_width, but defensive.
            return;
        }

        row[x..].rotate_right(count);
        // Fill the newly created blank spaces.
        for fill_x_idx in x..(x + count) {
            // fill_x_idx is guaranteed to be < current_width due to 'count' calculation.
            if let Some(cell) = row.get_mut(fill_x_idx) {
                *cell = fill_glyph.clone();
            }
        }
    }

    /// Deletes `n` characters in line `y` starting at column `x`.
    /// Characters to the right are shifted left. Space freed at the end is filled with `fill_glyph`.
    /// The line `y` is marked dirty.
    pub fn delete_chars_in_line(&mut self, y: usize, x: usize, n: usize, fill_glyph: Glyph) {
        if y >= self.height {
            warn!("delete_chars_in_line: y coordinate {} out of bounds.", y);
            return;
        }
        if y < self.dirty.len() { self.dirty[y] = 1; } // Mark line dirty early.

        if x >= self.width || n == 0 { // No-op if deleting at/beyond width or deleting zero chars.
            return;
        }

        let current_width = self.width; // Cache.
        let row = match self.active_grid_mut().get_mut(y) {
            Some(r) => r,
            None => { warn!("delete_chars_in_line: Failed to get row {} for deletion.", y); return; }
        };

        let row_len = current_width; // Assuming row.len() == self.width.
        // Number of chars to delete, capped by remaining chars in line from x.
        let count = n.min(row_len - x);
        if count == 0 { // Should be caught by x >= row_len, but defensive.
            return;
        }

        row[x..].rotate_left(count);
        // Fill the end of the line with blank glyphs.
        let fill_start_idx = row_len.saturating_sub(count);
        for fill_x_idx in fill_start_idx..row_len {
            if let Some(cell) = row.get_mut(fill_x_idx) {
                *cell = fill_glyph.clone();
            }
        }
    }

    /// Resizes the screen to `new_width` and `new_height`, and updates the scrollback limit.
    ///
    /// This operation recreates the main and alternate display grids.
    /// Existing content in the scrollback buffer has its lines resized to `new_width`;
    /// the scrollback buffer is also truncated if its length exceeds `new_scrollback_limit`.
    /// Cursor positions are clamped to the new dimensions. All lines are marked dirty.
    ///
    /// **Note:** This currently does not attempt to reflow content from the old grid to the new one.
    pub fn resize(&mut self, new_width: usize, new_height: usize, new_scrollback_limit: usize) {
        warn!("Screen resize from {}x{} (scrollback: {}) to {}x{} (scrollback: {})",
              self.width, self.height, self.scrollback_limit,
              new_width, new_height, new_scrollback_limit);

        let old_width = self.width;
        self.width = new_width;
        self.height = new_height;
        self.scrollback_limit = new_scrollback_limit;

        let default_glyph = Glyph { c: ' ', attr: self.default_attributes };

        self.grid = vec![vec![default_glyph.clone(); new_width]; new_height];
        self.alt_grid = vec![vec![default_glyph.clone(); new_width]; new_height];

        // Adjust scrollback content if width changed.
        if old_width != new_width {
            for row_ref in self.scrollback.iter_mut() {
                row_ref.resize(new_width, default_glyph.clone());
            }
        }
        // Enforce new scrollback line limit.
        while self.scrollback.len() > self.scrollback_limit {
            self.scrollback.pop_front(); // Remove oldest lines.
        }
        // Optionally shrink capacity if much larger than limit.
        if self.scrollback.capacity() > self.scrollback_limit + SOME_REASONABLE_SLACK {
            self.scrollback.shrink_to_fit();
        }

        // Reset scroll region to full new height.
        self.scroll_top = 0;
        self.scroll_bot = new_height.saturating_sub(1);

        // Clamp cursor positions to new bounds.
        self.cursor.x = min(self.cursor.x, new_width.saturating_sub(1));
        self.cursor.y = min(self.cursor.y, new_height.saturating_sub(1));
        self.saved_cursor_primary.x = min(self.saved_cursor_primary.x, new_width.saturating_sub(1));
        self.saved_cursor_primary.y = min(self.saved_cursor_primary.y, new_height.saturating_sub(1));
        self.saved_cursor_alternate.x = min(self.saved_cursor_alternate.x, new_width.saturating_sub(1));
        self.saved_cursor_alternate.y = min(self.saved_cursor_alternate.y, new_height.saturating_sub(1));

        // Reinitialize tab stops for the new width.
        // Ensure loop variable `i` is `usize` for correct comparison and indexing.
        self.tabs = vec![false; new_width];
        for i in (DEFAULT_TAB_INTERVAL as usize..new_width).step_by(DEFAULT_TAB_INTERVAL as usize) {
            if i < self.tabs.len() { self.tabs[i] = true; }
        }
        // Resize dirty flags and mark all new lines dirty.
        self.dirty = vec![1; new_height];
        trace!("Screen resized. New dimensions: {}x{}. All lines marked dirty.", new_width, new_height);
    }

    /// Marks all lines on the screen as dirty, typically forcing a full redraw.
    pub fn mark_all_dirty(&mut self) {
        self.dirty.fill(1);
    }

    /// Marks a single line `y` as dirty.
    /// Warns if `y` is out of bounds for the dirty flags vector.
    pub fn mark_line_dirty(&mut self, y: usize) {
        if y < self.dirty.len() {
            self.dirty[y] = 1;
        } else {
            // This implies y >= self.height, which should ideally be caught by callers.
            warn!("mark_line_dirty: y coordinate {} is out of bounds for dirty flags (len {})", y, self.dirty.len());
        }
    }

    /// Clears all dirty flags. Typically called after the screen has been rendered.
    pub fn clear_dirty_flags(&mut self) {
        self.dirty.fill(0);
    }

    /// Switches to the alternate screen buffer.
    ///
    /// If not already active, it saves the primary cursor's state, switches to the
    /// alternate grid, and optionally clears the alternate screen. The alternate
    /// cursor's state is then restored. All lines are marked dirty to trigger a redraw.
    ///
    /// # Arguments
    /// * `clear_alt_screen` - If true, the alternate screen is filled with default glyphs.
    pub fn enter_alt_screen(&mut self, clear_alt_screen: bool) {
        if self.alt_screen_active { return; } // Already in alt screen.
        self.alt_screen_active = true;
        self.saved_cursor_primary = self.cursor; // Save primary screen's cursor state.

        if clear_alt_screen {
            let fill_glyph = Glyph { c: ' ', attr: self.default_attributes };
            for y_idx in 0..self.height {
                // Use fill_region to ensure dirty flags are set correctly for each line.
                self.fill_region(y_idx, 0, self.width, fill_glyph.clone());
            }
        }
        // Restore or reset alternate screen's cursor state.
        self.cursor = self.saved_cursor_alternate;
        self.cursor.x = min(self.cursor.x, self.width.saturating_sub(1));
        self.cursor.y = min(self.cursor.y, self.height.saturating_sub(1));

        self.mark_all_dirty(); // Alt screen content is now active and needs redraw.
        trace!("Entered alt screen. All lines marked dirty.");
    }

    /// Switches back to the primary screen buffer from the alternate screen.
    ///
    /// If active, it saves the alternate cursor's state, switches to the primary grid,
    /// and restores the primary cursor's state. All lines are marked dirty.
    pub fn exit_alt_screen(&mut self) {
        if !self.alt_screen_active { return; } // Already in primary screen.
        self.saved_cursor_alternate = self.cursor; // Save alt screen's cursor state.
        self.alt_screen_active = false;
        self.cursor = self.saved_cursor_primary; // Restore primary screen's cursor state.
        self.cursor.x = min(self.cursor.x, self.width.saturating_sub(1));
        self.cursor.y = min(self.cursor.y, self.height.saturating_sub(1));

        self.mark_all_dirty(); // Primary screen content is now active and needs redraw.
        trace!("Exited alt screen. All lines marked dirty.");
    }

    /// Sets the scrolling region (DECSTBM).
    /// `top` and `bottom` are 1-based inclusive row numbers from the terminal's perspective.
    /// The cursor is moved to the home position (0,0) after setting the region.
    /// If `origin_mode` is active, this home position is relative to the new margins;
    /// otherwise, it's absolute (0,0) of the screen.
    pub fn set_scrolling_region(&mut self, top: usize, bottom: usize) {
        // Convert 1-based input to 0-based internal representation.
        let t = top.saturating_sub(1);
        let b = bottom.saturating_sub(1);

        if t < b && b < self.height { // Validate region: top < bottom and bottom within screen.
            self.scroll_top = t;
            self.scroll_bot = b;
        } else {
            // Invalid region parameters, default to full screen scrolling.
            self.scroll_top = 0;
            self.scroll_bot = self.height.saturating_sub(1);
            warn!("Invalid scrolling region ({}, {}), defaulting to full screen.", top, bottom);
        }

        // Per xterm behavior, DECSTBM moves the cursor to (0,0) of the new effective screen.
        // If origin mode is on, (0,0) is relative to the margins (i.e., self.scroll_top).
        // If origin mode is off, (0,0) is absolute.
        // The `Screen`'s cursor.y is always absolute. The `TerminalEmulator` translates
        // logical cursor y based on origin_mode. Here, we set the screen's cursor.
        // The `TerminalEmulator` will later sync its logical cursor from this.
        self.cursor.x = 0;
        if self.origin_mode {
            self.cursor.y = self.scroll_top; // Top of the margin.
        } else {
            self.cursor.y = 0; // Top of the screen.
        }
        trace!("Scrolling region set to ({}, {}) (0-based). Cursor moved to ({}, {}).",
               self.scroll_top, self.scroll_bot, self.cursor.x, self.cursor.y);
    }

    /// Gets a clone of the glyph at the specified `(x, y)` coordinates from the active screen.
    /// Returns a default glyph (space with default attributes) if coordinates are out of bounds.
    pub fn get_glyph(&self, x: usize, y: usize) -> Glyph {
        let grid_to_use = self.active_grid();
        if y < grid_to_use.len() && x < grid_to_use[y].len() {
            grid_to_use[y][x].clone()
        } else {
            // Use fully qualified path for DEFAULT_GLYPH as it's not imported directly.
            crate::glyph::DEFAULT_GLYPH.clone()
        }
    }

    /// Sets the glyph at the specified `(x, y)` coordinates on the active screen.
    /// Marks the line `y` as dirty. Warns if coordinates are out of bounds.
    pub fn set_glyph(&mut self, x: usize, y: usize, glyph: Glyph) {
        if y >= self.height || x >= self.width {
            warn!("set_glyph: coordinates ({},{}) out of screen bounds ({}x{})", x, y, self.width, self.height);
            return;
        }

        let grid_to_use = self.active_grid_mut();
        // This check should ideally be redundant if the above y < self.height and x < self.width pass
        // and grids are always sized correctly to self.width/self.height.
        if y < grid_to_use.len() && x < grid_to_use[y].len() {
            grid_to_use[y][x] = glyph;
            self.mark_line_dirty(y);
        } else {
            // This implies an inconsistency between self.width/height and actual grid dimensions.
            warn!("set_glyph: coordinates ({},{}) out of grid internal bounds. This indicates a bug.", x, y);
        }
    }

    /// Gets the effective absolute y-coordinate of the screen's cursor,
    /// considering the screen's current `origin_mode` and scroll margins.
    /// This is primarily for the `TerminalEmulator` to query.
    pub fn get_effective_cursor_y(&self) -> usize {
        if self.origin_mode {
            // If origin_mode is on, screen.cursor.y is already absolute but should be
            // interpreted as being within the scroll_top to scroll_bot range.
            // This function helps confirm that.
            self.cursor.y.clamp(self.scroll_top, self.scroll_bot)
        } else {
            self.cursor.y // Already absolute and within 0..height.
        }
    }

    /// Sets the screen's cursor y-coordinate.
    /// If `origin_mode` is active, `new_y_logical` is treated as relative to the
    /// scroll margins and converted to an absolute screen coordinate.
    /// Otherwise, `new_y_logical` is treated as an absolute coordinate.
    /// The resulting cursor `y` is clamped to the valid range.
    pub fn set_effective_cursor_y(&mut self, new_y_logical: usize) {
        if self.origin_mode {
            // new_y_logical is relative to scroll_top. Convert to absolute.
            let abs_y = self.scroll_top + new_y_logical;
            self.cursor.y = abs_y.clamp(self.scroll_top, self.scroll_bot);
        } else {
            self.cursor.y = new_y_logical.clamp(0, self.height.saturating_sub(1));
        }
    }

    /// Clears a segment of line `y` from `x_start` to `x_end` (exclusive)
    /// by filling it with space characters using the screen's default attributes.
    /// Marks the line `y` as dirty.
    pub fn clear_line_segment(&mut self, y: usize, x_start: usize, x_end: usize) {
        let fill_glyph = Glyph { c: ' ', attr: self.default_attributes };
        self.fill_region(y, x_start, x_end, fill_glyph);
    }

    // --- Tab stop methods ---

    /// Sets a tab stop at the given column `x`.
    pub fn set_tabstop(&mut self, x: usize) {
        if x < self.tabs.len() {
            self.tabs[x] = true;
        } else {
            warn!("set_tabstop: column {} is out of bounds for tabs (width {})", x, self.tabs.len());
        }
    }

    /// Clears tab stops based on the `mode`.
    ///
    /// # Arguments
    /// * `current_cursor_x` - The current horizontal cursor position (0-based),
    ///   used when `mode` is `TabClearMode::CurrentColumn`.
    /// * `mode` - The `TabClearMode` specifying which tab stops to clear.
    pub fn clear_tabstops(&mut self, current_cursor_x: usize, mode: TabClearMode) {
        match mode {
            TabClearMode::CurrentColumn => {
                if current_cursor_x < self.tabs.len() {
                    self.tabs[current_cursor_x] = false;
                }
            }
            TabClearMode::All => {
                self.tabs.fill(false);
            }
            TabClearMode::Unsupported => {
                 warn!("Unsupported tab clear mode used.");
            }
        }
    }

    /// Finds the next tab stop column at or after the given column `x`.
    /// Returns `None` if no further tab stops are set on the line.
    pub fn get_next_tabstop(&self, x: usize) -> Option<usize> {
        // Search for the next `true` value in self.tabs, starting from x + 1.
        self.tabs.iter().skip(x.saturating_add(1)).position(|&is_set| is_set)
            .map(|pos_after_skip| x.saturating_add(1) + pos_after_skip)
    }
}

/// A placeholder constant for the scrollback shrink logic in `resize`.
/// This determines how much larger the capacity can be than the limit
/// before `shrink_to_fit` is called. This should be tuned or made configurable.
const SOME_REASONABLE_SLACK: usize = 20; // Example value
