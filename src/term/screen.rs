// myterm/src/term/screen.rs

//! Represents the state of the terminal screen, including display grids,
//! scrollback, and styling attributes.
//!
//! This struct is responsible for managing the visual state of the terminal.
//! It handles operations like character insertion, deletion, scrolling, and resizing,
//! while tracking which lines have changed (`dirty` flags) to optimize rendering.
//! Cursor management is handled externally by `term::cursor::CursorController`.
//! Clearing operations use the `default_attributes` field of this struct, which
//! is expected to be kept in sync by `TerminalEmulator`.

use std::cmp::min;
use std::collections::VecDeque;

use crate::glyph::{Attributes, DEFAULT_GLYPH, Glyph};
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
            _ => {
                warn!("Unsupported tab clear mode value: {}", value);
                TabClearMode::Unsupported
            }
        }
    }
}

/// Represents the state of the terminal screen.
///
/// Manages the primary and alternate display grids, scrollback buffer,
/// scrolling region, tab stops, and dirty line tracking.
/// It does NOT manage the cursor's position or attributes directly;
/// that is handled by the `CursorController` in the parent `term` module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Screen {
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
    /// `TerminalEmulator` is responsible for keeping this field updated
    /// with the current SGR attributes that should apply to cleared areas.
    pub default_attributes: Attributes,
    /// Origin Mode (DECOM); affects how cursor positions are interpreted
    /// relative to the scrolling margins by the `TerminalEmulator`.
    /// `Screen` itself uses absolute coordinates for grid operations.
    /// This flag is primarily for `TerminalEmulator` to construct `ScreenContext`.
    pub origin_mode: bool,
}

impl Screen {
    /// Creates a new `Screen` with the given dimensions and scrollback limit.
    ///
    /// Initializes primary and alternate grids, the scrollback buffer,
    /// tab stops, and default attributes. All lines are initially marked dirty.
    ///
    /// # Arguments
    /// * `width` - The width of the screen in columns. Clamped to be at least 1.
    /// * `height` - The height of the screen in rows. Clamped to be at least 1.
    /// * `scrollback_limit` - The maximum number of lines to keep in the scrollback buffer.
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        // Initialize default_attributes from a global constant or a sensible default.
        // TerminalEmulator will be responsible for updating this as SGR commands are processed.
        let default_attributes = DEFAULT_GLYPH.attr;
        let default_fill_char = Glyph {
            c: ' ',
            attr: default_attributes,
        };

        trace!(
            "Creating new Screen: {}x{}, scrollback: {}",
            w, h, scrollback_limit
        );

        let grid = vec![vec![default_fill_char.clone(); w]; h];
        let alt_grid = vec![vec![default_fill_char.clone(); w]; h];
        let scrollback = VecDeque::with_capacity(scrollback_limit);

        let mut tabs = vec![false; w];
        for i in (DEFAULT_TAB_INTERVAL as usize..w).step_by(DEFAULT_TAB_INTERVAL as usize) {
            if i < tabs.len() {
                tabs[i] = true;
            }
        }

        Screen {
            grid,
            alt_grid,
            scrollback,
            scrollback_limit,
            alt_screen_active: false,
            width: w,
            height: h,
            scroll_top: 0,
            scroll_bot: h.saturating_sub(1),
            tabs,
            dirty: vec![1; h],
            default_attributes, // Store the initial default attributes
            origin_mode: false,
        }
    }

    /// Returns a mutable reference to the currently active grid (primary or alternate).
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

    /// Helper to get the glyph used for filling cleared areas.
    /// It uses a space character with the screen's `default_attributes`.
    fn get_default_fill_glyph(&self) -> Glyph {
        Glyph {
            c: ' ',
            attr: self.default_attributes,
        }
    }

    /// Fills a rectangular region of a single line `y` from `x_start` (inclusive)
    /// to `x_end` (exclusive) with the provided `fill_glyph`.
    /// Marks the line `y` as dirty. Coordinates are clamped to screen dimensions.
    /// This method is kept for cases where a specific, non-default glyph is needed for filling.
    /// For standard clearing, other methods will use `get_default_fill_glyph()`.
    ///
    /// # Arguments
    /// * `y` - The 0-based row index.
    /// * `x_start` - The starting column index (inclusive).
    /// * `x_end` - The ending column index (exclusive).
    /// * `fill_glyph` - The `Glyph` to fill the region with.
    pub fn fill_region_with_glyph(
        &mut self,
        y: usize,
        x_start: usize,
        x_end: usize,
        fill_glyph: Glyph,
    ) {
        if y >= self.height {
            warn!(
                "fill_region_with_glyph: y coordinate {} is out of bounds (height {})",
                y, self.height
            );
            return;
        }
        let width = self.width;
        let height = self.height;

        let row = match self.active_grid_mut().get_mut(y) {
            Some(r) => r,
            None => {
                warn!(
                    "fill_region_with_glyph: Failed to get row {} despite bounds check (height {}). Internal inconsistency.",
                    y, height
                );
                return;
            }
        };

        let start_clamped = min(x_start, width);
        let end_clamped = min(x_end, width);

        if start_clamped < end_clamped {
            for cell in row[start_clamped..end_clamped].iter_mut() {
                *cell = fill_glyph.clone();
            }
        }
        self.mark_line_dirty(y);
    }

    /// Scrolls the content of the defined scrolling region up by `n` lines.
    /// New lines appearing at the bottom are filled using `self.default_attributes`.
    ///
    /// # Arguments
    /// * `n` - The number of lines to scroll up.
    pub fn scroll_up_serial(&mut self, n: usize) {
        let fill_glyph = self.get_default_fill_glyph();
        // --- Rest of the logic is the same as before, but uses the locally obtained fill_glyph ---
        if self.scroll_top > self.scroll_bot || self.scroll_bot >= self.height {
            warn!(
                "scroll_up_serial: Invalid scroll region top={}, bot={}, height={}",
                self.scroll_top, self.scroll_bot, self.height
            );
            return;
        }
        let n_val = n.min(self.scroll_bot.saturating_sub(self.scroll_top) + 1);
        if n_val == 0 {
            return;
        }
        trace!(
            "Scrolling up by {} lines in region ({}, {}) with default fill",
            n_val, self.scroll_top, self.scroll_bot
        );

        let top = self.scroll_top;
        let bot = self.scroll_bot;

        if !self.alt_screen_active && top == 0 {
            for i in 0..n_val {
                if self.scrollback.len() >= self.scrollback_limit && self.scrollback_limit > 0 {
                    self.scrollback.pop_front();
                }
                if self.scrollback_limit > 0 && i < self.grid.len() {
                    self.scrollback.push_back(self.grid[i].clone());
                }
            }
        }

        let active_grid = self.active_grid_mut();
        active_grid[top..=bot].rotate_left(n_val);

        for y_idx in (bot.saturating_sub(n_val) + 1)..=bot {
            if let Some(row_to_fill) = active_grid.get_mut(y_idx) {
                row_to_fill.fill(fill_glyph.clone());
            }
        }

        for y_idx in top..=bot {
            self.mark_line_dirty(y_idx);
        }
    }

    /// Scrolls the content of the defined scrolling region down by `n` lines.
    /// New lines appearing at the top are filled using `self.default_attributes`.
    ///
    /// # Arguments
    /// * `n` - The number of lines to scroll down.
    pub fn scroll_down_serial(&mut self, n: usize) {
        let fill_glyph = self.get_default_fill_glyph();
        // --- Rest of the logic is the same as before ---
        if self.scroll_top > self.scroll_bot || self.scroll_bot >= self.height {
            warn!(
                "scroll_down_serial: Invalid scroll region top={}, bot={}, height={}",
                self.scroll_top, self.scroll_bot, self.height
            );
            return;
        }
        let n_val = n.min(self.scroll_bot.saturating_sub(self.scroll_top) + 1);
        if n_val == 0 {
            return;
        }
        trace!(
            "Scrolling down by {} lines in region ({}, {}) with default fill",
            n_val, self.scroll_top, self.scroll_bot
        );

        let top = self.scroll_top;
        let bot = self.scroll_bot;

        let active_grid = self.active_grid_mut();
        active_grid[top..=bot].rotate_right(n_val);

        for y_idx in top..(top + n_val) {
            if let Some(row_to_fill) = active_grid.get_mut(y_idx) {
                row_to_fill.fill(fill_glyph.clone());
            }
        }

        for y_idx in top..=bot {
            self.mark_line_dirty(y_idx);
        }
    }

    /// Inserts `n` blank characters (using `self.default_attributes`) in line `y` at column `x`.
    ///
    /// # Arguments
    /// * `y` - The 0-based row index.
    /// * `x` - The 0-based column index where insertion starts.
    /// * `n` - The number of blank characters to insert.
    pub fn insert_blank_chars_in_line(&mut self, y: usize, x: usize, n: usize) {
        let fill_glyph = self.get_default_fill_glyph();
        // --- Rest of the logic is the same as before ---
        if y >= self.height {
            warn!(
                "insert_blank_chars_in_line: y coordinate {} out of bounds (height {}).",
                y, self.height
            );
            return;
        }
        let width = self.width;

        if x >= width || n == 0 {
            return;
        }

        let row = match self.active_grid_mut().get_mut(y) {
            Some(r) => r,
            None => {
                warn!(
                    "insert_blank_chars_in_line: Failed to get row {} for insertion.",
                    y
                );
                return;
            }
        };

        let count = n.min(width.saturating_sub(x));
        if count == 0 {
            return;
        }

        row[x..].rotate_right(count);
        for fill_x_idx in x..(x + count) {
            if let Some(cell) = row.get_mut(fill_x_idx) {
                *cell = fill_glyph.clone();
            }
        }
        self.mark_line_dirty(y);
    }

    /// Deletes `n` characters in line `y` starting at column `x`.
    /// Space freed at the end is filled using `self.default_attributes`.
    ///
    /// # Arguments
    /// * `y` - The 0-based row index.
    /// * `x` - The 0-based column index where deletion starts.
    /// * `n` - The number of characters to delete.
    pub fn delete_chars_in_line(&mut self, y: usize, x: usize, n: usize) {
        let fill_glyph = self.get_default_fill_glyph();
        // --- Rest of the logic is the same as before ---
        if y >= self.height {
            warn!(
                "delete_chars_in_line: y coordinate {} out of bounds (height {}).",
                y, self.height
            );
            return;
        }

        let width = self.width;
        if x >= width || n == 0 {
            return;
        }

        let row = match self.active_grid_mut().get_mut(y) {
            Some(r) => r,
            None => {
                warn!(
                    "delete_chars_in_line: Failed to get row {} for deletion.",
                    y
                );
                return;
            }
        };

        let count = n.min(width.saturating_sub(x));
        if count == 0 {
            return;
        }

        row[x..].rotate_left(count);
        let fill_start_idx = width.saturating_sub(count);
        for fill_x_idx in fill_start_idx..width {
            if let Some(cell) = row.get_mut(fill_x_idx) {
                *cell = fill_glyph.clone();
            }
        }
        self.mark_line_dirty(y);
    }

    /// Resizes the screen. `TerminalEmulator` updates `self.default_attributes` separately.
    pub fn resize(&mut self, new_width: usize, new_height: usize, new_scrollback_limit: usize) {
        let nw = new_width.max(1);
        let nh = new_height.max(1);
        warn!(
            "Screen resize from {}x{} (scrollback: {}) to {}x{} (scrollback: {})",
            self.width, self.height, self.scrollback_limit, nw, nh, new_scrollback_limit
        );

        let old_width = self.width;
        self.width = nw;
        self.height = nh;
        self.scrollback_limit = new_scrollback_limit;

        // Use the *current* self.default_attributes for filling new cells.
        // TerminalEmulator is responsible for ensuring this is appropriate *before* calling resize,
        // or for updating it immediately after if the resize implies a specific default state.
        let fill_glyph = self.get_default_fill_glyph();

        self.grid.resize_with(nh, || vec![fill_glyph.clone(); nw]);
        for row_ref in self.grid.iter_mut() {
            row_ref.resize(nw, fill_glyph.clone());
        }
        self.alt_grid
            .resize_with(nh, || vec![fill_glyph.clone(); nw]);
        for row_ref in self.alt_grid.iter_mut() {
            row_ref.resize(nw, fill_glyph.clone());
        }

        if old_width != nw {
            for row_ref in self.scrollback.iter_mut() {
                row_ref.resize(nw, fill_glyph.clone());
            }
        }
        while self.scrollback.len() > self.scrollback_limit {
            self.scrollback.pop_front();
        }
        if self.scrollback.capacity() > self.scrollback_limit + SOME_REASONABLE_SLACK {
            self.scrollback.shrink_to_fit();
        }

        self.scroll_top = 0;
        self.scroll_bot = nh.saturating_sub(1);

        self.tabs = vec![false; nw];
        for i in (DEFAULT_TAB_INTERVAL as usize..nw).step_by(DEFAULT_TAB_INTERVAL as usize) {
            if i < self.tabs.len() {
                self.tabs[i] = true;
            }
        }

        self.dirty = vec![1; nh];
        trace!(
            "Screen resized. New dimensions: {}x{}. All lines marked dirty.",
            nw, nh
        );
    }

    /// Marks all lines on the screen as dirty.
    pub fn mark_all_dirty(&mut self) {
        self.dirty.fill(1);
    }

    /// Marks a single line `y` as dirty.
    pub fn mark_line_dirty(&mut self, y: usize) {
        if y < self.dirty.len() {
            self.dirty[y] = 1;
        } else {
            warn!(
                "mark_line_dirty: y coordinate {} is out of bounds for dirty flags (len {}), screen height is {}",
                y,
                self.dirty.len(),
                self.height
            );
        }
    }

    /// Clears all dirty flags.
    pub fn clear_dirty_flags(&mut self) {
        self.dirty.fill(0);
    }

    /// Switches to the alternate screen buffer.
    /// If `clear_alt_screen` is true, the alternate screen is filled using `self.default_attributes`.
    ///
    /// # Arguments
    /// * `clear_alt_screen` - If true, the alternate screen is cleared.
    pub fn enter_alt_screen(&mut self, clear_alt_screen: bool) {
        if self.alt_screen_active {
            return;
        }
        self.alt_screen_active = true;

        if clear_alt_screen {
            let fill_glyph = self.get_default_fill_glyph();
            for y_idx in 0..self.height {
                // Use fill_region_with_glyph to be explicit about the fill.
                self.fill_region_with_glyph(y_idx, 0, self.width, fill_glyph.clone());
            }
        }
        self.mark_all_dirty();
        trace!("Entered alt screen. All lines marked dirty.");
    }

    /// Switches back to the primary screen buffer.
    pub fn exit_alt_screen(&mut self) {
        if !self.alt_screen_active {
            return;
        }
        self.alt_screen_active = false;
        self.mark_all_dirty();
        trace!("Exited alt screen. All lines marked dirty.");
    }

    /// Sets the scrolling region (DECSTBM).
    pub fn set_scrolling_region(&mut self, top_1_based: usize, bottom_1_based: usize) {
        let t = top_1_based.saturating_sub(1);
        let b = bottom_1_based.saturating_sub(1);

        if t < b && b < self.height {
            self.scroll_top = t;
            self.scroll_bot = b;
        } else {
            self.scroll_top = 0;
            self.scroll_bot = self.height.saturating_sub(1);
            warn!(
                "Invalid scrolling region ({}, {}), defaulting to full screen (0-based: {}, {}). Screen height: {}",
                top_1_based, bottom_1_based, self.scroll_top, self.scroll_bot, self.height
            );
        }
        trace!(
            "Scrolling region set to (0-based: {}, {}).",
            self.scroll_top, self.scroll_bot
        );
    }

    /// Gets a clone of the glyph at the specified `(x, y)` coordinates.
    pub fn get_glyph(&self, x: usize, y: usize) -> Glyph {
        let grid_to_use = self.active_grid();
        if y < grid_to_use.len() && x < grid_to_use.get(y).map_or(0, |row| row.len()) {
            grid_to_use[y][x].clone()
        } else {
            Glyph {
                c: ' ',
                attr: self.default_attributes,
            }
        }
    }

    /// Sets the glyph at the specified `(x, y)` coordinates.
    pub fn set_glyph(&mut self, x: usize, y: usize, glyph: Glyph) {
        if y >= self.height || x >= self.width {
            warn!(
                "set_glyph: coordinates ({},{}) out of screen bounds ({}x{})",
                x, y, self.width, self.height
            );
            return;
        }
        let width = self.width;
        let height = self.height;

        let grid_to_use = self.active_grid_mut();
        if y < grid_to_use.len() && x < grid_to_use.get(y).map_or(0, |row| row.len()) {
            grid_to_use[y][x] = glyph;
            self.mark_line_dirty(y);
        } else {
            warn!(
                "set_glyph: coordinates ({},{}) out of grid internal bounds. Screen: {}x{}, Grid row {} len: {:?}",
                x,
                y,
                width,
                height,
                y,
                grid_to_use.get(y).map(|r| r.len())
            );
        }
    }

    /// Clears a segment of line `y` from `x_start` to `x_end` (exclusive)
    /// by filling it with space characters using `self.default_attributes`.
    /// Marks the line `y` as dirty.
    ///
    /// # Arguments
    /// * `y` - The 0-based row index.
    /// * `x_start` - The starting column index (inclusive).
    /// * `x_end` - The ending column index (exclusive).
    pub fn clear_line_segment(&mut self, y: usize, x_start: usize, x_end: usize) {
        let fill_glyph = self.get_default_fill_glyph();
        // Use fill_region_with_glyph to perform the fill, ensuring dirty flags are set.
        self.fill_region_with_glyph(y, x_start, x_end, fill_glyph);
    }

    // --- Tab stop methods ---

    /// Sets a tab stop at the given column `x`.
    pub fn set_tabstop(&mut self, x: usize) {
        if x < self.tabs.len() {
            self.tabs[x] = true;
        } else {
            warn!(
                "set_tabstop: column {} is out of bounds for tabs (width {})",
                x,
                self.tabs.len()
            );
        }
    }

    /// Clears tab stops based on the `mode`.
    pub fn clear_tabstops(&mut self, current_cursor_x: usize, mode: TabClearMode) {
        match mode {
            TabClearMode::CurrentColumn => {
                if current_cursor_x < self.tabs.len() {
                    self.tabs[current_cursor_x] = false;
                } else {
                    warn!(
                        "clear_tabstops (CurrentColumn): cursor_x {} out of bounds for tabs (width {})",
                        current_cursor_x,
                        self.tabs.len()
                    );
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
    pub fn get_next_tabstop(&self, x: usize) -> Option<usize> {
        self.tabs
            .iter()
            .skip(x.saturating_add(1))
            .position(|&is_set| is_set)
            .map(|pos_after_skip| x.saturating_add(1) + pos_after_skip)
    }
}

/// A placeholder constant for the scrollback shrink logic in `resize`.
const SOME_REASONABLE_SLACK: usize = 20;
