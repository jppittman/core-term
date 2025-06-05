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
use std::cmp::{max, min as std_min}; // For local min/max, renamed from std::cmp::min
use std::collections::VecDeque;

use crate::config::CONFIG;
use crate::glyph::{AttrFlags, Attributes, ContentCell, Glyph}; // Added AttrFlags
use crate::term::snapshot::{Point, Selection, SelectionMode, SelectionRange};
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
            0 => TabClearMode::CurrentColumn, // CSI Ps = 0 => Clear Current Column Tab Stop
            2 | 5 => TabClearMode::All, // CSI Ps = 2 or 5 => Clear All Tab Stops. Mode 5 is common.
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
    /// Current selection state.
    pub selection: Selection,
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
    pub fn new(width: usize, height: usize) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let default_attributes = Attributes {
            fg: CONFIG.colors.foreground,
            bg: CONFIG.colors.background,
            flags: AttrFlags::empty(),
        };
        let default_fill_char = Glyph::Single(ContentCell {
            c: ' ',
            attr: default_attributes,
        });

        let scrollback_limit_from_config = CONFIG.behavior.scrollback_lines;
        trace!(
            "Creating new Screen: {}x{}, scrollback: {}",
            w,
            h,
            scrollback_limit_from_config
        );

        let grid = vec![vec![default_fill_char.clone(); w]; h];
        let alt_grid = vec![vec![default_fill_char.clone(); w]; h];
        let scrollback = VecDeque::with_capacity(scrollback_limit_from_config);

        let mut tabs = vec![false; w];
        for i in (CONFIG.behavior.tabspaces as usize..w).step_by(CONFIG.behavior.tabspaces as usize)
        {
            if i < tabs.len() {
                tabs[i] = true;
            }
        }

        Screen {
            grid,
            alt_grid,
            scrollback,
            scrollback_limit: scrollback_limit_from_config,
            alt_screen_active: false,
            width: w,
            height: h,
            scroll_top: 0,
            scroll_bot: h.saturating_sub(1),
            tabs,
            dirty: vec![1; h],
            default_attributes,
            origin_mode: CONFIG.behavior.default_origin_mode,
            selection: Selection::default(),
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
    fn get_default_fill_glyph(&self) -> Glyph {
        Glyph::Single(ContentCell {
            c: ' ',
            attr: self.default_attributes,
        })
    }

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
        let height_for_log = self.height; // Used for logging only

        let row = match self.active_grid_mut().get_mut(y) {
            Some(r) => r,
            None => {
                warn!(
                    "fill_region_with_glyph: Failed to get row {} despite bounds check (height {}). Internal inconsistency.",
                    y, height_for_log
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

    /// Scrolls lines within the defined scrolling region upwards by `n` lines.
    /// New lines at the bottom of the region are filled with the default fill glyph.
    /// Lines scrolled off the top of the region are discarded (no scrollback).
    pub fn scroll_up_serial(&mut self, n: usize) {
        let fill_glyph = self.get_default_fill_glyph();
        if self.scroll_top > self.scroll_bot || self.scroll_bot >= self.height {
            warn!(
                "scroll_up_serial: Invalid scroll region top={}, bot={}, height={}",
                self.scroll_top, self.scroll_bot, self.height
            );
            return;
        }
        // Determine the actual number of lines to scroll, capped by region height
        let n_val = n.min(self.scroll_bot.saturating_sub(self.scroll_top) + 1);
        if n_val == 0 {
            return; // No scrolling needed
        }
        trace!(
            "Scrolling up by {} lines in region ({}, {}) with default fill (NO SCROLLBACK)",
            n_val,
            self.scroll_top,
            self.scroll_bot
        );

        let top = self.scroll_top;
        let bot = self.scroll_bot;

        // Get a mutable reference to the currently active grid
        let active_grid = self.active_grid_mut();

        // Perform the rotation: lines from `top + n_val` to `bot` move to `top` to `bot - n_val`.
        // The `rotate_left` method on a slice shifts elements to the left,
        // and elements shifted off the beginning are wrapped around to the end.
        active_grid[top..=bot].rotate_left(n_val);

        // Fill the newly vacated lines at the bottom of the scrolling region
        // The lines that were wrapped around by rotate_left are now at the end of the slice.
        // These are the lines that need to be cleared.
        for y_idx in (bot.saturating_sub(n_val) + 1)..=bot {
            if let Some(row_to_fill) = active_grid.get_mut(y_idx) {
                row_to_fill.fill(fill_glyph.clone());
            } else {
                // This should ideally not happen if bounds are correct
                warn!(
                    "scroll_up_serial: Attempted to fill non-existent row {} during scroll.",
                    y_idx
                );
            }
        }

        // Mark all lines within the scrolled region as dirty
        for y_idx in top..=bot {
            self.mark_line_dirty(y_idx);
        }
    }

    /// Scrolls lines within the defined scrolling region downwards by `n` lines.
    /// New lines at the top of the region are filled with the default fill glyph.
    pub fn scroll_down_serial(&mut self, n: usize) {
        let fill_glyph = self.get_default_fill_glyph();
        if self.scroll_top > self.scroll_bot || self.scroll_bot >= self.height {
            warn!(
                "scroll_down_serial: Invalid scroll region top={}, bot={}, height={}",
                self.scroll_top, self.scroll_bot, self.height
            );
            return;
        }
        // Determine the actual number of lines to scroll, capped by region height
        let n_val = n.min(self.scroll_bot.saturating_sub(self.scroll_top) + 1);
        if n_val == 0 {
            return; // No scrolling needed
        }
        trace!(
            "Scrolling down by {} lines in region ({}, {}) with default fill",
            n_val,
            self.scroll_top,
            self.scroll_bot
        );

        let top = self.scroll_top;
        let bot = self.scroll_bot;

        // Get a mutable reference to the currently active grid
        let active_grid = self.active_grid_mut();

        // Perform the rotation: lines from `top` to `bot - n_val` move to `top + n_val` to `bot`.
        // The `rotate_right` method on a slice shifts elements to the right,
        // and elements shifted off the end are wrapped around to the beginning.
        active_grid[top..=bot].rotate_right(n_val);

        // Fill the newly vacated lines at the top of the scrolling region.
        // These are the lines that were wrapped around by rotate_right.
        for y_idx in top..(top + n_val) {
            if let Some(row_to_fill) = active_grid.get_mut(y_idx) {
                row_to_fill.fill(fill_glyph.clone());
            } else {
                // This should ideally not happen if bounds are correct
                warn!(
                    "scroll_down_serial: Attempted to fill non-existent row {} during scroll.",
                    y_idx
                );
            }
        }

        // Mark all lines within the scrolled region as dirty
        for y_idx in top..=bot {
            self.mark_line_dirty(y_idx);
        }
    }

    pub fn insert_blank_chars_in_line(&mut self, y: usize, x: usize, n: usize) {
        let fill_glyph = self.get_default_fill_glyph();
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

    pub fn delete_chars_in_line(&mut self, y: usize, x: usize, n: usize) {
        let fill_glyph = self.get_default_fill_glyph();
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

    pub fn resize(&mut self, new_width: usize, new_height: usize) {
        let nw = new_width.max(1); // Ensure new width is at least 1
        let nh = new_height.max(1); // Ensure new height is at least 1

        // Log the resize operation
        warn!(
            "Screen resize from {}x{} to {}x{} (scrollback is NOT handled by this resize function)",
            self.width, self.height, nw, nh
        );

        // 1. Clear selection before any structural changes
        self.clear_selection();

        let old_height = self.height;
        // let old_width = self.width; // Not strictly needed if only copying row by row and then resizing rows
        let fill_glyph = self.get_default_fill_glyph();

        // 2. If scrollback field exists but is unsupported, ensure it's managed minimally.
        //    If st-like behavior means no scrollback AT ALL, this field might be vestigial.
        //    For now, we just ensure it doesn't interfere with resize.
        //    If CONFIG.behavior.scrollback_lines is 0, this should be fine.
        self.scrollback_limit = CONFIG.behavior.scrollback_lines; // Or explicitly 0 if always unsupported
        if self.scrollback_limit == 0 {
            if !self.scrollback.is_empty() {
                self.scrollback.clear();
                self.scrollback.shrink_to_fit(); // Free memory if it was used
            }
        } else {
            // If scrollback_limit > 0 but we are emulating st (no scrollback during resize)
            // we still might want to adjust width of existing scrollback lines if any exist
            // for consistency, though they wouldn't be populated by this resize.
            // This part is a bit ambiguous if scrollback is "unsupported" but the field exists.
            // For st-like resize, scrollback isn't part of the visible screen change.
            // Let's assume if scrollback_limit is > 0, we still resize its lines for width.
            if nw != self.width {
                // self.width is still old_width here
                for row_ref in self.scrollback.iter_mut() {
                    row_ref.resize(nw, fill_glyph.clone());
                }
            }
            while self.scrollback.len() > self.scrollback_limit {
                self.scrollback.pop_front();
            }
        }

        // 3. Create new primary grid and copy content
        //    Content is anchored to the top-left.
        let mut new_primary_grid = vec![vec![fill_glyph.clone(); nw]; nh];
        for y in 0..std_min(old_height, nh) {
            // Check if the row y exists in the old grid
            if let Some(old_row_content) = self.grid.get(y) {
                let mut new_row = vec![fill_glyph.clone(); nw];
                for x in 0..std_min(old_row_content.len(), nw) {
                    new_row[x] = old_row_content[x].clone();
                }
                // new_primary_grid[y] is guaranteed to exist due to initialization size
                new_primary_grid[y] = new_row;
            }
            // If old_height > nh, lines beyond nh-1 are truncated.
            // If old_height < nh, new lines at the bottom remain as fill_glyph.
        }
        self.grid = new_primary_grid;

        // 4. Create new alternate grid and copy content (similarly)
        let mut new_alt_grid = vec![vec![fill_glyph.clone(); nw]; nh];
        for y in 0..std_min(old_height, nh) {
            if let Some(old_row_content) = self.alt_grid.get(y) {
                let mut new_row = vec![fill_glyph.clone(); nw];
                for x in 0..std_min(old_row_content.len(), nw) {
                    new_row[x] = old_row_content[x].clone();
                }
                new_alt_grid[y] = new_row;
            }
        }
        self.alt_grid = new_alt_grid;

        // 5. Update screen dimensions and related properties
        self.width = nw;
        self.height = nh;

        // Reset scrolling region to full screen (typical for st-like behavior on resize)
        self.scroll_top = 0;
        self.scroll_bot = nh.saturating_sub(1);

        // Reinitialize tab stops for the new width
        self.tabs = vec![false; nw];
        for i in
            (CONFIG.behavior.tabspaces as usize..nw).step_by(CONFIG.behavior.tabspaces as usize)
        {
            if i < self.tabs.len() {
                self.tabs[i] = true;
            }
        }

        // Mark all lines in the new grid as dirty
        self.dirty = vec![1; nh];

        trace!(
            "Screen resized (st-like, no scrollback handling in resize). New dimensions: {}x{}. All lines marked dirty.",
            nw,
            nh
        );
    }

    pub fn mark_all_dirty(&mut self) {
        self.dirty.fill(1);
    }

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

    pub fn enter_alt_screen(&mut self, clear_alt_screen: bool) {
        if self.alt_screen_active {
            return;
        }
        self.clear_selection();
        self.alt_screen_active = true;

        if clear_alt_screen {
            let fill_glyph = self.get_default_fill_glyph();
            for y_idx in 0..self.height {
                self.fill_region_with_glyph(y_idx, 0, self.width, fill_glyph.clone());
            }
        }
        self.mark_all_dirty();
        trace!("Entered alt screen. All lines marked dirty.");
    }

    pub fn exit_alt_screen(&mut self) {
        if !self.alt_screen_active {
            return;
        }
        self.clear_selection();
        self.alt_screen_active = false;
        self.mark_all_dirty();
        trace!("Exited alt screen. All lines marked dirty.");
    }

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
            self.scroll_top,
            self.scroll_bot
        );
    }

    pub fn set_glyph(&mut self, x: usize, y: usize, glyph: Glyph) {
        if y >= self.height || x >= self.width {
            warn!(
                "set_glyph: coordinates ({},{}) out of screen bounds ({}x{})",
                x, y, self.width, self.height
            );
            return;
        }
        let width_for_log = self.width;
        let height_for_log = self.height;

        let grid_to_use = self.active_grid_mut();
        if y < grid_to_use.len() && x < grid_to_use.get(y).map_or(0, |row| row.len()) {
            grid_to_use[y][x] = glyph;
            self.mark_line_dirty(y);
        } else {
            warn!(
                "set_glyph: coordinates ({},{}) out of grid internal bounds. Screen: {}x{}, Grid row {} len: {:?}",
                x,
                y,
                width_for_log,
                height_for_log,
                y,
                grid_to_use.get(y).map(|r| r.len())
            );
        }
    }

    pub fn clear_line_segment(&mut self, y: usize, x_start: usize, x_end: usize) {
        let fill_glyph = self.get_default_fill_glyph();
        self.fill_region_with_glyph(y, x_start, x_end, fill_glyph);
    }

    // --- Tab stop methods ---
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

    pub fn get_next_tabstop(&self, x: usize) -> Option<usize> {
        self.tabs
            .iter()
            .skip(x.saturating_add(1))
            .position(|&is_set| is_set)
            .map(|pos_after_skip| x.saturating_add(1) + pos_after_skip)
    }

    // --- Selection methods ---

    /// Marks lines within the current selection range as dirty.
    /// This is an internal helper called when selection changes to ensure
    /// the visual representation of the selection is updated.
    fn mark_dirty_for_selection(&mut self) {
        if let Some(range) = &self.selection.range {
            let top_row = std_min(range.start.y, range.end.y);
            let bottom_row = max(range.start.y, range.end.y);
            for y in top_row..=bottom_row {
                if y < self.height {
                    self.mark_line_dirty(y);
                }
            }
        }
    }

    /// Starts a new text selection or replaces an existing one.
    ///
    /// When a new selection is started, any previous selection is marked dirty
    /// for re-rendering. The new selection is initialized with the given `point`
    /// as both its start and end, set to the specified `mode`, and marked as active.
    /// The line(s) covered by this initial single-point selection are also marked dirty.
    ///
    /// # Arguments
    /// * `point` - The starting `Point` (column and row) of the selection.
    /// * `mode` - The `SelectionMode` (e.g., `Normal`, `Block`) for the new selection.
    pub fn start_selection(&mut self, point: Point, mode: SelectionMode) {
        if self.selection.range.is_some() {
            self.mark_dirty_for_selection();
        }

        self.selection = Selection {
            range: Some(SelectionRange {
                start: point,
                end: point,
            }),
            mode,
            is_active: true,
        };
        self.mark_dirty_for_selection();
        trace!(
            "Selection started at ({}, {}) with mode {:?}. Active: {}",
            point.x,
            point.y,
            mode,
            self.selection.is_active
        );
    }

    /// Updates the end point of the current active selection.
    ///
    /// If no selection is currently active (i.e., `selection.is_active` is `false`),
    /// this function does nothing. Otherwise, it updates the selection's
    /// end point to the given `point`. Both the previously selected region (before
    /// this update) and the newly defined region (after this update) are marked dirty
    /// to ensure correct re-rendering. The selection remains active.
    ///
    /// # Arguments
    /// * `point` - The new end `Point` for the selection.
    pub fn update_selection(&mut self, point: Point) {
        if !self.selection.is_active {
            return;
        }
        self.mark_dirty_for_selection();
        if let Some(range) = &mut self.selection.range {
            range.end = point;
        }
        self.mark_dirty_for_selection();
        if let Some(range) = &self.selection.range {
            trace!(
                "Selection updated. End point: ({}, {}). Active: {}",
                range.end.x,
                range.end.y,
                self.selection.is_active
            );
        }
    }

    /// Deactivates the current selection.
    ///
    /// This sets `selection.is_active` to `false`, indicating that the selection
    /// process (e.g., mouse drag) has ended. The selection's coordinates (`start`, `end`, `mode`)
    /// are preserved, allowing the selection to remain visually highlighted until
    /// it's explicitly cleared or a new selection is started.
    /// No lines are marked dirty by this action itself, as the visual state of the
    /// selection highlight does not change upon deactivation.
    pub fn end_selection(&mut self) {
        if self.selection.is_active {
            self.selection.is_active = false;
            trace!("Selection ended. Active: {}", self.selection.is_active);
        }
    }

    /// Clears the current selection entirely and marks the previously selected area as dirty.
    ///
    /// This resets the selection state to its default (no selection active, no start/end points).
    /// If a selection was present before clearing, the lines it covered are marked dirty
    /// to ensure the selection highlighting is removed upon the next render.
    pub fn clear_selection(&mut self) {
        if self.selection.range.is_some() {
            self.mark_dirty_for_selection();
        }
        self.selection = Selection::default();
        trace!("Selection cleared.");
    }

    /// Checks if a given grid cell `point` is part of the current selection.
    ///
    /// This is primarily used for rendering to determine if a cell should be
    /// highlighted.
    ///
    /// # Arguments
    /// * `point` - The grid cell coordinates (`x` for column, `y` for row) to check.
    ///
    /// # Returns
    /// `true` if the cell is selected, `false` otherwise.
    pub fn is_selected(&self, point: Point) -> bool {
        if point.x >= self.width || point.y >= self.height {
            return false;
        }

        let Some(range) = &self.selection.range else {
            return false;
        };

        let raw_start = range.start;
        let raw_end = range.end;

        match self.selection.mode {
            SelectionMode::Cell => {
                // Normalize selection points: (start_y, start_x) should be top-left of the virtual box.
                // However, for 'Cell' (normal) selection, the actual start/end points matter for line-wise logic.
                let (sel_start_y, sel_end_y) = if raw_start.y <= raw_end.y {
                    (raw_start.y, raw_end.y)
                } else {
                    (raw_end.y, raw_start.y)
                };

                // Point is outside the vertical span of the selection.
                if point.y < sel_start_y || point.y > sel_end_y {
                    return false;
                }

                // Single-line selection
                if raw_start.y == raw_end.y {
                    let line_min_x = std_min(raw_start.x, raw_end.x);
                    let line_max_x = max(raw_start.x, raw_end.x);
                    return point.x >= line_min_x && point.x <= line_max_x;
                }

                // Multi-line selection:
                // Check if the point is on the start line of the selection.
                if point.y == raw_start.y {
                    return if raw_start.y < raw_end.y {
                        // Selection goes downwards
                        point.x >= raw_start.x
                    } else {
                        // Selection goes upwards (raw_start_y > raw_end_y)
                        point.x <= raw_start.x
                    };
                }
                // Check if the point is on the end line of the selection.
                if point.y == raw_end.y {
                    return if raw_start.y < raw_end.y {
                        // Selection goes downwards
                        point.x <= raw_end.x
                    } else {
                        // Selection goes upwards
                        point.x >= raw_end.x
                    };
                }
                // Point is on a line between the start and end lines.
                // In 'Cell' mode, this means the entire line is selected.
                return true;
            } // TODO: Implement other selection modes like Block if they are added.
              // For now, Block, SemanticLine, SemanticWord are commented out or not present
              // in the SelectionMode enum used by this logic.
              // e.g., SelectionMode::Block => {
              //     let min_x = std_min(raw_start.x, raw_end.x);
              //     let max_x = max(raw_start.x, raw_end.x);
              //     let min_y = std_min(raw_start.y, raw_end.y);
              //     let max_y = max(raw_start.y, raw_end.y);
              //     return point.x >= min_x
              //         && point.x <= max_x
              //         && point.y >= min_y
              //         && point.y <= max_y;
              // }
              // SelectionMode::SemanticLine | SelectionMode::SemanticWord => {
              //     // For semantic selections, is_selected might behave like Normal or Block
              //     // depending on how the range was defined by the semantic logic.
              //     // Assuming for now it behaves like Normal for highlighting purposes.
              //     let (box_start_y, box_end_y) = if raw_start.y <= raw_end.y {
              //         (raw_start.y, raw_end.y)
              //     } else {
              //         (raw_end.y, raw_start.y)
              //     };

              //     if point.y < box_start_y || point.y > box_end_y {
              //         return false;
              //     }

              //     if raw_start.y == raw_end.y {
              //         let line_min_x = std_min(raw_start.x, raw_end.x);
              //         let line_max_x = max(raw_start.x, raw_end.x);
              //         return point.x >= line_min_x && point.x <= line_max_x;
              //     }
              //      // For multi-line semantic (like line selection), assume full lines are selected
              //     return true;
              // }
        }
    }

    /// Retrieves the text content of the current selection.
    ///
    /// This method currently primarily handles `SelectionMode::Cell` (similar to typical
    /// terminal "normal" or "stream" selection). Text is ordered logically from the
    /// selection start to end (top-left to bottom-right after normalization).
    ///
    /// For `SelectionMode::Cell`, it attempts to replicate common terminal behavior regarding
    /// line endings and the trimming of trailing whitespace from lines that are fully selected
    /// but are not the last line of the selection.
    ///
    /// Other modes like `Block` selection are not implemented in this function.
    ///
    /// # Returns
    /// An `Option<String>` containing the selected text, or `None` if there's
    /// no valid selection or the selection is empty.
    pub fn get_selected_text(&self) -> Option<String> {
        let Some(range) = &self.selection.range else {
            return None; // No active selection range.
        };

        // Normalize selection points so norm_start_point is visually above or to the left of norm_end_point.
        let (norm_start_point, norm_end_point) = if range.start.y > range.end.y
            || (range.start.y == range.end.y && range.start.x > range.end.x)
        {
            (range.end, range.start)
        } else {
            (range.start, range.end)
        };

        let mut selected_text_buffer = String::new();
        let grid_to_use = self.active_grid();

        match self.selection.mode {
            SelectionMode::Cell => {
                for y_abs in norm_start_point.y..=norm_end_point.y {
                    if y_abs >= grid_to_use.len() {
                        continue;
                    } // Should not happen with valid points

                    let current_row_glyphs = &grid_to_use[y_abs];
                    let mut current_line_text = String::new();

                    // Determine the start and end columns for text extraction on the current line.
                    let line_col_start = if y_abs == norm_start_point.y {
                        norm_start_point.x
                    } else {
                        0
                    };
                    let line_col_end = if y_abs == norm_end_point.y {
                        norm_end_point.x
                    } else {
                        self.width.saturating_sub(1)
                    };

                    // Ensure line_col_end does not exceed grid width for safety, though std_min handles this too.
                    let effective_line_col_end =
                        std_min(line_col_end, self.width.saturating_sub(1));

                    if line_col_start > effective_line_col_end
                        && !(line_col_start == 0 && effective_line_col_end == 0 && self.width == 0)
                    {
                        // This handles cases like zero-width selection or start beyond end on a line.
                        // If it's a multi-line selection, we still need the newline.
                        if y_abs < norm_end_point.y {
                            selected_text_buffer.push('\n');
                        }
                        continue;
                    }

                    for x_abs in line_col_start..=effective_line_col_end {
                        if x_abs < current_row_glyphs.len() {
                            current_line_text.push(current_row_glyphs[x_abs].display_char());
                        } else {
                            // If selection extends beyond available glyphs on the line, append space.
                            current_line_text.push(' ');
                        }
                    }

                    // Trim trailing whitespace if:
                    // 1. It's a multi-line selection (norm_start_point.y != norm_end_point.y)
                    // 2. This is NOT the last line of the multi-line selection (y_abs < norm_end_point.y)
                    // 3. The selection on this line went all the way to the end of the screen width (line_col_end included self.width - 1)
                    if norm_start_point.y != norm_end_point.y && y_abs < norm_end_point.y {
                        if line_col_end >= self.width.saturating_sub(1) {
                            // Check if selection extended to line end
                            if let Some(last_char_idx) = current_line_text.rfind(|c: char| c != ' ')
                            {
                                current_line_text.truncate(last_char_idx + 1);
                            } else {
                                current_line_text.clear(); // Line was all spaces
                            }
                        }
                    }

                    selected_text_buffer.push_str(&current_line_text);
                    if y_abs < norm_end_point.y {
                        selected_text_buffer.push('\n');
                    }
                }
            } // TODO: Implement Block selection text retrieval if/when that mode is fully supported.
              // SelectionMode::Block => {
              //     let min_x = std_min(start_point.x, end_point.x);
              //     let max_x = max(start_point.x, end_point.x);

              //     for y in norm_start_point.y..=norm_end_point.y {
              //         if y >= grid_to_use.len() { continue; }
              //         let current_row_glyphs = &grid_to_use[y];
              //         let mut current_line_text = String::new();

              //         for x in min_x..=max_x {
              //             if x < current_row_glyphs.len() {
              //                 current_line_text.push(current_row_glyphs[x].c);
              //             } else {
              //                 current_line_text.push(' ');
              //             }
              //         }
              //         selected_text_buffer.push_str(&current_line_text);
              //         if y < norm_end_point.y {
              //             selected_text_buffer.push('\n');
              //         }
              //     }
              // }
        }

        if selected_text_buffer.is_empty() {
            None
        } else {
            Some(selected_text_buffer)
        }
    }
}

const SOME_REASONABLE_SLACK: usize = 20;

#[cfg(test)]
mod tests {
    use super::{
        Attributes,
        ContentCell,
        Glyph,
        Point,
        Screen,
        Selection,
        SelectionMode, // Removed AttrFlags
        SelectionRange,
    }; // Added ContentCell

    // Helper function for tests, using a fixed scrollback for test predictability
    // or allowing tests to specify if some need to test scrollback behavior explicitly.
    fn create_test_screen_with_scrollback(
        width: usize,
        height: usize,
        scrollback: usize,
    ) -> Screen {
        // Temporarily override config for consistent tests if Screen::new now solely relies on CONFIG
        // This is tricky. For now, let's assume tests might need to adapt or Screen::new might need
        // a test-specific constructor or tests mock CONFIG (which is hard with static Lazy).
        // For now, let's assume Screen::new is adapted and we pass what's needed.
        // The current Screen::new was changed to NOT take scrollback_limit.
        // So, test utility must change.
        let mut screen = Screen::new(width, height);
        // If tests need to check scrollback functionality with specific limits,
        // they might need to manipulate CONFIG or this helper needs to be very creative.
        // For simplicity, we'll let it use the default from CONFIG for now,
        // meaning scrollback related tests might become less predictable if CONFIG changes.
        // OR, we acknowledge that create_test_screen was passing 0, so we can try to
        // preserve that behavior if it's critical for existing tests not focusing on scrollback.
        screen.scrollback_limit = scrollback; // Manually set for test purposes if needed.
        screen
    }

    fn create_test_screen(width: usize, height: usize) -> Screen {
        // Most tests used 0, implying scrollback wasn't their focus or they expected no scrollback.
        create_test_screen_with_scrollback(width, height, 0)
    }

    fn fill_screen_with_pattern(screen: &mut Screen) {
        for r in 0..screen.height {
            for c in 0..screen.width {
                let char_val =
                    char::from_u32(('a' as u32) + (c % 26) as u32 + (r % 3) as u32).unwrap_or('?');
                screen.grid[r][c] = Glyph::Single(ContentCell {
                    c: char_val,
                    attr: Attributes::default(),
                });
            }
        }
    }

    #[test]
    fn test_selection_default_state() {
        let screen = create_test_screen(10, 5);
        assert_eq!(screen.selection, Selection::default());
    }

    #[test]
    fn test_start_selection() {
        let mut screen = create_test_screen(10, 5);
        let start_point = Point { x: 1, y: 1 };
        screen.dirty.fill(0);
        screen.start_selection(start_point, SelectionMode::Cell); // Replaced Normal with Cell
        assert_eq!(
            screen.selection.range,
            Some(SelectionRange {
                start: start_point,
                end: start_point
            })
        );
        assert!(screen.selection.is_active);
        assert_eq!(screen.dirty[start_point.y], 1);
    }

    #[test]
    fn test_update_selection() {
        let mut screen = create_test_screen(10, 5);
        let start_point = Point { x: 1, y: 1 };
        let update_point = Point { x: 5, y: 2 };
        screen.start_selection(start_point, SelectionMode::Cell); // Replaced Normal with Cell
        screen.dirty.fill(0);
        screen.mark_line_dirty(start_point.y);
        screen.update_selection(update_point);
        assert_eq!(screen.selection.range.map(|r| r.end), Some(update_point));
        assert!(screen.selection.is_active);
        assert_eq!(screen.dirty[update_point.y], 1);
    }

    #[test]
    fn test_update_selection_marks_old_and_new_lines_dirty() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 1, y: 1 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 3, y: 1 });
        screen.dirty.fill(0);
        screen.update_selection(Point { x: 5, y: 2 });
        assert_eq!(screen.dirty[1], 1);
        assert_eq!(screen.dirty[2], 1);
    }

    #[test]
    fn test_update_selection_when_not_active() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 1, y: 1 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.selection.is_active = false;
        let original_selection_state = screen.selection.clone();
        screen.update_selection(Point { x: 5, y: 2 });
        assert_eq!(screen.selection, original_selection_state);
    }

    #[test]
    fn test_end_selection() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 1, y: 1 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.end_selection();
        assert!(!screen.selection.is_active);
    }

    #[test]
    fn test_clear_selection() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 1, y: 1 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 3, y: 2 });
        screen.dirty.fill(0);
        screen.clear_selection();
        assert_eq!(screen.selection, Selection::default());
        assert_eq!(screen.dirty[1], 1);
        assert_eq!(screen.dirty[2], 1);
    }

    #[test]
    fn test_is_selected_normal_no_selection() {
        let screen = create_test_screen(10, 5);
        assert!(!screen.is_selected(Point { x: 1, y: 1 }));
    }

    #[test]
    fn test_is_selected_normal_single_line() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 2, y: 1 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 5, y: 1 });
        assert!(!screen.is_selected(Point { x: 1, y: 1 }));
        assert!(screen.is_selected(Point { x: 2, y: 1 }));
        assert!(screen.is_selected(Point { x: 5, y: 1 }));
        assert!(!screen.is_selected(Point { x: 6, y: 1 }));
    }

    #[test]
    fn test_is_selected_normal_multi_line() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 3, y: 1 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 2, y: 3 });
        assert!(screen.is_selected(Point { x: 4, y: 1 }));
        assert!(screen.is_selected(Point { x: 1, y: 2 }));
        assert!(screen.is_selected(Point { x: 1, y: 3 }));
        assert!(!screen.is_selected(Point { x: 2, y: 1 }));
        assert!(!screen.is_selected(Point { x: 3, y: 3 }));
    }

    #[test]
    fn test_is_selected_normal_multi_line_selection_ends_at_width_minus_1() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 8, y: 0 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 2, y: 2 });
        assert!(screen.is_selected(Point { x: 9, y: 0 }));
        assert!(screen.is_selected(Point { x: 0, y: 1 }));
        assert!(screen.is_selected(Point { x: 9, y: 1 }));
        assert!(screen.is_selected(Point { x: 0, y: 2 }));
        assert!(screen.is_selected(Point { x: 2, y: 2 }));
    }

    #[test]
    fn test_is_selected_normal_reverse_selection_points() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 5, y: 2 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 1, y: 1 });
        assert!(screen.is_selected(Point { x: 1, y: 1 }));
        assert!(screen.is_selected(Point { x: 3, y: 1 }));
        assert!(screen.is_selected(Point { x: 5, y: 1 }));
        assert!(screen.is_selected(Point { x: 1, y: 2 }));
        assert!(screen.is_selected(Point { x: 3, y: 2 }));
        assert!(screen.is_selected(Point { x: 5, y: 2 }));
    }

    #[test]
    fn test_is_selected_point_equals_start_or_end() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 2, y: 2 }, SelectionMode::Cell); // Replaced Normal with Cell
        assert!(screen.is_selected(Point { x: 2, y: 2 }));
        screen.update_selection(Point { x: 4, y: 2 });
        assert!(screen.is_selected(Point { x: 2, y: 2 }));
        assert!(screen.is_selected(Point { x: 4, y: 2 }));
    }

    #[test]
    fn test_is_selected_out_of_bounds_point() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 0, y: 0 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point {
            x: screen.width - 1,
            y: screen.height - 1,
        });
        assert!(!screen.is_selected(Point {
            x: screen.width,
            y: 0
        }));
        assert!(!screen.is_selected(Point {
            x: 0,
            y: screen.height
        }));
    }

    // Commenting out Block tests as Block mode is not defined
    // #[test]
    // fn test_is_selected_block_no_selection() {
    //     let screen = create_test_screen(10, 5);
    //     assert!(!screen.is_selected(Point { x: 1, y: 1 }));
    // }

    // #[test]
    // fn test_is_selected_block_simple() {
    //     let mut screen = create_test_screen(10, 5);
    //     screen.start_selection(Point { x: 1, y: 1 }, SelectionMode::Block);
    //     screen.update_selection(Point { x: 3, y: 3 });
    //     assert!(screen.is_selected(Point { x: 2, y: 2 }));
    //     assert!(!screen.is_selected(Point { x: 0, y: 2 }));
    // }

    // #[test]
    // fn test_is_selected_block_reverse_points() {
    //     let mut screen = create_test_screen(10, 5);
    //     screen.start_selection(Point { x: 3, y: 3 }, SelectionMode::Block);
    //     screen.update_selection(Point { x: 1, y: 1 });
    //     assert!(screen.is_selected(Point { x: 2, y: 2 }));
    // }

    #[test]
    fn test_get_selected_text_normal_no_selection() {
        let screen = create_test_screen(10, 5);
        assert_eq!(screen.get_selected_text(), None);
    }

    #[test]
    fn test_get_selected_text_normal_single_char() {
        let mut screen = create_test_screen(5, 3);
        fill_screen_with_pattern(&mut screen);
        screen.start_selection(Point { x: 1, y: 1 }, SelectionMode::Cell); // Replaced Normal with Cell
        assert_eq!(screen.get_selected_text(), Some("c".to_string()));
    }

    #[test]
    fn test_get_selected_text_normal_single_line_partial() {
        let mut screen = create_test_screen(5, 3);
        fill_screen_with_pattern(&mut screen);
        screen.start_selection(Point { x: 1, y: 0 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 3, y: 0 });
        assert_eq!(screen.get_selected_text(), Some("bcd".to_string()));
    }

    #[test]
    fn test_get_selected_text_normal_single_line_full() {
        let mut screen = create_test_screen(5, 3);
        fill_screen_with_pattern(&mut screen);
        screen.start_selection(Point { x: 0, y: 0 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point {
            x: screen.width - 1,
            y: 0,
        });
        assert_eq!(screen.get_selected_text(), Some("abcde".to_string()));
    }

    #[test]
    fn test_get_selected_text_normal_multi_line() {
        let mut screen = create_test_screen(3, 3);
        fill_screen_with_pattern(&mut screen);
        screen.start_selection(Point { x: 1, y: 0 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 1, y: 2 });
        assert_eq!(screen.get_selected_text(), Some("bc\nbcd\ncd".to_string()));
    }

    #[test]
    fn test_get_selected_text_normal_multi_line_reversed_points() {
        let mut screen = create_test_screen(3, 3);
        fill_screen_with_pattern(&mut screen);
        screen.start_selection(Point { x: 1, y: 2 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 1, y: 0 });
        assert_eq!(screen.get_selected_text(), Some("bc\nbcd\ncd".to_string()));
    }

    #[test]
    fn test_get_selected_text_normal_trailing_spaces_behavior() {
        let mut screen = create_test_screen(5, 2);
        screen.grid[0][0] = Glyph::Single(ContentCell {
            c: 'a',
            attr: Attributes::default(),
        });
        screen.grid[0][1] = Glyph::Single(ContentCell {
            c: 'a',
            attr: Attributes::default(),
        });
        screen.grid[0][2] = Glyph::Single(ContentCell {
            c: ' ',
            attr: Attributes::default(),
        });
        screen.grid[0][3] = Glyph::Single(ContentCell {
            c: ' ',
            attr: Attributes::default(),
        });
        screen.grid[0][4] = Glyph::Single(ContentCell {
            c: ' ',
            attr: Attributes::default(),
        });
        screen.grid[1][0] = Glyph::Single(ContentCell {
            c: 'b',
            attr: Attributes::default(),
        });
        screen.grid[1][1] = Glyph::Single(ContentCell {
            c: 'b',
            attr: Attributes::default(),
        });
        screen.grid[1][2] = Glyph::Single(ContentCell {
            c: ' ',
            attr: Attributes::default(),
        });
        screen.grid[1][3] = Glyph::Single(ContentCell {
            c: ' ',
            attr: Attributes::default(),
        });
        screen.grid[1][4] = Glyph::Single(ContentCell {
            c: ' ',
            attr: Attributes::default(),
        });

        screen.start_selection(Point { x: 0, y: 0 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 4, y: 0 });
        assert_eq!(screen.get_selected_text(), Some("aa   ".to_string()));

        screen.start_selection(Point { x: 0, y: 0 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 1, y: 1 });
        assert_eq!(screen.get_selected_text(), Some("aa\nbb".to_string()));
    }

    // Commenting out Block tests as Block mode is not defined
    // #[test]
    // fn test_get_selected_text_block_no_selection() {
    //     let mut screen = create_test_screen(10, 5);
    //     screen.selection.mode = SelectionMode::Block;
    //     assert_eq!(screen.get_selected_text(), None);
    // }

    // #[test]
    // fn test_get_selected_text_block_simple() {
    //     let mut screen = create_test_screen(5, 4);
    //     fill_screen_with_pattern(&mut screen);
    //     screen.start_selection(Point { x: 1, y: 0 }, SelectionMode::Block);
    //     screen.update_selection(Point { x: 3, y: 2 });
    //     assert_eq!(screen.get_selected_text(), Some("bcd\ncde\ndef".to_string()));
    // }

    // #[test]
    // fn test_get_selected_text_block_reversed_points() {
    //     let mut screen = create_test_screen(5, 4);
    //     fill_screen_with_pattern(&mut screen);
    //     screen.start_selection(Point { x: 3, y: 2 }, SelectionMode::Block);
    //     screen.update_selection(Point { x: 1, y: 0 });
    //     assert_eq!(screen.get_selected_text(), Some("bcd\ncde\ndef".to_string()));
    // }

    // #[test]
    // fn test_get_selected_text_block_one_column() {
    //     let mut screen = create_test_screen(5, 4);
    //     fill_screen_with_pattern(&mut screen);
    //     screen.start_selection(Point { x: 1, y: 0 }, SelectionMode::Block);
    //     screen.update_selection(Point { x: 1, y: 2 });
    //     assert_eq!(screen.get_selected_text(), Some("b\nc\nd".to_string()));
    // }

    // #[test]
    // fn test_get_selected_text_block_one_row() {
    //     let mut screen = create_test_screen(5, 4);
    //     fill_screen_with_pattern(&mut screen);
    //     screen.start_selection(Point { x: 1, y: 1 }, SelectionMode::Block);
    //     screen.update_selection(Point { x: 3, y: 1 });
    //     assert_eq!(screen.get_selected_text(), Some("cde".to_string()));
    // }

    // #[test]
    // fn test_get_selected_text_block_beyond_line_length() {
    //     let mut screen = create_test_screen(3, 2);
    //     screen.grid[0][0] = Glyph::Single(ContentCell { c: 'a', attr: Attributes::default() });
    //     screen.grid[0][1] = Glyph::Single(ContentCell { c: ' ', attr: Attributes::default() });
    //     screen.grid[0][2] = Glyph::Single(ContentCell { c: ' ', attr: Attributes::default() });
    //     screen.grid[1][0] = Glyph::Single(ContentCell { c: 'b', attr: Attributes::default() });
    //     screen.grid[1][1] = Glyph::Single(ContentCell { c: ' ', attr: Attributes::default() });
    //     screen.grid[1][2] = Glyph::Single(ContentCell { c: ' ', attr: Attributes::default() });
    //     screen.start_selection(Point { x: 0, y: 0 }, SelectionMode::Block);
    //     screen.update_selection(Point { x: 1, y: 1 });
    //     assert_eq!(screen.get_selected_text(), Some("a \nb ".to_string()));
    //     screen.start_selection(Point { x: 0, y: 0 }, SelectionMode::Block);
    //     screen.update_selection(Point { x: 2, y: 1 });
    //     assert_eq!(screen.get_selected_text(), Some("a  \nb  ".to_string()));
    // }

    #[test]
    fn test_selection_cleared_on_resize() {
        let mut screen = create_test_screen(10, 5);
        screen.start_selection(Point { x: 1, y: 1 }, SelectionMode::Cell); // Replaced Normal with Cell
        screen.update_selection(Point { x: 5, y: 2 });
        assert!(screen.selection.range.is_some());
        screen.resize(20, 10); // Removed scrollback limit argument
        assert_eq!(screen.selection, Selection::default());
    }
}
