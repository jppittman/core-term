# Design: Advanced Font Rendering & Fallback

**Version:** 1.0
**Status:** Proposed

## 1. Overview & Goal

The `core-term` project currently faces an issue where certain Unicode characters (e.g., Braille patterns) fail to render, appearing as invisible cells. This is caused by a simplified rendering pipeline that relies on a single font file.

The goal of this refactor is to implement a robust, `st`-like font fallback mechanism. This will allow the terminal to find and use glyphs from any appropriate font installed on the system, ensuring correct rendering of a wide range of Unicode characters in all styles (regular, bold, italic).

## 2. Architectural Principles

This design adheres to the core principles of the `core-term` project:

* **Separation of Concerns:** The platform-agnostic `Renderer` will continue to produce abstract drawing commands. All platform-specific font discovery and rendering logic will be encapsulated entirely within the platform-specific `Graphics` driver (e.g., for X11).
* **Testability:** The design must allow for the complex logic of the `Graphics` driver to be unit-tested in isolation, without depending on a live X server or system font configuration.
* **Cross-Platform Foundation:** The architecture will establish a clear pattern (the "`FontManager`") that can be replicated on other platforms like macOS or Wayland, fulfilling the project's goal of being an "extendible core terminal".

## 3. Core Components

### 3.1. `FontManager` Trait

To enable testing and provide a clear blueprint for other platforms, we will introduce a `FontManager` trait.

* **Purpose:** Defines a platform-agnostic interface for a component that can resolve any character into a renderable glyph.
* **Location:** A new file, e.g., `src/platform/font_manager.rs`.
* **Key Structs & Methods:**

    ```rust
    // A platform-agnostic representation of a found glyph.
    pub struct ResolvedGlyph {
        pub glyph_id: u32,       // The platform-specific ID for the glyph (e.g., Xft's glyph index).
        pub font_id: usize,      // An internal ID we assign to the font it belongs to.
    }

    // The trait defining the contract for a font manager.
    pub trait FontManager {
        /// Creates a new manager, loading primary fonts from the config.
        fn new(config: &FontConfig) -> Result<Self> where Self: Sized;

        /// The core function: finds a glyph for a given character and its attributes.
        fn get_glyph(&mut self, character: char, attributes: AttrFlags) -> Option<ResolvedGlyph>;

        /// Retrieves a platform-specific font handle using our internal ID.
        fn get_font_handle(&self, font_id: usize) -> /* e.g., *mut xft::XftFont */ ;
    }
    ```

### 3.2. `X11FontManager` Concrete Implementation

This will be the X11-specific implementation of the `FontManager` trait.

* **Purpose:** To encapsulate all `fontconfig` and `Xft` logic for finding and caching fonts.
* **Location:** Within the X11 backend, e.g., `src/platform/backends/x11/font_manager.rs`.
* **Responsibilities:**
    * Load the four primary font styles (regular, bold, italic, bold-italic) from the configuration on initialization.
    * Maintain an internal cache of fallback fonts that have been discovered.
    * Implement the `get_glyph` method by searching for a character first in the primary style font, then the fallback cache, and finally by using `fontconfig` to find a new system font if necessary.

### 3.3. `Graphics` Driver

The existing `Graphics` struct in `src/platform/backends/x11/graphics.rs` will be refactored.

* **Ownership:** It will own an instance of `X11FontManager`. It will no longer own `SafeXftFont`s directly.
* **Consumer Role:** It becomes the sole consumer of the `X11FontManager`.

## 4. Refined Rendering Data Flow

The new data flow ensures a clean separation of concerns:

1.  The **`Renderer`** processes the `RenderSnapshot` and produces a `Vec<RenderCommand>`. The `DrawTextRun` command remains abstract, containing only a `String` and `Attributes`.

2.  The **`AppOrchestrator`** passes these commands to the **`XDriver`**.

3.  The **`XDriver`** passes them to its `Graphics` subsystem.

4.  The **`Graphics::draw_text_run`** method receives the `DrawTextRun` command. This is where the core logic now lives:
    * It iterates through the characters of the input string.
    * For each `char`, it calls `self.font_manager.get_glyph(char, ...)`.
    * It builds a `Vec<xft::XftGlyphFontSpec>`, which pairs each glyph ID with the correct, platform-specific font handle provided by the font manager.
    * It calls the `unsafe` C function `XftDrawGlyphFontSpec` with this completed list to render all the characters in a single, efficient call.

This flow ensures the `Renderer` is completely decoupled from the platform-specific complexities of font handling.

## 5. Phased Implementation Plan

The refactor can be broken down into three distinct phases:

* **Phase 1: Enhance Font Loading**
    * Update `FontConfig` in `config.rs` to include fields for `bold`, `italic`, and `bold_italic` font strings.
    * Modify the `Graphics` struct to hold four `SafeXftFont` instances.
    * Update the `Graphics` constructor to load all four primary font styles.

* **Phase 2: Build the `X11FontManager`**
    * Create the `FontManager` trait.
    * Create the concrete `X11FontManager` struct, implementing the trait.
    * Implement the font caching and `fontconfig` lookup logic inside `X11FontManager::get_glyph`, stealing heavily from `st`'s `xmakeglyphfontspecs` function.

* **Phase 3: Refactor the `Graphics` Driver**
    * Integrate the `X11FontManager` into the `Graphics` struct.
    * Completely rewrite the `Graphics::draw_text_run` method to use the new `GlyphFontSpec` rendering path, as detailed in section 4.
