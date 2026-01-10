//! Tests for wide character edge rendering.
//!
//! This test file specifically targets the bug where edges of wide characters
//! like "m" don't render correctly. The issue may be related to how glyph
//! bounding boxes are normalized using max_dim for both axes.

use pixelflow_graphics::fonts::{text, Font};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::{execute, TensorShape};

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

/// Helper to visualize a character render as ASCII art
fn visualize_render(pixels: &[Rgba8], width: usize, height: usize) -> String {
    let mut result = String::new();
    for y in 0..height {
        for x in 0..width {
            let val = pixels[y * width + x].r();
            let ch = if val > 200 {
                '#'
            } else if val > 100 {
                '+'
            } else if val > 50 {
                '.'
            } else {
                ' '
            };
            result.push(ch);
        }
        result.push('\n');
    }
    result
}

/// Helper to find the topmost and bottommost rows with any coverage
fn find_vertical_extent(pixels: &[Rgba8], width: usize, height: usize) -> (usize, usize) {
    let mut topmost = height;
    let mut bottommost = 0;

    for y in 0..height {
        for x in 0..width {
            if pixels[y * width + x].r() > 0 {
                topmost = topmost.min(y);
                bottommost = bottommost.max(y);
            }
        }
    }

    (topmost, bottommost)
}

/// Helper to render a single character and return the pixel buffer
fn render_char(font: &Font, ch: char, size: f32, width: usize, height: usize) -> Vec<Rgba8> {
    let glyph = text(font, &ch.to_string(), size);
    let lifted = Grayscale(glyph);
    let mut pixels = vec![Rgba8::default(); width * height];
    execute(&lifted, &mut pixels, TensorShape::new(width, height));
    pixels
}

/// Helper to find the leftmost and rightmost columns with any coverage
fn find_horizontal_extent(pixels: &[Rgba8], width: usize, height: usize) -> (usize, usize) {
    let mut leftmost = width;
    let mut rightmost = 0;

    for y in 0..height {
        for x in 0..width {
            if pixels[y * width + x].r() > 0 {
                leftmost = leftmost.min(x);
                rightmost = rightmost.max(x);
            }
        }
    }

    (leftmost, rightmost)
}

/// Helper to count pixels with coverage above a threshold in a column
fn column_coverage(pixels: &[Rgba8], width: usize, height: usize, col: usize, threshold: u8) -> usize {
    (0..height)
        .filter(|&y| pixels[y * width + col].r() > threshold)
        .count()
}

/// Test that the letter "m" renders with proper left and right edges.
///
/// The "m" is a wide character - if there's a bug with max_dim normalization,
/// the edges may get clipped or have incorrect coverage.
#[test]
fn wide_char_m_has_both_edges() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 48.0;
    let width = 60;
    let height = 60;

    let pixels = render_char(&font, 'm', size, width, height);
    let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);

    // The character should span a reasonable width
    let rendered_width = rightmost - leftmost + 1;
    assert!(
        rendered_width > 10,
        "Character 'm' should have substantial width, got {} pixels (left={}, right={})",
        rendered_width,
        leftmost,
        rightmost
    );

    // Check that the leftmost column has meaningful coverage (not just edge artifacts)
    let left_edge_coverage = column_coverage(&pixels, width, height, leftmost, 50);
    assert!(
        left_edge_coverage >= 2,
        "Left edge of 'm' should have at least 2 pixels with coverage > 50, got {}",
        left_edge_coverage
    );

    // Check that the rightmost column has meaningful coverage
    let right_edge_coverage = column_coverage(&pixels, width, height, rightmost, 50);
    assert!(
        right_edge_coverage >= 2,
        "Right edge of 'm' should have at least 2 pixels with coverage > 50, got {}",
        right_edge_coverage
    );
}

/// Test that "m" renders symmetrically (left and right legs should be similar).
#[test]
fn wide_char_m_symmetric_legs() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 48.0;
    let width = 60;
    let height = 60;

    let pixels = render_char(&font, 'm', size, width, height);
    let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);

    // Count total coverage in left quarter and right quarter of the rendered glyph
    let glyph_width = rightmost - leftmost + 1;
    let quarter = glyph_width / 4;

    let mut left_quarter_coverage = 0usize;
    let mut right_quarter_coverage = 0usize;

    for y in 0..height {
        for x in 0..quarter {
            left_quarter_coverage += pixels[y * width + leftmost + x].r() as usize;
        }
        for x in (glyph_width - quarter)..glyph_width {
            right_quarter_coverage += pixels[y * width + leftmost + x].r() as usize;
        }
    }

    // The coverage should be roughly symmetric (within 30% difference)
    let ratio = if left_quarter_coverage > right_quarter_coverage {
        right_quarter_coverage as f64 / left_quarter_coverage as f64
    } else {
        left_quarter_coverage as f64 / right_quarter_coverage as f64
    };

    assert!(
        ratio > 0.7,
        "Character 'm' should be roughly symmetric: left_coverage={}, right_coverage={}, ratio={}",
        left_quarter_coverage,
        right_quarter_coverage,
        ratio
    );
}

/// Test that wide characters like "M" and "W" render correctly.
#[test]
fn wide_chars_m_and_w_render_correctly() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 48.0;
    let width = 60;
    let height = 60;

    for ch in ['M', 'W', 'm', 'w'] {
        let pixels = render_char(&font, ch, size, width, height);
        let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);

        let rendered_width = if rightmost >= leftmost {
            rightmost - leftmost + 1
        } else {
            0
        };

        assert!(
            rendered_width > 10,
            "Character '{}' should render with width > 10, got {}",
            ch,
            rendered_width
        );

        // Ensure both edges have coverage
        if rendered_width > 0 {
            let left_coverage = column_coverage(&pixels, width, height, leftmost, 30);
            let right_coverage = column_coverage(&pixels, width, height, rightmost, 30);

            assert!(
                left_coverage > 0 && right_coverage > 0,
                "Character '{}' edges should have coverage: left={}, right={}",
                ch,
                left_coverage,
                right_coverage
            );
        }
    }
}

/// Test that the rendered width of glyphs matches the advance width.
///
/// This helps detect if glyph geometry is being incorrectly bounded.
#[test]
fn glyph_rendered_width_matches_advance() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 48.0;
    let width = 80;
    let height = 60;

    for ch in ['m', 'i', 'M', 'l'] {
        let pixels = render_char(&font, ch, size, width, height);
        let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);

        let rendered_width = if rightmost >= leftmost {
            (rightmost - leftmost + 1) as f32
        } else {
            0.0
        };

        let advance = font.advance_scaled(ch, size).unwrap_or(0.0);

        // The rendered width should be less than or equal to advance width
        // (there should be some sidebearing)
        assert!(
            rendered_width <= advance + 2.0,
            "Character '{}' rendered width ({}) should not exceed advance ({})",
            ch,
            rendered_width,
            advance
        );

        // But it shouldn't be drastically smaller either
        assert!(
            rendered_width > advance * 0.3,
            "Character '{}' rendered width ({}) is too small relative to advance ({})",
            ch,
            rendered_width,
            advance
        );
    }
}

/// Test that all three vertical strokes of lowercase 'm' are visible.
///
/// This is a direct test for the edge rendering bug - if the rightmost stroke
/// is missing, this test should fail.
#[test]
fn lowercase_m_has_three_strokes() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 64.0;
    let width = 80;
    let height = 80;

    let pixels = render_char(&font, 'm', size, width, height);
    let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);
    let (topmost, bottommost) = find_vertical_extent(&pixels, width, height);

    eprintln!("Glyph 'm' extent: x=[{}, {}], y=[{}, {}]", leftmost, rightmost, topmost, bottommost);
    eprintln!("Rendered 'm':\n{}", visualize_render(&pixels, width, height));

    if rightmost <= leftmost {
        panic!("Character 'm' did not render any pixels");
    }

    // Find the vertical middle of the GLYPH (not the buffer)
    let mid_y = (topmost + bottommost) / 2;
    let scan_range = 5; // Scan a few rows around the middle

    eprintln!("Scanning for peaks at y={} (range {})", mid_y, scan_range);

    // Collect column coverage at the vertical middle
    let mut column_coverages: Vec<(usize, usize)> = Vec::new();
    for x in leftmost..=rightmost {
        let coverage: usize = (mid_y.saturating_sub(scan_range)..=(mid_y + scan_range).min(height - 1))
            .map(|y| pixels[y * width + x].r() as usize)
            .sum();
        column_coverages.push((x, coverage));
    }

    eprintln!("Column coverages: {:?}", column_coverages);

    // Find runs of high coverage (strokes) separated by runs of low coverage (gaps)
    // A stroke is a contiguous run of columns with coverage > threshold
    let threshold = 100;
    let mut strokes: Vec<(usize, usize)> = Vec::new(); // (start_x, end_x)
    let mut in_stroke = false;
    let mut stroke_start = 0;

    for (x, cov) in &column_coverages {
        if *cov > threshold {
            if !in_stroke {
                in_stroke = true;
                stroke_start = *x;
            }
        } else if in_stroke {
            strokes.push((stroke_start, *x - 1));
            in_stroke = false;
        }
    }
    // Handle case where last column is part of a stroke
    if in_stroke {
        if let Some((last_x, _)) = column_coverages.last() {
            strokes.push((stroke_start, *last_x));
        }
    }

    eprintln!("Detected strokes: {:?}", strokes);

    assert!(
        strokes.len() >= 3,
        "Character 'm' should have 3 vertical strokes, found {}: {:?}",
        strokes.len(),
        strokes
    );
}

/// Test that right edge of 'm' has proper antialiasing.
///
/// The bug may manifest as missing antialiasing on the right edge
/// due to incorrect bounding box handling.
#[test]
fn m_right_edge_has_antialiasing() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 48.0;
    let width = 60;
    let height = 60;

    let pixels = render_char(&font, 'm', size, width, height);
    let (_, rightmost) = find_horizontal_extent(&pixels, width, height);
    let (topmost, bottommost) = find_vertical_extent(&pixels, width, height);

    // Check that the right edge has similar coverage pattern to left edge
    let (leftmost, _) = find_horizontal_extent(&pixels, width, height);

    let mut left_edge_sum = 0usize;
    let mut right_edge_sum = 0usize;

    for y in topmost..=bottommost {
        left_edge_sum += pixels[y * width + leftmost].r() as usize;
        right_edge_sum += pixels[y * width + rightmost].r() as usize;
    }

    // The sums should be in the same ballpark (within 50%)
    let ratio = if left_edge_sum > right_edge_sum {
        right_edge_sum as f64 / left_edge_sum as f64
    } else {
        left_edge_sum as f64 / right_edge_sum as f64
    };

    assert!(
        ratio > 0.5,
        "Left and right edge coverage should be similar: left={}, right={}, ratio={}",
        left_edge_sum,
        right_edge_sum,
        ratio
    );
}

/// Test rendering at various sizes to find size-dependent edge bugs.
#[test]
fn m_renders_at_all_sizes() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    for size in [8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0] {
        let width = (size * 2.0) as usize;
        let height = (size * 2.0) as usize;

        let pixels = render_char(&font, 'm', size, width, height);
        let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);

        let rendered_width = if rightmost >= leftmost {
            rightmost - leftmost + 1
        } else {
            0
        };

        assert!(
            rendered_width > 0,
            "Character 'm' at size {} should have non-zero width",
            size
        );

        // Check that both edges have coverage
        if rendered_width > 0 {
            let (topmost, bottommost) = find_vertical_extent(&pixels, width, height);
            let height_pixels = bottommost - topmost + 1;

            // Left edge should have coverage
            let left_coverage: usize = (topmost..=bottommost)
                .map(|y| pixels[y * width + leftmost].r() as usize)
                .sum();

            // Right edge should have coverage
            let right_coverage: usize = (topmost..=bottommost)
                .map(|y| pixels[y * width + rightmost].r() as usize)
                .sum();

            assert!(
                left_coverage > height_pixels * 50,
                "Character 'm' at size {} should have left edge coverage, got {}",
                size,
                left_coverage
            );

            assert!(
                right_coverage > height_pixels * 50,
                "Character 'm' at size {} should have right edge coverage, got {} (LEFT: {})",
                size,
                right_coverage,
                left_coverage
            );
        }
    }
}

/// Test that characters with descenders (g, j, p, q, y) render correctly.
#[test]
fn descender_chars_render_correctly() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 48.0;
    let width = 60;
    let height = 80;

    for ch in ['g', 'j', 'p', 'q', 'y'] {
        let pixels = render_char(&font, ch, size, width, height);
        let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);
        let (topmost, bottommost) = find_vertical_extent(&pixels, width, height);

        let rendered_width = if rightmost >= leftmost {
            rightmost - leftmost + 1
        } else {
            0
        };

        let rendered_height = if bottommost >= topmost {
            bottommost - topmost + 1
        } else {
            0
        };

        assert!(
            rendered_width > 5,
            "Descender char '{}' should have reasonable width, got {}",
            ch,
            rendered_width
        );

        assert!(
            rendered_height > 10,
            "Descender char '{}' should have reasonable height (including descender), got {}",
            ch,
            rendered_height
        );

        // Check bottom edge (descender) has coverage
        let bottom_coverage: usize = (leftmost..=rightmost)
            .map(|x| pixels[bottommost * width + x].r() as usize)
            .sum();

        assert!(
            bottom_coverage > 0,
            "Descender char '{}' should have bottom edge coverage",
            ch
        );
    }
}

/// Test that cached glyphs preserve right edges.
///
/// The CachedGlyph is rasterized to a square bucket×bucket frame.
/// This test verifies that wide characters aren't clipped on the right.
#[test]
fn cached_glyph_preserves_right_edge() {
    use pixelflow_graphics::fonts::GlyphCache;

    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let mut cache = GlyphCache::new();

    // Get advance width for 'm' at size 48
    let size = 48.0;
    let advance = font.advance_scaled('m', size).unwrap();

    eprintln!("Advance width for 'm' at size {}: {}", size, advance);

    // Get the cached glyph
    let cached = cache.get(&font, 'm', size).expect("Failed to cache 'm'");

    // The cached glyph's width/height should be the bucket size
    let bucket_size = cached.width();
    eprintln!("Bucket size: {}", bucket_size);

    // Check that advance width is reasonable relative to bucket size
    // For a monospace font, advance should be <= bucket size
    // If advance > bucket, we might be clipping!
    if advance > bucket_size as f32 {
        panic!(
            "POTENTIAL BUG: Advance width ({}) > bucket size ({}) - right edge may be clipped!",
            advance, bucket_size
        );
    }

    // Now render through the cache and check the right edge
    use pixelflow_graphics::render::color::Grayscale;

    let lifted = Grayscale(cached);
    let width = 60;
    let height = 60;
    let mut pixels = vec![Rgba8::default(); width * height];

    use pixelflow_graphics::render::{execute, TensorShape};
    execute(&lifted, &mut pixels, TensorShape::new(width, height));

    let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);

    eprintln!("Cached glyph 'm' extent: x=[{}, {}]", leftmost, rightmost);

    // Check that the cached glyph has proper left and right edges
    let (topmost, bottommost) = find_vertical_extent(&pixels, width, height);

    let left_coverage: usize = (topmost..=bottommost)
        .map(|y| pixels[y * width + leftmost].r() as usize)
        .sum();

    let right_coverage: usize = (topmost..=bottommost)
        .map(|y| pixels[y * width + rightmost].r() as usize)
        .sum();

    eprintln!("Left edge coverage: {}, Right edge coverage: {}", left_coverage, right_coverage);

    assert!(
        right_coverage > 0,
        "Cached glyph 'm' right edge should have coverage"
    );

    // Edges should have similar coverage (within 80%)
    let ratio = right_coverage as f64 / left_coverage as f64;
    assert!(
        ratio > 0.2,
        "Cached glyph right edge coverage ({}) is much lower than left ({}) - ratio: {}",
        right_coverage,
        left_coverage,
        ratio
    );
}

/// Test that simulates terminal cell rendering with typical 10x16 cell dimensions.
///
/// The terminal bounds glyphs to cell_width × cell_height, which might clip
/// the right edge if cell_width is too narrow for the glyph.
#[test]
fn terminal_cell_rendering_simulation() {
    use pixelflow_graphics::fonts::GlyphCache;

    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let mut cache = GlyphCache::new();

    // Test with multiple cell sizes
    for (cell_width, cell_height) in [(10.0f32, 16.0f32), (20.0, 32.0), (30.0, 48.0)] {
        eprintln!("\n=== Testing cell size {}x{} ===", cell_width, cell_height);
        test_cell_size(&font, &mut cache, cell_width, cell_height);
    }
}

fn test_cell_size(font: &Font, cache: &mut pixelflow_graphics::fonts::GlyphCache, cell_width: f32, cell_height: f32) {
    use pixelflow_core::{Ge, Le, Select, X, Y};

    // Get cached glyph at cell_height size (like terminal does)
    let cached = cache.get(font, 'm', cell_height).expect("Failed to cache 'm'");
    let bucket_size = cached.width();

    eprintln!("Cell dimensions: {}x{}", cell_width, cell_height);
    eprintln!("Cached glyph bucket size: {}", bucket_size);

    // Check the glyph's actual extent when rendered at this size
    let advance = font.advance_scaled('m', cell_height).unwrap();
    eprintln!("Advance width for 'm' at size {}: {}", cell_height, advance);

    // CRITICAL CHECK: Does the glyph fit within cell_width?
    // If advance > cell_width, the terminal will clip the right edge!
    if advance > cell_width {
        eprintln!("WARNING: Glyph advance ({}) > cell_width ({}) - RIGHT EDGE WILL BE CLIPPED!",
            advance, cell_width);
    }

    // Simulate the terminal's bounding condition (from terminal_app.rs:98)
    // let cond = X.ge(0.0) & X.le(cell_width) & Y.ge(0.0) & Y.le(cell_height);

    // Render the glyph bounded like the terminal does
    let bounded = Select {
        cond: Ge(X, 0.0f32) & Le(X, cell_width) & Ge(Y, 0.0f32) & Le(Y, cell_height),
        if_true: cached.clone(),
        if_false: 0.0f32,
    };
    let lifted = Grayscale(bounded);

    // Render to a buffer larger than cell size to see what's being clipped
    let render_width = (cell_width * 2.0) as usize;
    let render_height = (cell_height * 2.0) as usize;
    let mut pixels = vec![Rgba8::default(); render_width * render_height];
    execute(&lifted, &mut pixels, TensorShape::new(render_width, render_height));

    // Visualize
    eprintln!("Terminal-style bounded 'm' (cell {}x{}):", cell_width, cell_height);
    eprintln!("{}", visualize_render(&pixels, render_width, render_height));

    let (leftmost, rightmost) = find_horizontal_extent(&pixels, render_width, render_height);
    eprintln!("Bounded glyph extent: x=[{}, {}]", leftmost, rightmost);

    // Now render WITHOUT the bounding to see the full glyph
    let unbounded = Grayscale(cached);
    let mut unbounded_pixels = vec![Rgba8::default(); render_width * render_height];
    execute(&unbounded, &mut unbounded_pixels, TensorShape::new(render_width, render_height));

    eprintln!("Unbounded 'm':");
    eprintln!("{}", visualize_render(&unbounded_pixels, render_width, render_height));

    let (unbounded_left, unbounded_right) = find_horizontal_extent(&unbounded_pixels, render_width, render_height);
    eprintln!("Unbounded glyph extent: x=[{}, {}]", unbounded_left, unbounded_right);

    // Now render the DIRECT glyph (not cached) for comparison
    let direct_glyph = text(font, "m", cell_height);
    let direct_lifted = Grayscale(direct_glyph);
    let mut direct_pixels = vec![Rgba8::default(); render_width * render_height];
    execute(&direct_lifted, &mut direct_pixels, TensorShape::new(render_width, render_height));

    eprintln!("Direct (uncached) 'm':");
    eprintln!("{}", visualize_render(&direct_pixels, render_width, render_height));

    let (direct_left, direct_right) = find_horizontal_extent(&direct_pixels, render_width, render_height);
    eprintln!("Direct glyph extent: x=[{}, {}]", direct_left, direct_right);

    // CRITICAL: Compare cached vs direct rendering
    if unbounded_right < direct_right {
        eprintln!("BUG DETECTED: Cached glyph is missing {} pixels on right edge compared to direct rendering!",
            direct_right - unbounded_right);
        panic!(
            "Cached glyph is truncated compared to direct rendering!\n\
             Cached extent: [{}, {}]\n\
             Direct extent: [{}, {}]\n\
             Missing {} pixels from right edge",
            unbounded_left, unbounded_right,
            direct_left, direct_right,
            direct_right - unbounded_right
        );
    }

    // Check if bounding is clipping the glyph
    if rightmost < unbounded_right {
        eprintln!("BUG DETECTED: Terminal bounding clips {} pixels from right edge!",
            unbounded_right - rightmost);

        // Fail the test to expose the bug
        panic!(
            "Terminal cell bounding clips the right edge of 'm'!\n\
             Unbounded extent: [{}, {}]\n\
             Bounded extent: [{}, {}]\n\
             Cell width: {}\n\
             Glyph advance: {}\n\
             {} pixels clipped from right edge",
            unbounded_left, unbounded_right,
            leftmost, rightmost,
            cell_width, advance,
            unbounded_right - rightmost
        );
    }
}

/// Test that all three strokes of 'm' have equal width.
///
/// This is the actual bug - the rightmost stroke renders thinner than the others.
#[test]
fn m_strokes_have_equal_width() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 64.0;
    let width = 80;
    let height = 80;

    let pixels = render_char(&font, 'm', size, width, height);
    let (leftmost, rightmost) = find_horizontal_extent(&pixels, width, height);
    let (topmost, bottommost) = find_vertical_extent(&pixels, width, height);

    // Find the vertical middle of the glyph where strokes are solid
    let mid_y = (topmost + bottommost) / 2;
    let scan_range = 3;

    // Find strokes (runs of high coverage)
    let threshold = 200; // High threshold for solid coverage
    let mut strokes: Vec<(usize, usize)> = Vec::new();
    let mut in_stroke = false;
    let mut stroke_start = 0;

    for x in leftmost..=rightmost {
        let coverage: usize = (mid_y.saturating_sub(scan_range)..=(mid_y + scan_range).min(height - 1))
            .map(|y| pixels[y * width + x].r() as usize)
            .sum();

        let is_solid = coverage > threshold * (scan_range * 2 + 1);
        if is_solid {
            if !in_stroke {
                in_stroke = true;
                stroke_start = x;
            }
        } else if in_stroke {
            strokes.push((stroke_start, x - 1));
            in_stroke = false;
        }
    }
    if in_stroke {
        strokes.push((stroke_start, rightmost));
    }

    eprintln!("Detected strokes: {:?}", strokes);

    // Should have 3 strokes
    assert!(strokes.len() >= 3, "Expected 3 strokes, found {}", strokes.len());

    // Calculate stroke widths
    let stroke_widths: Vec<usize> = strokes.iter().map(|(s, e)| e - s + 1).collect();
    eprintln!("Stroke widths: {:?}", stroke_widths);

    // All strokes should have similar width (within 2 pixels)
    let max_width = *stroke_widths.iter().max().unwrap();
    let min_width = *stroke_widths.iter().min().unwrap();
    let width_diff = max_width - min_width;

    eprintln!("Width difference between strokes: {} (max={}, min={})", width_diff, max_width, min_width);

    // The bug: rightmost stroke is thinner
    if width_diff > 2 {
        let first_stroke_width = stroke_widths[0];
        let last_stroke_width = stroke_widths[stroke_widths.len() - 1];

        if last_stroke_width < first_stroke_width - 1 {
            panic!(
                "BUG: Rightmost stroke is thinner than leftmost!\n\
                 First stroke width: {}\n\
                 Last stroke width: {}\n\
                 Stroke widths: {:?}",
                first_stroke_width, last_stroke_width, stroke_widths
            );
        }
    }

    assert!(
        width_diff <= 2,
        "Stroke widths should be similar, but difference is {} pixels: {:?}",
        width_diff, stroke_widths
    );
}

/// Compare narrow character 'i' vs wide character 'm' edge quality.
///
/// If there's a width-dependent bug, the narrow character should render
/// correctly while the wide one has issues.
#[test]
fn compare_narrow_vs_wide_char_edges() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let size = 48.0;
    let width = 80;
    let height = 60;

    // Render narrow 'l' (lowercase L)
    let narrow_pixels = render_char(&font, 'l', size, width, height);
    let (narrow_left, narrow_right) = find_horizontal_extent(&narrow_pixels, width, height);

    // Render wide 'm'
    let wide_pixels = render_char(&font, 'm', size, width, height);
    let (wide_left, wide_right) = find_horizontal_extent(&wide_pixels, width, height);

    // Both should have reasonable extents
    let narrow_width = narrow_right.saturating_sub(narrow_left);
    let wide_width = wide_right.saturating_sub(wide_left);

    assert!(
        narrow_width > 0,
        "Narrow character 'l' should have some width"
    );
    assert!(
        wide_width > narrow_width,
        "Wide character 'm' ({}) should be wider than 'l' ({})",
        wide_width,
        narrow_width
    );

    // Check edge coverage for both
    let narrow_right_coverage = column_coverage(&narrow_pixels, width, height, narrow_right, 30);
    let wide_right_coverage = column_coverage(&wide_pixels, width, height, wide_right, 30);

    assert!(
        narrow_right_coverage > 0,
        "Narrow char 'l' right edge should have coverage"
    );
    assert!(
        wide_right_coverage > 0,
        "Wide char 'm' right edge should have coverage (this is the bug we're looking for)"
    );
}
