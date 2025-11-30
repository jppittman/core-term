
#[cfg(test)]
mod tests {
    use pixelflow_fonts::{Font, glyph};
    use pixelflow_core::batch::Batch;
    use pixelflow_core::pipe::Surface;

    // A minimal valid TTF header/table to avoid crash?
    // ttf-parser needs valid data. We can't mock easily without a real font file.
    // We'll skip integration tests if no font is available, or mock Font if possible.
    // Font struct wraps ttf_parser::Face, which is hard to mock.
    // Let's rely on unit tests for components.
}
