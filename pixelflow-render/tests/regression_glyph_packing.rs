use pixelflow_core::{batch::Batch, TensorView};

// Simulate the FIXED build.rs packing logic (Padded per row)
fn pack_4bit_padded(bitmap: &[u8], width: usize) -> Vec<u8> {
    // Calculate stride in bytes
    let stride = (width + 1) / 2;
    let height = bitmap.len() / width;

    let mut packed = Vec::with_capacity(stride * height);

    for row in 0..height {
        let row_start = row * width;
        let mut current_byte = 0u8;
        let mut high_nibble = true;

        for col in 0..width {
            let pixel = bitmap[row_start + col];
            // Simple quantization for test
            let val_4bit = ((pixel as u16 + 8) / 17).min(15) as u8;

            if high_nibble {
                current_byte = val_4bit << 4;
                high_nibble = false;
            } else {
                current_byte |= val_4bit;
                packed.push(current_byte);
                high_nibble = true;
                current_byte = 0; // Reset
            }
        }

        // If row ends on high nibble, push the partial byte
        if !high_nibble {
            packed.push(current_byte);
        }
    }

    packed
}

#[test]
fn test_odd_width_glyph_rendering_correctness() {
    // 3x2 glyph
    // Row 0: 0, 255, 0 -> 0, F, 0
    // Row 1: 255, 0, 255 -> F, 0, F

    // Padded Packing:
    // Row 0: 0F, 0(pad) -> Byte 0: 0F, Byte 1: 00
    // Row 1: F0, F(pad) -> Byte 2: F0, Byte 3: F0

    // Total 4 bytes.

    let width = 3;
    let height = 2;
    let bitmap = vec![0, 255, 0, 255, 0, 255];

    // 1. Pack with padding (New build.rs behavior)
    let packed = pack_4bit_padded(&bitmap, width);

    assert_eq!(packed.len(), 4);
    assert_eq!(packed[0], 0x0F);
    assert_eq!(packed[1], 0x00);
    assert_eq!(packed[2], 0xF0);
    assert_eq!(packed[3], 0xF0);

    // 2. Setup TensorView (New glyph.rs behavior)
    let stride = (width + 1) / 2;
    let view = TensorView::new(&packed, width, height, stride);

    // 3. Verify correctness
    unsafe {
        // (0, 0) -> 0
        let val = view
            .gather_4bit(Batch::splat(0), Batch::splat(0))
            .to_array_usize()[0];
        assert_eq!(val, 0);

        // (1, 0) -> F
        let val = view
            .gather_4bit(Batch::splat(1), Batch::splat(0))
            .to_array_usize()[0];
        assert_eq!(val, 0xFF);

        // (0, 1) -> F (Previously failed here)
        let val = view
            .gather_4bit(Batch::splat(0), Batch::splat(1))
            .to_array_usize()[0];
        assert_eq!(
            val, 0xFF,
            "Glyph corruption at (0,1). Got {:#x}, expected 0xFF",
            val
        );

        // (1, 1) -> 0
        let val = view
            .gather_4bit(Batch::splat(1), Batch::splat(1))
            .to_array_usize()[0];
        assert_eq!(val, 0);

        // (2, 1) -> F
        let val = view
            .gather_4bit(Batch::splat(2), Batch::splat(1))
            .to_array_usize()[0];
        assert_eq!(val, 0xFF);
    }
}
