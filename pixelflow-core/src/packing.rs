use alloc::vec::Vec;

/// Computes the stride (in bytes) for a 4-bit packed bitmap with the given width.
///
/// Rows are padded to the nearest byte.
///
/// # Parameters
/// * `width` - The width of the image in pixels.
///
/// # Returns
/// * The number of bytes per row.
#[inline]
pub fn compute_stride_4bit(width: usize) -> usize {
    (width + 1) / 2
}

/// Packs an 8-bit grayscale bitmap into 4-bit format (2 pixels per byte).
///
/// Each byte contains two pixels. The high nibble corresponds to the even column (x),
/// and the low nibble corresponds to the odd column (x+1).
///
/// # Parameters
/// * `bitmap` - The 8-bit grayscale bitmap data (row-major).
/// * `width` - The width of the image in pixels.
///
/// # Returns
/// * A `Vec<u8>` containing the packed data.
pub fn pack_4bit(bitmap: &[u8], width: usize) -> Vec<u8> {
    if width == 0 {
        return Vec::new();
    }

    // Calculate stride in bytes
    let stride = compute_stride_4bit(width);
    let height = bitmap.len() / width;

    let mut packed = Vec::with_capacity(stride * height);

    for row in 0..height {
        let row_start = row * width;
        let mut current_byte = 0u8;
        let mut high_nibble = true;

        for col in 0..width {
            let pixel = bitmap[row_start + col];
            // Quantize 0..255 -> 0..15
            // (pixel + 8) / 17 is equivalent to round(pixel / 17)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TensorView, batch::Batch};
    use alloc::vec;

    #[test]
    fn test_compute_stride() {
        assert_eq!(compute_stride_4bit(0), 0);
        assert_eq!(compute_stride_4bit(1), 1);
        assert_eq!(compute_stride_4bit(2), 1);
        assert_eq!(compute_stride_4bit(3), 2);
        assert_eq!(compute_stride_4bit(4), 2);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let width = 3;
        let height = 2;
        // 0x11 -> 1, 0xFF -> 15, 0x80 -> ~7/8 (136)
        // 136 / 17 = 8.
        // Row 0: 0x11 (17), 0xFF (255), 0x88 (136)
        // Row 1: 0x00 (0),  0x22 (34),  0xEE (238) -> 238/17=14
        let data = [
            0x11, 0xFF, 0x88,
            0x00, 0x22, 0xEE
        ];
        let packed = pack_4bit(&data, width);

        // Stride = 2
        // Row 0:
        // Col 0: 17 -> 1. High nibble. 0x10.
        // Col 1: 255 -> 15. Low nibble. 0x1F.
        // Col 2: 136 -> 8. High nibble. 0x80.
        // Row 0 bytes: 0x1F, 0x80.

        // Row 1:
        // Col 0: 0 -> 0. High nibble. 0x00.
        // Col 1: 34 -> 2. Low nibble. 0x02.
        // Col 2: 238 -> 14. High nibble. 0xE0.
        // Row 1 bytes: 0x02, 0xE0.

        assert_eq!(packed, vec![0x1F, 0x80, 0x02, 0xE0]);

        let stride = compute_stride_4bit(width);
        let view = TensorView::new(&packed, width, height, stride);

        // Check Row 0
        unsafe {
            let x = Batch::new(0, 1, 2, 0); // 4th lane is dummy (0)
            let y = Batch::splat(0);
            let gathered = view.gather_4bit(x, y);
            let vals = gathered.to_array_usize();

            // Expected: 1->0x11, 15->0xFF, 8->0x88. Dummy 0->0x11 (from x=0)
            assert_eq!(vals[0], 0x11);
            assert_eq!(vals[1], 0xFF);
            assert_eq!(vals[2], 0x88);
            assert_eq!(vals[3], 0x11);
        }

        // Check Row 1
        unsafe {
            let x = Batch::new(0, 1, 2, 0);
            let y = Batch::splat(1);
            let gathered = view.gather_4bit(x, y);
            let vals = gathered.to_array_usize();

            // Expected: 0->0x00, 2->0x22, 14->0xEE
            assert_eq!(vals[0], 0x00);
            assert_eq!(vals[1], 0x22);
            assert_eq!(vals[2], 0xEE);
        }
    }
}
