fn main() {
    println!("static PALETTE_256: [(u8, u8, u8); 256] = [");

    // 0-15: Named Colors
    let named_colors = [
        (0, 0, 0),       // Black
        (205, 0, 0),     // Red
        (0, 205, 0),     // Green
        (205, 205, 0),   // Yellow
        (0, 0, 238),     // Blue
        (205, 0, 205),   // Magenta
        (0, 205, 205),   // Cyan
        (229, 229, 229), // White
        (127, 127, 127), // BrightBlack
        (255, 0, 0),     // BrightRed
        (0, 255, 0),     // BrightGreen
        (255, 255, 0),   // BrightYellow
        (92, 92, 255),   // BrightBlue
        (255, 0, 255),   // BrightMagenta
        (0, 255, 255),   // BrightCyan
        (255, 255, 255), // BrightWhite
    ];

    for (r, g, b) in named_colors.iter() {
        println!("    ({}, {}, {}),", r, g, b);
    }

    // 16-231: 6x6x6 Color Cube
    for idx in 16..232 {
        let cube_idx = idx - 16;
        let r_comp = (cube_idx / (6 * 6)) % 6;
        let g_comp = (cube_idx / 6) % 6;
        let b_comp = cube_idx % 6;

        let r_val = if r_comp == 0 { 0 } else { r_comp * 40 + 55 };
        let g_val = if g_comp == 0 { 0 } else { g_comp * 40 + 55 };
        let b_val = if b_comp == 0 { 0 } else { b_comp * 40 + 55 };
        println!("    ({}, {}, {}),", r_val, g_val, b_val);
    }

    // 232-255: Grayscale Ramp
    for idx in 232..256 {
        let gray_idx = idx - 232;
        let level = gray_idx * 10 + 8;
        println!("    ({}, {}, {}),", level, level, level);
    }

    println!("];");
}
