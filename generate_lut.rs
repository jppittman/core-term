fn main() {
    let mut named = vec![
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

    let mut cube = Vec::new();
    for r in 0..6 {
        for g in 0..6 {
            for b in 0..6 {
                let r_val = if r == 0 { 0 } else { r * 40 + 55 };
                let g_val = if g == 0 { 0 } else { g * 40 + 55 };
                let b_val = if b == 0 { 0 } else { b * 40 + 55 };
                cube.push((r_val, g_val, b_val));
            }
        }
    }

    let mut gray = Vec::new();
    for i in 0..24 {
        let v = i * 10 + 8;
        gray.push((v, v, v));
    }

    let mut all = Vec::new();
    all.extend(named);
    all.extend(cube);
    all.extend(gray);

    println!("const PALETTE_256: [(u8, u8, u8); 256] = [");
    for (i, (r, g, b)) in all.iter().enumerate() {
        if i % 4 == 0 {
            print!("    ");
        }
        print!("({}, {}, {}), ", r, g, b);
        if (i + 1) % 4 == 0 {
            println!("");
        }
    }
    println!("];");
}
