use pixelflow_render::{process_frame, Color, NamedColor, Op, ScreenViewMut};

#[test]
fn test_clear() {
    let mut fb = vec![0u32; 100];
    let ops: Vec<Op<&[u8]>> = vec![Op::Clear {
        color: Color::Named(NamedColor::Green),
    }];
    let mut screen = ScreenViewMut::new(&mut fb, 10, 10, 1, 1);
    process_frame(&mut screen, &ops);

    let green: u32 = Color::Named(NamedColor::Green).into();
    for px in fb {
        assert_eq!(px, green);
    }
}

#[test]
fn test_blit_clipping() {
    let width = 10;
    let height = 10;
    let mut fb = vec![0u32; width * height];

    // 2x2 data
    let data = vec![0xFF; 4 * 4]; // White

    // Blit at (9, 9).
    // Size 2x2.
    // Pixel (0,0) of data -> (9,9) of fb. Valid.
    // Pixel (1,0) of data -> (10,9) of fb. Clipped X.
    // Pixel (0,1) of data -> (9,10) of fb. Clipped Y.
    // Pixel (1,1) of data -> (10,10) of fb. Clipped.

    let ops = vec![Op::Blit {
        data: &data,
        w: 2,
        x: 9,
        y: 9,
    }];

    let mut screen = ScreenViewMut::new(&mut fb, width, height, 1, 1);
    process_frame(&mut screen, &ops);

    // Only (9,9) should be set.
    let idx = 9 * width + 9;
    assert_eq!(fb[idx], 0xFFFFFFFF);

    // Check neighbors are 0
    assert_eq!(fb[idx - 1], 0);
    assert_eq!(fb[idx - width], 0);
}

#[test]
#[should_panic]
fn test_blit_bad_data_len() {
    let mut fb = vec![0u32; 10];
    // Data length 3 (not div by 4)
    let data = vec![0, 0, 0];
    let ops = vec![Op::Blit {
        data: &data,
        w: 1,
        x: 0,
        y: 0,
    }];

    let mut screen = ScreenViewMut::new(&mut fb, 10, 1, 1, 1);
    process_frame(&mut screen, &ops);
}

#[test]
fn test_text_coordinates() {
    // Op::Text coordinates are documented as pixels, but implementation behaves as grid cells?
    // Let's verify.
    // cell_width=10, cell_height=20.
    // Op::Text x=1, y=1.
    // If pixels: glyph at (1, 1).
    // If grid: glyph at (1*10, 1*20) = (10, 20).

    let width = 50;
    let height = 50;
    let mut fb = vec![0u32; width * height];

    // Use 'A' which should produce some pixels.
    // Bg=Black, Fg=White.
    let ops: Vec<Op<&[u8]>> = vec![Op::Text {
        ch: 'A',
        x: 1,
        y: 1,
        fg: Color::Named(NamedColor::White),
        bg: Color::Named(NamedColor::Black),
        bold: false,
        italic: false,
    }];

    let mut screen = ScreenViewMut::new(&mut fb, width, height, 10, 20);
    process_frame(&mut screen, &ops);

    // Check where pixels are drawn.
    // If grid (1,1) -> (10, 20).
    // Background clearing happens on cell.
    // Cell (1,1) spans x=[10, 19], y=[20, 39].
    // If background is drawn there, it proves grid coordinates.
    // Since default FB is 0, and Black is 0, we can't tell unless we init FB to something else.
    // Let's init FB to Red.
}

#[test]
fn test_text_coordinates_verification() {
    let width = 50;
    let height = 50;
    let mut fb = vec![0xFF0000FF; width * height]; // Red background

    let ops: Vec<Op<&[u8]>> = vec![Op::Text {
        ch: ' ', // Space, just clears background
        x: 1,
        y: 1,
        fg: Color::Named(NamedColor::White),
        bg: Color::Named(NamedColor::Black), // 0x000000FF (Black opaque) -> Wait, NamedColor::Black is (0,0,0). Alpha?
        bold: false,
        italic: false,
    }];
    // NamedColor::Black -> (0,0,0). Alpha 255. 0xFF000000.

    let mut screen = ScreenViewMut::new(&mut fb, width, height, 10, 20);
    process_frame(&mut screen, &ops);

    // Check (10, 20). If it's Black, then grid coordinates used.
    // If (1, 1) is Black, then pixel coordinates used.

    let idx_grid = 20 * width + 10;
    let idx_pixel = 1 * width + 1;

    let black: u32 = Color::Named(NamedColor::Black).into();

    // Assuming grid coordinates based on code review
    if fb[idx_grid] == black {
        // Confirmed grid coordinates
        assert_eq!(fb[idx_grid], black);
    } else if fb[idx_pixel] == black {
        // Confirmed pixel coordinates
        panic!("Op::Text used pixel coordinates, contrary to expectation from code review!");
    } else {
        // Maybe neither?
        // Check if anything changed?
        // Maybe ' ' doesn't clear background? Implementation says:
        // "1. Clear the cell background ... procedurally first"
        // So it should clear.

        // Let's debug print a few pixels if it fails
        // panic!("Neither grid (10,20) nor pixel (1,1) was cleared to black.");

        // Actually, let's just assert grid coordinates because that's what the code says.
        assert_eq!(
            fb[idx_grid], black,
            "Expected grid coordinates (10,20) to be cleared"
        );
    }
}
