use pixelflow_render::{render, render_to_buffer, render_u32, Frame, Rgba, Surface, Batch};

struct PatternSurface;

impl Surface<u32> for PatternSurface {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u32> {
        // Pattern: x + y * 1000
        x + y * Batch::splat(1000)
    }
}

// For render generic test
#[derive(Clone, Copy)]
struct ColorSurface(Rgba);

impl Surface<Rgba> for ColorSurface {
    fn eval(&self, _: Batch<u32>, _: Batch<u32>) -> Batch<Rgba> {
        let b = Batch::splat(self.0.0);
        b.transmute()
    }
}

#[test]
fn test_render_u32_simd() {
    let surface = PatternSurface;
    // Width 6: 4 (SIMD) + 2 (Scalar) to test both paths
    let width = 6;
    let height = 2;
    let mut buffer = vec![0u32; width * height];

    render_u32(&surface, &mut buffer, width, height);

    // Row 0: 0, 1, 2, 3, 4, 5
    // Row 1: 1000, 1001, 1002, 1003, 1004, 1005
    let expected = vec![
        0, 1, 2, 3, 4, 5,
        1000, 1001, 1002, 1003, 1004, 1005
    ];

    assert_eq!(buffer, expected);
}

#[test]
fn test_render_u32_small_width() {
    let surface = PatternSurface;
    // Width 3: 0 (SIMD) + 3 (Scalar)
    let width = 3;
    let height = 2;
    let mut buffer = vec![0u32; width * height];

    render_u32(&surface, &mut buffer, width, height);

    let expected = vec![
        0, 1, 2,
        1000, 1001, 1002
    ];

    assert_eq!(buffer, expected);
}

#[test]
fn test_render_u32_exact_simd_width() {
    let surface = PatternSurface;
    // Width 4: 4 (SIMD) + 0 (Scalar)
    let width = 4;
    let height = 1;
    let mut buffer = vec![0u32; width * height];

    render_u32(&surface, &mut buffer, width, height);

    let expected = vec![
        0, 1, 2, 3
    ];

    assert_eq!(buffer, expected);
}

#[test]
fn test_render_frame() {
    let color = Rgba::new(1, 2, 3, 4);
    let surface = ColorSurface(color);
    let mut frame = Frame::<Rgba>::new(4, 4);

    render(surface, &mut frame);

    for pixel in frame.as_slice() {
        assert_eq!(*pixel, color);
    }
}

#[test]
fn test_render_to_buffer() {
    let color = Rgba::new(5, 6, 7, 8);
    let surface = ColorSurface(color);
    let width = 2;
    let height = 2;
    let mut buffer = vec![Rgba::default(); width * height];

    render_to_buffer(surface, &mut buffer, width, height);

    for pixel in buffer {
        assert_eq!(pixel, color);
    }
}

#[test]
fn test_render_u32_unaligned_slice() {
    let surface = PatternSurface;
    let width = 4;
    let height = 1;
    let mut backing = vec![0u32; width * height + 1];
    let buffer = &mut backing[1..];

    render_u32(&surface, buffer, width, height);

    let expected = vec![0, 1, 2, 3];
    assert_eq!(buffer, expected);
}

#[test]
#[should_panic]
fn test_render_u32_buffer_too_small() {
    let surface = PatternSurface;
    let width = 100;
    let height = 1;
    let mut buffer = vec![0u32; 10];
    render_u32(&surface, &mut buffer, width, height);
}

#[test]
#[should_panic]
fn test_render_to_buffer_too_small() {
    let color = Rgba::new(0, 0, 0, 0);
    let surface = ColorSurface(color);
    let width = 10;
    let height = 10;
    let mut buffer = vec![Rgba::default(); 50]; // Half the needed size

    render_to_buffer(surface, &mut buffer, width, height);
}
