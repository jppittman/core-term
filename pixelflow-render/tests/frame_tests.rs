use pixelflow_render::{Frame, Rgba, Bgra, Surface, Batch};

#[test]
fn test_frame_new() {
    let frame = Frame::<Rgba>::new(100, 200);
    assert_eq!(frame.width, 100);
    assert_eq!(frame.height, 200);
    assert_eq!(frame.data.len(), 20000);
    // Check default initialization (all zeros)
    assert_eq!(frame.as_slice()[0], Rgba::default());
}

#[test]
#[should_panic]
fn test_frame_from_data_panic() {
    let data = vec![Rgba::default(); 10].into_boxed_slice();
    Frame::from_data(data, 10, 2); // 10 * 2 = 20 != 10
}

#[test]
fn test_frame_convert() {
    // R: 0x11, G: 0x22, B: 0x33, A: 0xFF
    let rgba = Rgba::new(0x11, 0x22, 0x33, 0xFF);
    let mut frame = Frame::<Rgba>::new(1, 1);
    frame.as_slice_mut()[0] = rgba;

    let bgra_frame: Frame<Bgra> = frame.convert();
    let bgra = bgra_frame.as_slice()[0];

    assert_eq!(bgra.b(), 0x33);
    assert_eq!(bgra.g(), 0x22);
    assert_eq!(bgra.r(), 0x11);
    assert_eq!(bgra.a(), 0xFF);
}

#[test]
fn test_frame_accessors() {
    let mut frame = Frame::<Rgba>::new(2, 2);
    // Fill with pattern
    // 0: R, 1: G, 2: B, 3: A
    frame.as_slice_mut()[0] = Rgba::new(255, 0, 0, 255);
    frame.as_slice_mut()[1] = Rgba::new(0, 255, 0, 255);
    frame.as_slice_mut()[2] = Rgba::new(0, 0, 255, 255);
    frame.as_slice_mut()[3] = Rgba::new(255, 255, 255, 255);

    let bytes = frame.as_bytes();
    assert_eq!(bytes.len(), 4 * 4); // 4 pixels * 4 bytes

    // Check first pixel (Rgba: R, G, B, A in memory)
    // Little endian u32: 0xAABBGGRR -> bytes [RR, GG, BB, AA]
    assert_eq!(bytes[0], 255); // R
    assert_eq!(bytes[1], 0);   // G
    assert_eq!(bytes[2], 0);   // B
    assert_eq!(bytes[3], 255); // A

    let u32s = frame.as_u32_slice();
    assert_eq!(u32s.len(), 4);
    assert_eq!(u32s[0], Rgba::new(255, 0, 0, 255).0);
}

#[test]
fn test_frame_surface_sampling() {
    let mut frame = Frame::<Rgba>::new(2, 2);
    // Layout:
    // 0: (0,0)  1: (1,0)
    // 2: (0,1)  3: (1,1)
    let p0 = Rgba::new(10, 10, 10, 10);
    let p1 = Rgba::new(20, 20, 20, 20);
    let p2 = Rgba::new(30, 30, 30, 30);
    let p3 = Rgba::new(40, 40, 40, 40);

    frame.as_slice_mut()[0] = p0;
    frame.as_slice_mut()[1] = p1;
    frame.as_slice_mut()[2] = p2;
    frame.as_slice_mut()[3] = p3;

    // Test direct access
    let x = Batch::new(0, 1, 0, 1);
    let y = Batch::new(0, 0, 1, 1);
    let res = frame.eval(x, y);

    let res_u32: Batch<u32> = res.transmute();
    let arr = res_u32.to_array_usize();

    assert_eq!(arr[0] as u32, p0.0);
    assert_eq!(arr[1] as u32, p1.0);
    assert_eq!(arr[2] as u32, p2.0);
    assert_eq!(arr[3] as u32, p3.0);

    // Test wrapping
    // (2, 0) -> (0, 0) -> p0
    // (0, 2) -> (0, 0) -> p0
    // (3, 3) -> (1, 1) -> p3
    // (10, 10) -> (0, 0) -> p0

    let x_wrap = Batch::new(2, 0, 3, 10);
    let y_wrap = Batch::new(0, 2, 3, 10);
    let res_wrap: Batch<u32> = frame.eval(x_wrap, y_wrap).transmute();
    let arr_wrap = res_wrap.to_array_usize();

    assert_eq!(arr_wrap[0] as u32, p0.0);
    assert_eq!(arr_wrap[1] as u32, p0.0);
    assert_eq!(arr_wrap[2] as u32, p3.0);
    assert_eq!(arr_wrap[3] as u32, p0.0);
}

#[test]
#[should_panic]
fn test_frame_zero_width_panic() {
    let frame = Frame::<Rgba>::new(0, 100);
    let x = Batch::splat(0);
    let y = Batch::splat(0);
    // Should panic due to modulo by zero
    frame.eval(x, y);
}

#[test]
#[should_panic]
fn test_frame_zero_height_panic() {
    let frame = Frame::<Rgba>::new(100, 0);
    let x = Batch::splat(0);
    let y = Batch::splat(0);
    // Should panic due to modulo by zero
    frame.eval(x, y);
}
