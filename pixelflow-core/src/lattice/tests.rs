use super::*;
use pixelflow_ir::Kernel;

// The kernels the tabulation tests bake. Written as `Kernel` arithmetic —
// the arena the compiler consumes — because `Lattice::bake` is the only
// evaluation entry: there is nothing here for a hand-written `Manifold` to
// be handed to.

/// `X + Y`.
fn x_plus_y() -> Kernel {
    Kernel::x().add(&Kernel::y())
}

/// `X`, for the 1D shapes.
fn x_only() -> Kernel {
    Kernel::x()
}

/// `100·Z + 10·Y + X` — every axis readable in one digit of the result, so a
/// transposed or dropped axis in the 4D layout is visible in the value.
fn z_times_100() -> Kernel {
    Kernel::z()
        .mul(&Kernel::constant(100.0))
        .add(&Kernel::y().mul(&Kernel::constant(10.0)))
        .add(&Kernel::x())
}

// ---- Frame coord generation ----

#[test]
fn frame_coord_generation() {
    let lattice = Lattice {
        extent: [4, 3, 1, 1],
        origin: [0.0, 0.0, 1.5, 2.0],
    };

    assert_eq!(lattice.len(), 12);
    assert!(!lattice.is_empty());

    // First pixel: (0, 0, z, w)
    assert_eq!(lattice.coord(0), (0.0, 0.0, 1.5, 2.0));
    // End of first row: (3, 0, z, w)
    assert_eq!(lattice.coord(3), (3.0, 0.0, 1.5, 2.0));
    // Start of second row: (0, 1, z, w)
    assert_eq!(lattice.coord(4), (0.0, 1.0, 1.5, 2.0));
    // Last pixel: (3, 2, z, w)
    assert_eq!(lattice.coord(11), (3.0, 2.0, 1.5, 2.0));

    // Loop axes: X and Y
    assert_eq!(lattice.loop_mask(), 0b0011);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn frame_coord_oob() {
    let lattice = Lattice::frame(4, 3, 0.0);
    let _c = lattice.coord(12);
}

// ---- Scanline coord generation ----

#[test]
fn scanline_coord_generation() {
    let lattice = Lattice::scanline(8, 5.0, 1.0, 0.0);
    assert_eq!(lattice.len(), 8);
    assert_eq!(lattice.coord(0), (0.0, 5.0, 1.0, 0.0));
    assert_eq!(lattice.coord(7), (7.0, 5.0, 1.0, 0.0));
    assert_eq!(lattice.loop_mask(), 0b0001);
}

// ---- Point bake = a single value ----

#[test]
fn point_bake_is_a_single_value() {
    let lattice = Lattice::point(3.0, 4.0, 0.0, 0.0);
    assert_eq!(lattice.len(), 1);
    assert_eq!(lattice.loop_mask(), 0);
    assert_eq!(lattice.coord(0), (3.0, 4.0, 0.0, 0.0));

    let discrete = lattice.bake(&x_plus_y());
    assert_eq!(discrete.width(), 1);
    assert_eq!(discrete.height(), 1);

    // X + Y = 3 + 4 = 7
    let buf = discrete.buffer();
    assert_eq!(buf.len(), 1);
    assert!((buf[0] - 7.0).abs() < 1e-5, "expected 7.0, got {}", buf[0]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn point_coord_oob() {
    let lattice = Lattice::point(0.0, 0.0, 0.0, 0.0);
    let _c = lattice.coord(1);
}

// ---- DiscreteManifold round-trip ----

#[test]
fn discrete_manifold_round_trip() {
    // Bake X + Y over a small grid, then read the buffer back as a manifold.
    let lattice = Lattice::frame(8, 4, 0.0);
    let discrete = lattice.bake(&x_plus_y());

    assert_eq!(discrete.width(), 8);
    assert_eq!(discrete.height(), 4);
    assert_eq!(discrete.buffer().len(), 32);

    // Check known values: buffer[y * width + x] should equal x + y.
    for y in 0..4 {
        for x in 0..8 {
            let expected = (x + y) as f32;
            let actual = discrete.buffer()[y * 8 + x];
            assert!(
                (actual - expected).abs() < 1e-5,
                "at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                actual,
            );
        }
    }

    // Now eval the DiscreteManifold at known coordinates.
    // Querying at (2.0, 1.0) should return buffer[1*8 + 2] = 3.0.
    let result = discrete.eval((
        Field::from(2.0),
        Field::from(1.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    let mut out = [0.0f32; PARALLELISM];
    result.store(&mut out);
    assert!((out[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", out[0],);
}

// ---- Tail handling (non-multiple-of-PARALLELISM width) ----

#[test]
fn frame_bake_non_aligned_width() {
    // Width that's not a multiple of PARALLELISM.
    let width = PARALLELISM + 1;
    let lattice = Lattice::frame(width, 2, 0.0);
    let discrete = lattice.bake(&x_only());

    assert_eq!(discrete.buffer().len(), width * 2);

    // Each pixel at (x, y) should have value x.
    for y in 0..2 {
        for x in 0..width {
            let expected = x as f32;
            let actual = discrete.buffer()[y * width + x];
            assert!(
                (actual - expected).abs() < 1e-5,
                "at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                actual,
            );
        }
    }
}

// ---- DiscreteManifold clamp behavior ----

#[test]
fn discrete_manifold_clamp_oob_coords() {
    let buffer = alloc::vec![10.0, 20.0, 30.0, 40.0];
    let dm = DiscreteManifold::new(buffer, 2, 2);
    // Layout: (0,0)=10, (1,0)=20, (0,1)=30, (1,1)=40

    // Negative coords should clamp to 0.
    let result = dm.eval((
        Field::from(-5.0),
        Field::from(-5.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    let mut out = [0.0f32; PARALLELISM];
    result.store(&mut out);
    assert!(
        (out[0] - 10.0).abs() < 1e-5,
        "expected 10.0 (clamped to 0,0), got {}",
        out[0],
    );

    // Coords beyond max should clamp.
    let result = dm.eval((
        Field::from(100.0),
        Field::from(100.0),
        Field::from(0.0),
        Field::from(0.0),
    ));
    result.store(&mut out);
    assert!(
        (out[0] - 40.0).abs() < 1e-5,
        "expected 40.0 (clamped to 1,1), got {}",
        out[0],
    );
}

#[test]
#[should_panic(expected = "does not match dimensions")]
fn discrete_manifold_size_mismatch() {
    let _manifold = DiscreteManifold::new(alloc::vec![1.0, 2.0, 3.0], 2, 2);
}

// ---- Constructor shapes ----

#[test]
fn constructor_shapes() {
    let f = Lattice::frame(1920, 1080, 0.5);
    assert_eq!(f.extent, [1920, 1080, 1, 1]);
    assert_eq!(f.origin, [0.0, 0.0, 0.5, 0.0]);

    let i = Lattice::index(132);
    assert_eq!(i.extent, [132, 1, 1, 1]);
    assert_eq!(i.loop_mask(), 0b0001);

    let m = Lattice::index2(64, 32);
    assert_eq!(m.extent, [64, 32, 1, 1]);
    assert_eq!(m.loop_mask(), 0b0011);
}

// ---- Scanline bake round-trip ----

#[test]
fn scanline_bake_round_trip() {
    let lattice = Lattice::scanline(16, 3.0, 0.0, 0.0);
    let discrete = lattice.bake(&x_plus_y());

    // Each pixel x should have value x + 3.0.
    for x in 0..16 {
        let expected = x as f32 + 3.0;
        let actual = discrete.buffer()[x];
        assert!(
            (actual - expected).abs() < 1e-5,
            "at x={}: expected {}, got {}",
            x,
            expected,
            actual,
        );
    }
}

// ---- Empty lattice ----

#[test]
fn frame_zero_dimensions() {
    let lattice = Lattice::frame(0, 0, 0.0);
    assert!(lattice.is_empty());
    assert_eq!(lattice.len(), 0);

    // Baking over an empty lattice produces an empty discrete manifold.
    let discrete = lattice.bake(&Kernel::constant(42.0));
    assert_eq!(discrete.buffer().len(), 0);
    assert_eq!(discrete.width(), 0);
    assert_eq!(discrete.height(), 0);
}

// ---- Index-space lattices (feature/tensor indexing) ----

#[test]
fn index_bake_identity() {
    let lattice = Lattice::index(4);
    let result = lattice.bake(&x_only());
    assert_eq!(result.width(), 4);
    assert_eq!(result.height(), 1);
    let buf = result.buffer();
    assert!((buf[0] - 0.0).abs() < 1e-6);
    assert!((buf[1] - 1.0).abs() < 1e-6);
    assert!((buf[2] - 2.0).abs() < 1e-6);
    assert!((buf[3] - 3.0).abs() < 1e-6);
}

#[test]
fn sum_over_an_index_domain() {
    // Sum of [0,1,2,3] = 6. The reduction is a binder inside the kernel, so
    // the lattice that tabulates it is a single point.
    let k = Kernel::sum_over(4, |i| i.clone());
    let result = Lattice::point(0.0, 0.0, 0.0, 0.0).bake(&k).into_buffer()[0];
    assert!((result - 6.0).abs() < 1e-5, "expected 6.0, got {result}");
}

#[test]
fn index2_bake_xy_sum() {
    // 3x2 lattice (width=3, height=2). Values = X + Y.
    // Row 0 (Y=0): [0,1,2]. Row 1 (Y=1): [1,2,3].
    let lattice = Lattice::index2(3, 2);
    let result = lattice.bake(&x_plus_y());
    assert_eq!(result.width(), 3);
    assert_eq!(result.height(), 2);
    let buf = result.buffer();
    // Row-major: [y=0,x=0],[y=0,x=1],[y=0,x=2],[y=1,x=0],[y=1,x=1],[y=1,x=2]
    assert!((buf[0] - 0.0).abs() < 1e-6); // x=0, y=0
    assert!((buf[1] - 1.0).abs() < 1e-6); // x=1, y=0
    assert!((buf[2] - 2.0).abs() < 1e-6); // x=2, y=0
    assert!((buf[3] - 1.0).abs() < 1e-6); // x=0, y=1
    assert!((buf[4] - 2.0).abs() < 1e-6); // x=1, y=1
    assert!((buf[5] - 3.0).abs() < 1e-6); // x=2, y=1
}

#[test]
fn nested_sums_over_two_index_domains() {
    // Sum over a 3x2 domain of (i + j) = 0+1+2 + 1+2+3 = 9. Two binders, so
    // the inner index is contracted before the outer fold sees it.
    let k = Kernel::sum_over(2, |j| {
        let j = j.clone();
        Kernel::sum_over(3, move |i| i.add(&j))
    });
    let result = Lattice::point(0.0, 0.0, 0.0, 0.0).bake(&k).into_buffer()[0];
    assert!((result - 9.0).abs() < 1e-5, "expected 9.0, got {result}");
}

// ---- 4D box bake (Z/W extents > 1) ----

#[test]
fn box_bake_4d_layout() {
    // 2 wide, 2 tall, 2 deep: buffer rows are (w, z, y) outer-to-inner.
    let lattice = Lattice {
        extent: [2, 2, 2, 1],
        origin: [0.0; 4],
    };
    assert_eq!(lattice.len(), 8);
    assert_eq!(lattice.loop_mask(), 0b0111);

    let discrete = lattice.bake(&z_times_100());
    assert_eq!(discrete.width(), 2);
    assert_eq!(discrete.height(), 4);
    let buf = discrete.buffer();
    // Rows in order: (z=0,y=0), (z=0,y=1), (z=1,y=0), (z=1,y=1)
    let expected = [0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (buf[i] - e).abs() < 1e-5,
            "at {}: expected {}, got {}",
            i,
            e,
            buf[i]
        );
    }
}

// ============================================================================
// BilinearSampler: JIT'd 4-tap read-back semantics
// ============================================================================

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod bilinear_sampler {
    use super::*;
    use crate::BilinearSampler;

    /// One point of the sampler, evaluated the way its production caller
    /// does — `CachedGlyph::eval` reads it a batch at a time — with lane 0
    /// read back here. A test owning its batch, not an evaluation API:
    /// `Lattice::bake` cannot tabulate this, because the sampler's texture
    /// is bound memory and `bake` binds none.
    fn sample(s: &BilinearSampler, x: f32, y: f32) -> f32 {
        let zero = Field::from(0.0);
        let batch = s.eval((Field::from(x), Field::from(y), zero, zero));
        let mut lanes = [0.0f32; PARALLELISM];
        batch.store(&mut lanes);
        lanes[0]
    }

    #[test]
    fn constant_field_is_identity() {
        // A 3x3 constant grid: bilinear must return the constant everywhere.
        let s = DiscreteManifold::new(vec![0.75; 9], 3, 3).bilinear();

        for &(x, y) in &[(0.0, 0.0), (0.5, 0.5), (1.25, 0.75), (1.9, 1.1)] {
            let v = sample(&s, x, y);
            assert!(
                (v - 0.75).abs() < 1e-6,
                "constant field not reproduced at ({x}, {y}): got {v}"
            );
        }
    }

    #[test]
    fn linear_gradient_reproduced_exactly() {
        // Buffer holding f(x, y) = x + 2y sampled at integer coords, 4x4.
        let mut buf = Vec::with_capacity(16);
        for y in 0..4 {
            for x in 0..4 {
                buf.push(x as f32 + 2.0 * y as f32);
            }
        }
        let s = DiscreteManifold::new(buf, 4, 4).bilinear();

        // Bilinear interpolation of an affine function is exact.
        for &(x, y) in &[(0.5, 0.5), (1.25, 2.75), (0.1, 0.9), (2.5, 1.0)] {
            let expect = x + 2.0 * y;
            let v = sample(&s, x, y);
            assert!(
                (v - expect).abs() < 1e-5,
                "gradient not reproduced at ({x}, {y}): got {v}, want {expect}"
            );
        }
    }

    #[test]
    fn exact_at_grid_points_blended_between() {
        // 2x2 checker: (0,0)=0, (1,0)=1, (0,1)=1, (1,1)=0.
        let s = DiscreteManifold::new(vec![0.0, 1.0, 1.0, 0.0], 2, 2).bilinear();

        // At integer grid points: exact stored values, no smoothing.
        assert!((sample(&s, 0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((sample(&s, 1.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((sample(&s, 0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((sample(&s, 1.0, 1.0) - 0.0).abs() < 1e-6);

        // Midpoint of an edge: average of its two endpoints.
        assert!((sample(&s, 0.5, 0.0) - 0.5).abs() < 1e-6);
        // Cell center: average of all four corners.
        assert!((sample(&s, 0.5, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn no_nearest_jumps_across_cell_boundaries() {
        // Step buffer: left column 0, right column 1. Nearest-neighbor would
        // jump 0 -> 1 crossing x = 1; bilinear must ramp smoothly.
        let s = DiscreteManifold::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 2).bilinear();

        let step = 0.125;
        let mut prev = sample(&s, 0.0, 0.5);
        let mut x = step;
        while x <= 2.0 {
            let v = sample(&s, x, 0.5);
            let jump = (v - prev).abs();
            assert!(
                jump < 0.25,
                "non-smooth jump {jump} at x = {x} (bilinear should ramp, not step)"
            );
            prev = v;
            x += step;
        }
        // And the ramp actually reaches the step's top value.
        assert!((sample(&s, 2.0, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_taps_clamp_to_edge() {
        // Queries outside the grid clamp to the edge texel, matching
        // DiscreteManifold::eval's convention.
        let s = DiscreteManifold::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).bilinear();

        assert!((sample(&s, -5.0, -5.0) - 1.0).abs() < 1e-6);
        assert!((sample(&s, 10.0, -5.0) - 2.0).abs() < 1e-6);
        assert!((sample(&s, -5.0, 10.0) - 3.0).abs() < 1e-6);
        assert!((sample(&s, 10.0, 10.0) - 4.0).abs() < 1e-6);
    }
}
