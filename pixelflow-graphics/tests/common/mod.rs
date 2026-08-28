//! Shared golden-image test support.
//!
//! Format is PPM (P6, binary RGB) -- a 3-line text header followed by raw
//! bytes, chosen because it needs no crate to write *or* read: no `image`
//! dependency, no codec, just a header and a byte-for-byte comparison. This
//! mirrors the `write_ppm` helper already duplicated across
//! `e2e_render_to_file.rs`, `scene3d_test.rs`, and `raymarch_sphere.rs`.
//!
//! `assert_golden` renders once, then either:
//! - writes the render as the golden if none exists yet, or `UPDATE_GOLDENS=1`
//!   is set (the deliberate-regeneration escape hatch for an intentional
//!   rendering change), or
//! - compares against the stored golden within `tolerance` per channel, and
//!   on mismatch writes the actual render and a visual diff next to the
//!   golden so the failure is something to look at, not just a number.
//!
//! The per-channel tolerance exists because this codebase documents
//! (CLAUDE.md) that `MulAdd` rounding and similar float behavior legitimately
//! differs by target ISA -- a real miscompile shifts coverage/color by whole
//! steps, not by a rounding ULP.
//!
//! That alone isn't enough for scenes with hard edges (a checkerboard grid
//! line, a sphere's silhouette): a sub-ULP difference in an edge-distance
//! calculation can land a boundary pixel on the *other side* of a threshold,
//! flipping it to a completely different color -- a large delta on a single
//! pixel, not a small delta on the whole image. No per-channel tolerance can
//! absorb that, because the two colors either side of an edge can be
//! arbitrarily different. `assert_golden` therefore also budgets a small
//! *fraction* of pixels allowed to exceed the per-channel tolerance, on the
//! premise that a real regression moves far more than a sliver of boundary
//! pixels, while a few hundred edge pixels flipping between x86 and aarch64
//! (or even between two x86 ISA levels) is exactly the platform divergence
//! CLAUDE.md says to expect. Confirmed on this repo's own CI: the initial
//! goldens, generated on one machine, failed on both ubuntu-latest and
//! macos-latest at ~0.4% of pixels, all at checkerboard/sphere edges.

use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

pub fn write_ppm<P: AsRef<Path>>(path: P, frame: &Frame<Rgba8>) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", frame.width, frame.height)?;
    writeln!(file, "255")?;
    for pixel in &frame.data {
        file.write_all(&[pixel.r(), pixel.g(), pixel.b()])?;
    }
    Ok(())
}

struct Ppm {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

fn read_ppm<P: AsRef<Path>>(path: P) -> std::io::Result<Ppm> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    // Header is exactly the three lines `write_ppm` emits: "P6\n", "W H\n",
    // "255\n". Parse it by hand rather than pulling in a PPM-parsing crate --
    // the format we write is fixed, so this doesn't need to be general.
    let mut lines = 0;
    let mut header_end = 0;
    while lines < 3 {
        let newline = buf[header_end..]
            .iter()
            .position(|&b| b == b'\n')
            .expect("malformed PPM header");
        header_end += newline + 1;
        lines += 1;
    }
    let header = std::str::from_utf8(&buf[..header_end]).expect("PPM header is not UTF-8");
    let mut header_lines = header.lines();
    assert_eq!(header_lines.next(), Some("P6"), "not a P6 PPM");
    let dims = header_lines.next().expect("missing PPM dimensions line");
    let (w, h) = dims.split_once(' ').expect("malformed PPM dimensions");
    let width: usize = w.parse().expect("non-numeric PPM width");
    let height: usize = h.parse().expect("non-numeric PPM height");

    Ok(Ppm {
        width,
        height,
        rgb: buf[header_end..].to_vec(),
    })
}

fn frame_to_rgb(frame: &Frame<Rgba8>) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(frame.data.len() * 3);
    for pixel in &frame.data {
        rgb.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b()]);
    }
    rgb
}

/// Compare `frame` against the stored golden named `name`. A pixel counts as
/// mismatched if any RGB channel differs by more than `tolerance`; the test
/// fails only if more than `max_mismatched_fraction` of all pixels mismatch
/// (see the module docs for why a pure per-pixel tolerance isn't enough on
/// its own). Panics on a real mismatch, a missing golden (unless
/// `UPDATE_GOLDENS=1`), or a dimension mismatch.
pub fn assert_golden(
    name: &str,
    frame: &Frame<Rgba8>,
    tolerance: u8,
    max_mismatched_fraction: f64,
) {
    let golden_path = golden_dir().join(format!("{name}.ppm"));

    if std::env::var("UPDATE_GOLDENS").is_ok() {
        std::fs::create_dir_all(golden_dir()).expect("create golden dir");
        write_ppm(&golden_path, frame).expect("write golden");
        println!("UPDATE_GOLDENS: wrote {}", golden_path.display());
        return;
    }

    if !golden_path.exists() {
        panic!(
            "no golden at {} -- run with UPDATE_GOLDENS=1 once to create it, \
             inspect the output, and commit it",
            golden_path.display()
        );
    }

    let golden = read_ppm(&golden_path).expect("read golden");
    assert_eq!(
        (golden.width, golden.height),
        (frame.width, frame.height),
        "{name}: rendered dimensions don't match the golden's"
    );

    let actual_rgb = frame_to_rgb(frame);
    let mut worst_delta = 0u8;
    let mut mismatched_pixels = 0usize;
    let mut diff_rgb = vec![0u8; actual_rgb.len()];
    for (actual_px, golden_px) in actual_rgb.chunks_exact(3).zip(golden.rgb.chunks_exact(3)) {
        let mut pixel_mismatched = false;
        for c in 0..3 {
            let delta = actual_px[c].abs_diff(golden_px[c]);
            worst_delta = worst_delta.max(delta);
            if delta > tolerance {
                pixel_mismatched = true;
            }
        }
        if pixel_mismatched {
            mismatched_pixels += 1;
        }
    }
    // Amplify small deltas so a genuine but subtle regression is visible by
    // eye rather than reading as all-black; a separate pass since the loop
    // above only needs per-pixel data, not the amplified byte stream.
    for i in 0..actual_rgb.len() {
        diff_rgb[i] = actual_rgb[i].abs_diff(golden.rgb[i]).saturating_mul(8);
    }

    let total = actual_rgb.len() / 3;
    let mismatched_fraction = mismatched_pixels as f64 / total as f64;

    if mismatched_pixels == 0 {
        return;
    }
    if mismatched_fraction <= max_mismatched_fraction {
        println!(
            "{name}: {mismatched_pixels}/{total} pixels exceeded tolerance {tolerance} \
             (worst delta {worst_delta}), within the {max_mismatched_fraction} budget -- \
             treating as platform edge-pixel noise, not a regression"
        );
        return;
    }

    let out_dir = std::env::temp_dir().join("pixelflow_golden_failures");
    std::fs::create_dir_all(&out_dir).expect("create failure output dir");
    let actual_path = out_dir.join(format!("{name}.actual.ppm"));
    let diff_path = out_dir.join(format!("{name}.diff.ppm"));
    write_ppm(&actual_path, frame).expect("write actual");
    let diff_frame = Frame {
        width: frame.width,
        height: frame.height,
        data: diff_rgb
            .chunks_exact(3)
            .map(|c| Rgba8::new(c[0], c[1], c[2], 255))
            .collect(),
    };
    write_ppm(&diff_path, &diff_frame).expect("write diff");

    panic!(
        "{name}: {mismatched_pixels}/{total} pixels exceeded tolerance {tolerance} \
         (worst delta {worst_delta}) -- above the {max_mismatched_fraction} budget for \
         platform edge-pixel noise, so this looks like a real regression.\n\
         golden: {golden_path}\n\
         actual: {actual_path}\n\
         diff:   {diff_path} (amplified 8x so small deltas are visible)\n\
         If this change is intentional, rerun with UPDATE_GOLDENS=1 and commit the new golden.",
        golden_path = golden_path.display(),
        actual_path = actual_path.display(),
        diff_path = diff_path.display(),
    );
}
