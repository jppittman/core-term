//! Automation tasks for the project.
//!
//! Commands:
//! - `bundle-run`: Build and run the bundled macOS app
//! - `bake-eigen`: Parse Stam's eigenstructure binary and generate Rust consts

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;

/// Entry point for xtask.
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  cargo bundle-run [cargo build options]");
        eprintln!("  cargo bake-eigen");
        eprintln!("Commands:");
        eprintln!("  bundle-run    Build and run the bundled macOS app");
        eprintln!("  bake-eigen    Parse Stam's eigenstructure binary → Rust consts");
        eprintln!("  isa-matrix    Run the workspace test suite at every x86-64 ISA");
        eprintln!("                level this host actually supports (SSE2/AVX2/AVX-512)");
        eprintln!("                [--clippy to also run clippy per level]");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "bundle-run" => {
            // Pass through any additional arguments after "bundle-run"
            let extra_args = if args.len() > 2 { &args[2..] } else { &[] };
            bundle_run(extra_args);
        }
        "bake-eigen" => {
            bake_eigen();
        }
        "isa-matrix" => {
            let with_clippy = args[2..].iter().any(|a| a == "--clippy");
            isa_matrix(with_clippy);
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}

/// Find the workspace root by looking for Cargo.toml with [workspace]
fn find_workspace_root() -> PathBuf {
    let mut current = env::current_dir().expect("Failed to get current directory");

    loop {
        let cargo_toml = current.join("Cargo.toml");

        if cargo_toml.exists() {
            // Check if this is the workspace root by reading Cargo.toml
            if let Ok(contents) = fs::read_to_string(&cargo_toml) {
                if contents.contains("[workspace]") {
                    return current;
                }
            }
        }

        // Move up to parent directory
        if !current.pop() {
            eprintln!("Could not find workspace root (no Cargo.toml with [workspace] found)");
            std::process::exit(1);
        }
    }
}

/// Builds the project in release mode and bundles it into a macOS .app structure.
/// Then launches the application.
///
/// # Parameters
/// * `extra_args` - Additional arguments to pass to `cargo build`.
fn bundle_run(extra_args: &[String]) {
    // Find workspace root so this works from any subdirectory
    let workspace_root = find_workspace_root();
    println!("Workspace root: {}", workspace_root.display());

    println!("Building core-term in release mode (opt-level=3, LTO)...");

    // Build the project with extra args (e.g., --features profiling)
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&workspace_root); // Run from workspace root
    cmd.args(["build", "--release", "-p", "core-term"]);

    // Filter out --release since we already added it
    let filtered_args: Vec<&String> = extra_args
        .iter()
        .filter(|arg| arg.as_str() != "--release")
        .collect();

    if !filtered_args.is_empty() {
        println!("Additional build args: {:?}", filtered_args);
        cmd.args(&filtered_args);
    }

    let status = cmd.status().expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("Build failed");
        std::process::exit(1);
    }

    // Copy binary to bundle (build.rs creates the bundle structure)
    let binary_src = workspace_root.join("target/release/core-term");
    let binary_dest = workspace_root.join("CoreTerm.app/Contents/MacOS/CoreTerm");

    if !binary_src.exists() {
        eprintln!("Binary not found at {}", binary_src.display());
        std::process::exit(1);
    }

    // Verify binary size - release with LTO should be reasonably sized
    let binary_size = fs::metadata(&binary_src)
        .expect("Failed to get binary metadata")
        .len();
    println!(
        "Binary size: {:.2} MB (release with LTO)",
        binary_size as f64 / (1024.0 * 1024.0)
    );

    println!("Copying binary to bundle...");
    fs::copy(&binary_src, &binary_dest).expect("Failed to copy binary to bundle");

    // Make it executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&binary_dest)
            .expect("Failed to get binary metadata")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&binary_dest, perms).expect("Failed to set executable permission");
    }

    // Copy icon file to bundle Resources
    let icon_src = workspace_root.join("assets/icons/icon.icns");
    let resources_dir = workspace_root.join("CoreTerm.app/Contents/Resources");
    let icon_dest = resources_dir.join("icon.icns");

    fs::create_dir_all(&resources_dir).expect("Failed to create Resources directory");

    if icon_src.exists() {
        println!("Copying icon to bundle...");
        fs::copy(&icon_src, &icon_dest).expect("Failed to copy icon to bundle");
    } else {
        println!("Warning: Icon not found at {}", icon_src.display());
    }

    // Copy font file to bundle Resources
    let font_src = workspace_root.join("pixelflow-graphics/assets/NotoSansMono-Regular.ttf");
    let font_dest = resources_dir.join("NotoSansMono-Regular.ttf");

    if font_src.exists() {
        println!("Copying font to bundle...");
        fs::copy(&font_src, &font_dest).expect("Failed to copy font to bundle");
    } else {
        eprintln!("ERROR: Font not found at {}", font_src.display());
        std::process::exit(1);
    }

    // Touch the app bundle to invalidate macOS icon cache
    let app_bundle = workspace_root.join("CoreTerm.app");
    println!("Refreshing app bundle metadata...");
    Command::new("touch")
        .arg(&app_bundle)
        .status()
        .expect("Failed to touch app bundle");

    println!("Launching CoreTerm.app...");
    println!("Logs will be written to /tmp/core-term.log");

    // Launch the bundled app using 'open'
    // Logs are written to /tmp/core-term.log (configured in main.rs)
    let status = Command::new("open")
        .arg(&app_bundle)
        .status()
        .expect("Failed to launch app");

    if !status.success() {
        eprintln!("Failed to launch CoreTerm.app");
        std::process::exit(1);
    }

    println!("CoreTerm.app launched successfully!");
    println!("Monitor logs with: tail -f /tmp/core-term.log");
}

// ============================================================================
// ISA test matrix
// ============================================================================
//
// `.cargo/config.toml` sets no `target-cpu`/`target-feature`, so a plain
// `cargo build`/`cargo test` always produces SSE2 code on x86-64 — even on a
// machine with AVX2 or AVX-512 available. Nothing in the default CI pipeline
// ever exercised the wider JIT backends. This runs the workspace test suite
// (and optionally clippy) once per x86-64 ISA level the CURRENT HOST actually
// supports, via explicit `-C target-feature` flags rather than
// `-C target-cpu=native` (which bakes in whatever the build machine happens
// to be and isn't reproducible across machines/CI runners).

/// One row of the ISA test matrix: a human-readable name, the
/// `-C target-feature` value to pass (empty = the SSE2 baseline, no flag),
/// and the `is_x86_feature_detected!` names that must all be present on this
/// host to attempt the level at all.
struct IsaLevel {
    name: &'static str,
    target_feature: &'static str,
    requires: &'static [&'static str],
}

const ISA_LEVELS: &[IsaLevel] = &[
    IsaLevel {
        name: "sse2 (baseline)",
        target_feature: "",
        requires: &[],
    },
    // Two AVX2 rows, because `Avx2Backend` is gated on `avx2` ALONE and carries
    // a documented software mul+add fallback for parts that shipped AVX2
    // without FMA3 (VIA, early Zen). Requiring both features would skip this
    // configuration entirely on such a host and never exercise that fallback —
    // so the matrix would not, in fact, cover every level the host supports.
    IsaLevel {
        name: "avx2 (no fma)",
        target_feature: "+avx2",
        requires: &["avx2"],
    },
    IsaLevel {
        name: "avx2+fma",
        target_feature: "+avx2,+fma",
        requires: &["avx2", "fma"],
    },
    IsaLevel {
        // DQ, not just F: `avx512::emit_compare` materializes a mask with
        // `vpmovm2d`, which is AVX-512DQ. An AVX-512F-only part (Knights
        // Landing) would take an illegal-instruction fault on any kernel
        // containing a comparison, so probing for F alone would "support" a
        // level that cannot run.
        name: "avx512f+dq",
        target_feature: "+avx512f,+avx512dq",
        requires: &["avx512f", "avx512dq"],
    },
];

/// Whether the host CPU has all the features an [`IsaLevel`] requires.
/// `is_x86_feature_detected!` only accepts a literal feature name, so this is
/// a fixed match rather than a generic lookup — extend it if `ISA_LEVELS`
/// grows a level needing a feature not listed here.
#[cfg(target_arch = "x86_64")]
fn host_has_feature(feature: &str) -> bool {
    match feature {
        "avx2" => std::is_x86_feature_detected!("avx2"),
        "fma" => std::is_x86_feature_detected!("fma"),
        "avx512f" => std::is_x86_feature_detected!("avx512f"),
        "avx512dq" => std::is_x86_feature_detected!("avx512dq"),
        other => panic!("isa-matrix: unknown feature {other:?} in ISA_LEVELS::requires"),
    }
}

/// Outcome of attempting one [`IsaLevel`].
enum LevelResult {
    /// Test suite (and clippy, if requested) both succeeded.
    Passed,
    /// A command ran and failed; the bool distinguishes which one for the summary.
    Failed { clippy: bool },
    /// Host lacks a required CPU feature; skipped cleanly, not a failure.
    Skipped { missing: &'static str },
}

/// Run the workspace test suite (`cargo test --workspace`), and optionally
/// `cargo clippy --workspace --all-targets -- -D warnings`, at every x86-64
/// ISA level this host supports. Non-x86-64 hosts (aarch64/NEON) have a
/// single ISA level already, so there is nothing to matrix — this prints a
/// note and exits 0 rather than silently doing nothing.
fn isa_matrix(with_clippy: bool) {
    let workspace_root = find_workspace_root();

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = workspace_root;
        println!(
            "isa-matrix: host is not x86-64 (no SSE2/AVX2/AVX-512 split to test here — \
             e.g. aarch64/NEON has one ISA level already)."
        );
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        println!("isa-matrix: workspace root {}", workspace_root.display());
        // The base flag every build in this repo already gets from
        // `.cargo/config.toml`'s `[build] rustflags`. Setting RUSTFLAGS in the
        // environment REPLACES (does not merge with) that config-file value,
        // so every level below must repeat it explicitly.
        const FP_CONTRACT: &str = "-C llvm-args=-fp-contract=fast";

        let mut results: Vec<(&str, LevelResult)> = Vec::new();

        for level in ISA_LEVELS {
            println!("\n=== ISA level: {} ===", level.name);

            let missing = level
                .requires
                .iter()
                .find(|&&feat| !host_has_feature(feat));
            if let Some(&feat) = missing {
                println!(
                    "isa-matrix: skipping {} — host CPU lacks {feat} \
                     (checked via is_x86_feature_detected!)",
                    level.name
                );
                results.push((level.name, LevelResult::Skipped { missing: feat }));
                continue;
            }

            let rustflags = if level.target_feature.is_empty() {
                FP_CONTRACT.to_string()
            } else {
                format!("{FP_CONTRACT} -C target-feature={}", level.target_feature)
            };
            println!("isa-matrix: RUSTFLAGS=\"{rustflags}\"");

            // Each level's RUSTFLAGS produce a disjoint artifact set (the
            // flags feed -C metadata), so levels sharing a target dir
            // accumulate ~one full workspace build EACH — three levels
            // exhausted a 21 GB disk the first time this ran for real.
            // Every level therefore builds in one dedicated dir, wiped
            // before each level: peak disk is a single artifact set, and
            // the developer's own target/ (their incremental cache) is
            // never touched.
            let matrix_target = workspace_root.join("target/isa-matrix");
            if matrix_target.exists() {
                if let Err(e) = std::fs::remove_dir_all(&matrix_target) {
                    panic!(
                        "isa-matrix: could not clear {}: {e}",
                        matrix_target.display()
                    );
                }
            }

            let test_ok = run_with_rustflags(
                &workspace_root,
                &matrix_target,
                &rustflags,
                &["test", "--workspace", "--no-fail-fast"],
            );
            if !test_ok {
                println!("isa-matrix: {} — cargo test FAILED", level.name);
                results.push((level.name, LevelResult::Failed { clippy: false }));
                continue;
            }
            println!("isa-matrix: {} — cargo test passed", level.name);

            if with_clippy {
                let clippy_ok = run_with_rustflags(
                    &workspace_root,
                    &matrix_target,
                    &rustflags,
                    &["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
                );
                if !clippy_ok {
                    println!("isa-matrix: {} — cargo clippy FAILED", level.name);
                    results.push((level.name, LevelResult::Failed { clippy: true }));
                    continue;
                }
                println!("isa-matrix: {} — cargo clippy passed", level.name);
            }

            results.push((level.name, LevelResult::Passed));
        }

        println!("\n=== ISA matrix summary ===");
        let mut any_failed = false;
        for (name, result) in &results {
            let line = match result {
                LevelResult::Passed => "PASS".to_string(),
                LevelResult::Failed { clippy } => {
                    any_failed = true;
                    if *clippy {
                        "FAIL (clippy)".to_string()
                    } else {
                        "FAIL (test)".to_string()
                    }
                }
                LevelResult::Skipped { missing } => {
                    format!("SKIP (host lacks {missing})")
                }
            };
            println!("  {name:<20} {line}");
        }

        if any_failed {
            std::process::exit(1);
        }
    }
}

/// Run `cargo <args>` from the workspace root with `RUSTFLAGS` set to
/// `rustflags` and artifacts confined to `target_dir`, streaming output
/// straight through. Returns whether it succeeded.
#[cfg(target_arch = "x86_64")]
fn run_with_rustflags(
    workspace_root: &std::path::Path,
    target_dir: &std::path::Path,
    rustflags: &str,
    args: &[&str],
) -> bool {
    Command::new("cargo")
        .current_dir(workspace_root)
        .args(args)
        .env("RUSTFLAGS", rustflags)
        .env("CARGO_TARGET_DIR", target_dir)
        // Deep-Manifold tests recurse near the stack limit by design (see
        // CLAUDE.md's dev-profile note), and 8-/16-lane builds have
        // proportionally larger frames. This raises the floor for libtest's
        // own threads; worker threads that set `stack_size` explicitly are
        // NOT covered by it and must size themselves (see
        // `rasterizer::parallel::STACK_SIZE`).
        .env("RUST_MIN_STACK", "16777216")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ============================================================================
// Eigenstructure Baking
// ============================================================================

/// Parse Stam's ccdata50NT.dat and generate Rust const arrays.
///
/// Binary format (little-endian):
/// - Header: i32 Nmax (maximum valence, typically 50)
/// - Per valence N (3..=Nmax):
///   - K = 2N + 8 eigenvalues (f64)
///   - K×K inverse eigenvector matrix (f64, row-major)
///   - 3 sets of K×16 spline coefficients (f64)
fn bake_eigen() {
    let workspace_root = find_workspace_root();
    let input_path = workspace_root.join("pixelflow-graphics/assets/ccdata50NT.dat");
    let output_path = workspace_root.join("pixelflow-graphics/src/subdiv/coeffs.rs");

    println!("Reading eigenstructure from: {}", input_path.display());

    let mut file = fs::File::open(&input_path).expect("Failed to open ccdata50NT.dat");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read file");

    // Parse header: Nmax as i32 (little-endian)
    let nmax = i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    println!("Maximum valence: {}", nmax);

    let mut offset = 4; // Skip header

    // Collect all eigenstructures
    let mut structures = Vec::new();

    for valence in 3..=nmax {
        let k = 2 * valence + 8; // Number of eigenvalues/bases

        // Read eigenvalues: K f64s
        let mut eigenvalues = Vec::with_capacity(k);
        for _ in 0..k {
            let val = read_f64_le(&data, offset);
            eigenvalues.push(val as f32);
            offset += 8;
        }

        // Read inverse eigenvector matrix: K×K f64s (row-major)
        let mut inv_eigenvectors = Vec::with_capacity(k * k);
        for _ in 0..(k * k) {
            let val = read_f64_le(&data, offset);
            inv_eigenvectors.push(val as f32);
            offset += 8;
        }

        // Read spline coefficients: 3 subpatches × K bases × 16 coeffs
        let mut spline_coeffs = vec![vec![vec![0.0f32; 16]; k]; 3];
        for subpatch_mut in spline_coeffs.iter_mut() {
            for basis_mut in subpatch_mut.iter_mut() {
                for coeff_ref in basis_mut.iter_mut() {
                    let val = read_f64_le(&data, offset);
                    *coeff_ref = val as f32;
                    offset += 8;
                }
            }
        }

        structures.push(EigenData {
            valence,
            k,
            eigenvalues,
            inv_eigenvectors,
            spline_coeffs,
        });
    }

    println!("Parsed {} valences", structures.len());
    println!("Generating Rust source: {}", output_path.display());

    // Generate Rust source
    let mut out = String::new();
    out.push_str("//! Baked Catmull-Clark eigenstructure coefficients.\n");
    out.push_str("//!\n");
    out.push_str("//! Auto-generated by `cargo bake-eigen` from ccdata50NT.dat.\n");
    out.push_str("//! Do not edit manually.\n");
    out.push_str("//!\n");
    out.push_str("//! Source: Stam, \"Exact Evaluation of Catmull-Clark Subdivision Surfaces\"\n");
    out.push_str(
        "//! Data from: https://www.dgp.toronto.edu/~stam/reality/Research/SubdivEval/\n\n",
    );

    out.push_str("/// Maximum supported valence.\n");
    out.push_str(&format!("pub const MAX_VALENCE: usize = {};\n\n", nmax));

    out.push_str("/// Eigenstructure data for a specific valence.\n");
    out.push_str("#[derive(Clone, Debug)]\n");
    out.push_str("pub struct EigenCoeffs {\n");
    out.push_str("    /// Valence (number of edges at extraordinary vertex)\n");
    out.push_str("    pub valence: usize,\n");
    out.push_str("    /// K = 2N + 8 (number of eigenvalues/bases)\n");
    out.push_str("    pub k: usize,\n");
    out.push_str("    /// Eigenvalues (K values)\n");
    out.push_str("    pub eigenvalues: &'static [f32],\n");
    out.push_str("    /// Inverse eigenvector matrix (K×K, row-major)\n");
    out.push_str("    pub inv_eigenvectors: &'static [f32],\n");
    out.push_str("    /// Spline coefficients [subpatch][basis][coeff] flattened\n");
    out.push_str("    /// Layout: 3 subpatches × K bases × 16 bicubic coeffs\n");
    out.push_str("    pub spline_coeffs: &'static [f32],\n");
    out.push_str("}\n\n");

    out.push_str("impl EigenCoeffs {\n");
    out.push_str("    /// Get spline coefficient for subpatch, basis, and coefficient index.\n");
    out.push_str("    #[inline]\n");
    out.push_str(
        "    pub fn spline(&self, subpatch: usize, basis: usize, coeff: usize) -> f32 {\n",
    );
    out.push_str("        self.spline_coeffs[subpatch * self.k * 16 + basis * 16 + coeff]\n");
    out.push_str("    }\n\n");
    out.push_str("    /// Get inverse eigenvector matrix element.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub fn inv_eigen(&self, row: usize, col: usize) -> f32 {\n");
    out.push_str("        self.inv_eigenvectors[row * self.k + col]\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    // Generate const arrays for each valence
    for s in &structures {
        let prefix = format!("V{}", s.valence);

        // Eigenvalues
        out.push_str(&format!(
            "const {}_EIGENVALUES: [f32; {}] = {};\n",
            prefix,
            s.k,
            format_f32_array(&s.eigenvalues)
        ));

        // Inverse eigenvectors
        out.push_str(&format!(
            "const {}_INV_EIGEN: [f32; {}] = {};\n",
            prefix,
            s.k * s.k,
            format_f32_array(&s.inv_eigenvectors)
        ));

        // Spline coeffs (flattened: 3 × K × 16)
        let mut flat_splines = Vec::with_capacity(3 * s.k * 16);
        for subpatch in 0..3 {
            for basis in 0..s.k {
                flat_splines.extend_from_slice(&s.spline_coeffs[subpatch][basis]);
            }
        }
        out.push_str(&format!(
            "const {}_SPLINES: [f32; {}] = {};\n\n",
            prefix,
            3 * s.k * 16,
            format_f32_array(&flat_splines)
        ));
    }

    // Generate lookup table
    out.push_str("/// Get eigenstructure for a given valence (3..=50).\n");
    out.push_str("pub fn get_eigen(valence: usize) -> Option<EigenCoeffs> {\n");
    out.push_str("    match valence {\n");
    for s in &structures {
        out.push_str(&format!(
            "        {} => Some(EigenCoeffs {{\n            valence: {},\n            k: {},\n            eigenvalues: &V{}_EIGENVALUES,\n            inv_eigenvectors: &V{}_INV_EIGEN,\n            spline_coeffs: &V{}_SPLINES,\n        }}),\n",
            s.valence, s.valence, s.k, s.valence, s.valence, s.valence
        ));
    }
    out.push_str("        _ => None,\n");
    out.push_str("    }\n");
    out.push_str("}\n");

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create subdiv directory");
    }

    let mut out_file = fs::File::create(&output_path).expect("Failed to create output file");
    out_file
        .write_all(out.as_bytes())
        .expect("Failed to write output");

    println!("Generated {} bytes of Rust source", out.len());
    println!("Done! Run `cargo fmt -p pixelflow-graphics` to format.");
}

/// Read f64 little-endian from byte slice.
fn read_f64_le(data: &[u8], offset: usize) -> f64 {
    let bytes: [u8; 8] = data[offset..offset + 8].try_into().unwrap();
    f64::from_le_bytes(bytes)
}

/// Format f32 array as Rust literal.
fn format_f32_array(values: &[f32]) -> String {
    let mut s = String::from("[\n    ");
    for (i, v) in values.iter().enumerate() {
        if i > 0 && i % 8 == 0 {
            s.push_str("\n    ");
        }
        // Use enough precision to round-trip
        s.push_str(&format!("{:e}, ", v));
    }
    s.push_str("\n]");
    s
}

/// Temporary struct for collecting parsed data.
struct EigenData {
    valence: usize,
    k: usize,
    eigenvalues: Vec<f32>,
    inv_eigenvectors: Vec<f32>,
    spline_coeffs: Vec<Vec<Vec<f32>>>,
}
