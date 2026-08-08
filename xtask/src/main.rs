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
        eprintln!("  isa-matrix    Build+lint every x86-64 ISA level (SSE2/AVX2/AVX-512);");
        eprintln!("                run the tests for those this host can execute");
        eprintln!("                [--clippy to also run clippy per level]");
        eprintln!("                [--build-only to skip test execution entirely --");
        eprintln!("                 presubmit's fast path; postsubmit runs the tests]");
        eprintln!("  launch-stability-check");
        eprintln!("                Build, bundle, and launch CoreTerm.app repeatedly,");
        eprintln!("                failing if any launch crashes");
        eprintln!("                [--runs N] [--watch-seconds S]");
        eprintln!("  bundle        Build and assemble CoreTerm.app WITHOUT launching it");
        eprintln!("                (for CI signing/notarization; see bundle-run to also launch)");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "bundle-run" => {
            // Pass through any additional arguments after "bundle-run"
            let extra_args = if args.len() > 2 { &args[2..] } else { &[] };
            bundle_run(extra_args);
        }
        "bundle" => {
            let extra_args = if args.len() > 2 { &args[2..] } else { &[] };
            let (_workspace_root, app_bundle) = build_and_bundle(extra_args);
            println!("Bundled (not launched): {}", app_bundle.display());
        }
        "bake-eigen" => {
            bake_eigen();
        }
        "isa-matrix" => {
            let with_clippy = args[2..].iter().any(|a| a == "--clippy");
            let mode = if args[2..].iter().any(|a| a == "--build-only") {
                IsaExecutionMode::BuildOnly
            } else {
                IsaExecutionMode::BuildAndTest
            };
            isa_matrix(with_clippy, mode);
        }
        "launch-stability-check" => {
            let runs =
                parse_u32_flag(&args[2..], "--runs").unwrap_or(LAUNCH_STABILITY_DEFAULT_RUNS);
            let watch_seconds = parse_u32_flag(&args[2..], "--watch-seconds")
                .map(u64::from)
                .unwrap_or(LAUNCH_STABILITY_DEFAULT_WATCH_SECONDS);
            launch_stability_check(runs, watch_seconds);
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}

/// Parses `--flag VALUE` out of an argument slice, if present.
fn parse_u32_flag(args: &[String], flag: &str) -> Option<u32> {
    let idx = args.iter().position(|a| a == flag)?;
    args.get(idx + 1)?.parse().ok()
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

/// Builds the project in release mode and assembles the macOS .app bundle,
/// without launching it. Shared by `bundle-run` (build + bundle + launch
/// once, for interactive dev use) and `launch-stability-check` (build +
/// bundle + launch repeatedly, for CI).
///
/// # Parameters
/// * `extra_args` - Additional arguments to pass to `cargo build`.
///
/// Returns `(workspace_root, app_bundle_path)`.
fn build_and_bundle(extra_args: &[String]) -> (PathBuf, PathBuf) {
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

    (workspace_root, app_bundle)
}

/// Builds the project in release mode and bundles it into a macOS .app structure.
/// Then launches the application.
///
/// # Parameters
/// * `extra_args` - Additional arguments to pass to `cargo build`.
fn bundle_run(extra_args: &[String]) {
    let (_workspace_root, app_bundle) = build_and_bundle(extra_args);

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
// macOS launch stability check
// ============================================================================
//
// MetalOps hand-pumps NSApplication's event queue (nextEventMatchingMask)
// instead of calling -[NSApplication run], because the actor scheduler needs
// to interleave event pumping with message handling on the main thread (see
// pixelflow-runtime/src/platform/macos/platform.rs). A background thread
// that called into AppKit unsynchronized with that pump (the original
// CocoaWaker::wake, see pixelflow-runtime/src/platform/waker.rs) corrupted
// AppKit's internal main-event-queue bookkeeping, which AppKit only actually
// noticed and asserted on later: observed 14-22s after launch, roughly 2 out
// of every 3 raw launches -- well after the window a human tester normally
// watches, and invisible to every other CI job. It needs a real window
// server session (a Linux runner has none at all) and a real LaunchServices
// bundle launch (`cargo run`/`cargo test` spawn the binary directly,
// bypassing that launch path). This is what would have caught it, and what
// stops it from silently coming back.
const LAUNCH_STABILITY_DEFAULT_RUNS: u32 = 15;
// Real crashes were observed 14-22s after launch; watch a few seconds past
// that so a slow CI runner doesn't get a false pass by checking too early.
const LAUNCH_STABILITY_DEFAULT_WATCH_SECONDS: u64 = 25;

fn launch_stability_check(runs: u32, watch_seconds: u64) {
    let (_workspace_root, app_bundle) = build_and_bundle(&[]);
    let binary_match = "CoreTerm.app/Contents/MacOS/CoreTerm";

    let crash_dir = PathBuf::from(env::var("HOME").expect("HOME not set"))
        .join("Library/Logs/DiagnosticReports");

    // Anything at or after this instant is a crash from THIS check, not a
    // stale report from an earlier, unrelated run.
    let check_start = std::time::SystemTime::now();

    let mut failures: u32 = 0;

    for run in 1..=runs {
        let status = Command::new("open")
            .arg(&app_bundle)
            .status()
            .expect("Failed to launch app");
        if !status.success() {
            eprintln!("::error::run {run}: 'open' itself failed");
            failures += 1;
            continue;
        }

        // LaunchServices returns before the process spawns, and a cold CI
        // runner (first registration of the bundle, cold dyld cache) can
        // take several seconds — one fixed sleep misread that as a crash.
        // Poll with a deadline instead; a genuinely failed launch still
        // fails, just honestly.
        const APPEAR_DEADLINE_MS: u64 = 15_000;
        const APPEAR_POLL_MS: u64 = 250;
        let mut pid = None;
        let mut waited = 0u64;
        while waited < APPEAR_DEADLINE_MS {
            std::thread::sleep(std::time::Duration::from_millis(APPEAR_POLL_MS));
            waited += APPEAR_POLL_MS;
            pid = find_pid(binary_match);
            if pid.is_some() {
                break;
            }
        }
        let Some(pid) = pid else {
            eprintln!(
                "::error::run {run}: process never appeared within {APPEAR_DEADLINE_MS}ms of launch"
            );
            failures += 1;
            continue;
        };

        let mut died_at = None;
        for elapsed in 1..=watch_seconds {
            std::thread::sleep(std::time::Duration::from_secs(1));
            if !pid_alive(pid) {
                died_at = Some(elapsed);
                break;
            }
        }

        match died_at {
            Some(t) => {
                eprintln!(
                    "::error::run {run} (pid={pid}): process died after {t}s (expected to survive {watch_seconds}s)"
                );
                failures += 1;
            }
            None => {
                println!("run {run} (pid={pid}): survived {watch_seconds}s");
            }
        }

        kill_pid(pid);
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    println!();
    println!(
        "Checking for new crash reports under {}...",
        crash_dir.display()
    );
    let new_crashes = find_crash_reports_since(&crash_dir, check_start);
    if !new_crashes.is_empty() {
        eprintln!(
            "::error::{} new crash report(s) appeared during {runs} launches:",
            new_crashes.len()
        );
        for f in &new_crashes {
            eprintln!("::error::  {}", f.display());
        }
        failures += 1;
    }

    if failures > 0 {
        eprintln!("::error::macOS launch stability check FAILED ({failures} issue(s) across {runs} launches)");
        std::process::exit(1);
    }

    println!(
        "macOS launch stability check passed: {runs}/{runs} launches survived {watch_seconds}s with no crash reports."
    );
}

/// Finds the PID of a running process by matching `pattern` against its full
/// command line, mirroring `pgrep -f`.
fn find_pid(pattern: &str) -> Option<u32> {
    let output = Command::new("pgrep").args(["-f", pattern]).output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().next()?.trim().parse().ok()
}

/// Checks whether `pid` is alive via `kill -0` (sends no signal, just probes).
fn pid_alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn kill_pid(pid: u32) {
    match Command::new("kill").arg(pid.to_string()).status() {
        Ok(status) if status.success() => {}
        // Most likely: the process already exited (e.g. it crashed) between
        // the last liveness check and here -- not a failure of the check
        // itself, so this is intentionally non-fatal.
        Ok(status) => {
            println!("note: 'kill {pid}' exited with {status}, process likely already gone")
        }
        Err(e) => println!("note: failed to run 'kill {pid}': {e}"),
    }
}

/// Lists `CoreTerm-*.ips` crash reports modified at or after `since`.
fn find_crash_reports_since(
    crash_dir: &std::path::Path,
    since: std::time::SystemTime,
) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(crash_dir) else {
        return Vec::new();
    };

    let mut found = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.starts_with("CoreTerm-") || !name.ends_with(".ips") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        if modified >= since {
            found.push(path);
        }
    }
    found.sort();
    found
}

// ============================================================================
// ISA test matrix
// ============================================================================
//
// `.cargo/config.toml` sets no `target-cpu`/`target-feature`, so a plain
// `cargo build`/`cargo test` always produces SSE2 code on x86-64 — even on a
// machine with AVX2 or AVX-512 available. Nothing in the default CI pipeline
// ever exercised the wider JIT backends. This walks every x86-64 ISA level via
// explicit `-C target-feature` flags — not `-C target-cpu=native`, which bakes
// in whatever the build machine happens to be and is not reproducible across
// machines or CI runners.
//
// Build and execute are separated deliberately, because they have different
// requirements. COMPILING for a target feature needs no matching hardware;
// only RUNNING the result does. So every level is built and linted
// unconditionally — that is what catches cfg mistakes, missing backend ops and
// lints, none of which need the CPU — and only the execution step is gated on
// `is_x86_feature_detected!`. A level the host cannot run therefore reports
// BUILT+LINT, not SKIP: an AVX-512-less runner still type-checks and lints the
// AVX-512 build, which is most of the value.
//
// Note the gate is per-LEVEL, not per-test: the whole workspace is compiled
// with the level's target-feature, so rustc may emit those instructions
// anywhere in the binary. It is not sufficient that an individual test avoids
// them.

/// One row of the ISA test matrix: a human-readable name, the
/// `-C target-feature` value to pass (empty = the SSE2 baseline, no flag),
/// and the `is_x86_feature_detected!` names that must all be present on this
/// host to attempt the level at all.
#[cfg(target_arch = "x86_64")]
struct IsaLevel {
    name: &'static str,
    target_feature: &'static str,
    requires: &'static [&'static str],
}

#[cfg(target_arch = "x86_64")]
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

/// Whether [`isa_matrix`] executes the tests it builds, or only builds and
/// lints. These are mutually exclusive execution modes, not an independent
/// toggle -- hence an enum rather than a `build_only: bool` alongside
/// `with_clippy: bool` (see the repository's "avoid boolean parameters"
/// convention in AGENTS.md). Not `x86_64`-gated like the rest of the matrix
/// machinery below: `main` constructs one from CLI args unconditionally,
/// before `isa_matrix` ever branches on host architecture.
enum IsaExecutionMode {
    /// Compile and lint every level; never execute tests, even on a host
    /// that could run them. Presubmit's fast path.
    BuildOnly,
    /// Compile, lint, and execute the tests for whichever levels this
    /// host's CPU can run. Postsubmit's job, once a change has landed.
    BuildAndTest,
}

/// Outcome of attempting one [`IsaLevel`].
#[cfg(target_arch = "x86_64")]
enum LevelResult {
    /// Compiled, linted, and the test binaries ran.
    Passed,
    /// Compiled and linted, but the tests were not executed — either
    /// [`IsaExecutionMode::BuildOnly`] was requested (presubmit's fast path)
    /// or the host CPU cannot run this level's instructions. NOT a skip —
    /// everything that can be checked without running the binary was
    /// checked.
    BuiltNotRun { reason: String },
    /// A command failed; `stage` names which for the summary.
    Failed { stage: &'static str },
}

/// Build and lint (`cargo test --workspace --no-run`, optionally `cargo
/// clippy --workspace --all-targets -- -D warnings`) every x86-64 ISA level.
/// Under [`IsaExecutionMode::BuildAndTest`], also runs the tests for
/// whichever levels this host's CPU can execute. `BuildOnly` is presubmit's
/// fast path: every PR gets compile+lint coverage for every level without
/// paying for a run; postsubmit calls this with `BuildAndTest` to actually
/// execute the tests once a change has landed. Non-x86-64 hosts
/// (aarch64/NEON) have a single ISA level already, so there is nothing to
/// matrix — this prints a note and exits 0 rather than silently doing
/// nothing.
fn isa_matrix(with_clippy: bool, mode: IsaExecutionMode) {
    let workspace_root = find_workspace_root();

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = workspace_root;
        let _ = with_clippy;
        let _ = mode;
        println!(
            "isa-matrix: host is not x86-64 (no SSE2/AVX2/AVX-512 split to test here — \
             e.g. aarch64/NEON has one ISA level already)."
        );
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

            let rustflags = if level.target_feature.is_empty() {
                FP_CONTRACT.to_string()
            } else {
                format!("{FP_CONTRACT} -C target-feature={}", level.target_feature)
            };
            println!("isa-matrix: RUSTFLAGS=\"{rustflags}\"");

            // Each level's RUSTFLAGS produce a disjoint artifact set (the flags
            // feed -C metadata), so levels sharing a target dir accumulate one
            // full workspace build EACH — three levels filled this container's
            // disk the first time it ran for real. One dedicated dir, wiped per
            // level: peak disk is a single artifact set, and the developer's
            // own target/ (their incremental cache) is never touched.
            let matrix_target = workspace_root.join("target/isa-matrix");
            if matrix_target.exists() {
                if let Err(e) = std::fs::remove_dir_all(&matrix_target) {
                    panic!(
                        "isa-matrix: could not clear {}: {e}",
                        matrix_target.display()
                    );
                }
            }

            // COMPILING a level needs no special hardware — only running the
            // result does. So build and lint every level unconditionally: that
            // is what catches cfg mistakes (a backend compiled out of a wide
            // build, a coverage sweep referring to a struct that no longer
            // exists there), missing ops, and lints, none of which need an
            // AVX-512 CPU to detect. Skipping a level outright, as this used to,
            // threw all of that away because the runner lacked one CPU flag.
            if !run_with_rustflags(
                &workspace_root,
                &matrix_target,
                &rustflags,
                &["test", "--workspace", "--no-run"],
            ) {
                println!("isa-matrix: {} — test build FAILED", level.name);
                results.push((level.name, LevelResult::Failed { stage: "build" }));
                continue;
            }
            println!("isa-matrix: {} — test build ok", level.name);

            if with_clippy {
                let clippy_ok = run_with_rustflags(
                    &workspace_root,
                    &matrix_target,
                    &rustflags,
                    &[
                        "clippy",
                        "--workspace",
                        "--all-targets",
                        "--",
                        "-D",
                        "warnings",
                    ],
                );
                if !clippy_ok {
                    println!("isa-matrix: {} — cargo clippy FAILED", level.name);
                    results.push((level.name, LevelResult::Failed { stage: "clippy" }));
                    continue;
                }
                println!("isa-matrix: {} — cargo clippy passed", level.name);
            }

            // Executing is the part that genuinely needs the CPU (and, under
            // `BuildOnly`, the part presubmit explicitly defers). Note the
            // whole workspace is built with this level's target-feature, so
            // rustc may emit those instructions anywhere in the binary — it is
            // not enough that a given test avoids them.
            let skip_reason = match mode {
                IsaExecutionMode::BuildOnly => {
                    Some("build-only mode: tests run in postsubmit".to_string())
                }
                IsaExecutionMode::BuildAndTest => level
                    .requires
                    .iter()
                    .find(|&&feat| !host_has_feature(feat))
                    .map(|&feat| format!("host lacks {feat}")),
            };

            if let Some(reason) = skip_reason {
                println!(
                    "isa-matrix: {} — built and linted; NOT running tests ({reason})",
                    level.name
                );
                results.push((level.name, LevelResult::BuiltNotRun { reason }));
                continue;
            }

            if !run_with_rustflags(
                &workspace_root,
                &matrix_target,
                &rustflags,
                &["test", "--workspace", "--no-fail-fast"],
            ) {
                println!("isa-matrix: {} — cargo test FAILED", level.name);
                results.push((level.name, LevelResult::Failed { stage: "test" }));
                continue;
            }
            println!("isa-matrix: {} — cargo test passed", level.name);

            results.push((level.name, LevelResult::Passed));
        }

        println!("\n=== ISA matrix summary ===");
        let mut any_failed = false;
        for (name, result) in &results {
            let line = match result {
                LevelResult::Passed => "PASS".to_string(),
                LevelResult::Failed { stage } => {
                    any_failed = true;
                    format!("FAIL ({stage})")
                }
                LevelResult::BuiltNotRun { reason } => {
                    format!("BUILT+LINT (not run: {reason})")
                }
            };
            println!("  {name:<20} {line}");
        }

        if any_failed {
            std::process::exit(1);
        }
    }
}

/// The triple this machine builds for, from `rustc -vV`.
///
/// Needed because `--target` must be passed *explicitly* to keep `RUSTFLAGS` off host
/// artifacts — see [`run_with_rustflags`]. Naming the host triple is a no-op for what gets
/// built; it is the act of specifying `--target` at all that cargo keys the behaviour on.
#[cfg(target_arch = "x86_64")]
fn host_triple() -> String {
    let out = Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("isa-matrix: could not run `rustc -vV` to discover the host triple");
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .expect("isa-matrix: `rustc -vV` printed no `host:` line")
        .trim()
        .to_string()
}

/// Run `cargo <args>` from the workspace root with `RUSTFLAGS` set to
/// `rustflags` and artifacts confined to `target_dir`, streaming output
/// straight through. Returns whether it succeeded.
///
/// # Why `--target` is always passed
///
/// Without it, `RUSTFLAGS` applies to **host** artifacts too — build scripts and proc macros —
/// and those are compiled *and then executed* by cargo on the machine doing the build. A level
/// that names an ISA the build machine lacks therefore kills the build script with `SIGILL`
/// before any of this workspace is even compiled. `proc-macro2`'s build script died exactly
/// that way on a CI runner without AVX-512.
///
/// Passing `--target` explicitly is what makes cargo apply `RUSTFLAGS` to target artifacts only.
/// The triple is the host's own, so nothing about the build changes except that host tooling
/// stops being compiled for a CPU that is not going to run it. This is what makes the matrix's
/// "compiling a level needs no special hardware" claim actually true: it is true of *this*
/// workspace's code, and was quietly false for everything cargo runs on the way there.
#[cfg(target_arch = "x86_64")]
fn run_with_rustflags(
    workspace_root: &std::path::Path,
    target_dir: &std::path::Path,
    rustflags: &str,
    args: &[&str],
) -> bool {
    // Before any `--`, or it would be forwarded to the tool behind the separator (clippy's lint
    // arguments) instead of being read by cargo.
    let split = args.iter().position(|a| *a == "--").unwrap_or(args.len());
    Command::new("cargo")
        .current_dir(workspace_root)
        .args(&args[..split])
        .arg("--target")
        .arg(host_triple())
        .args(&args[split..])
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
