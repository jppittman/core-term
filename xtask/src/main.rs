//! Automation tasks for the project.
//!
//! Currently supports bundling the macOS application.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Entry point for xtask.
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: cargo xtask <command>");
        eprintln!("Commands:");
        eprintln!("  bundle-run    Build and run the bundled macOS app");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "bundle-run" => {
            // Pass through any additional arguments after "bundle-run"
            let extra_args = if args.len() > 2 { &args[2..] } else { &[] };
            bundle_run(extra_args);
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

    println!("Building core-term in release mode...");

    // Build the project with extra args (e.g., --features profiling)
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&workspace_root); // Run from workspace root
    cmd.args(["build", "--release"]);

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
    let icon_dest = workspace_root.join("CoreTerm.app/Contents/Resources/icon.icns");

    if icon_src.exists() {
        println!("Copying icon to bundle...");
        fs::create_dir_all(icon_dest.parent().unwrap())
            .expect("Failed to create Resources directory");
        fs::copy(&icon_src, &icon_dest).expect("Failed to copy icon to bundle");
    } else {
        println!("Warning: Icon not found at {}", icon_src.display());
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
