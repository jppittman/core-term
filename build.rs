// build.rs

/// This build script automatically detects the target operating system and sets
/// the appropriate configuration flags (`cfg`) for conditional compilation.
/// It uses the `pkg-config` utility to find system libraries in a robust
/// and portable way, which is the standard practice for Rust projects
/// with C dependencies.
fn main() {
    // This line tells Cargo to re-run this script only if the script itself changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icons/icon.icns");

    // Cargo sets this environment variable during the build process.
    // We read it to determine which OS we are compiling for.
    let target_os = match std::env::var("CARGO_CFG_TARGET_OS") {
        Ok(os) => os,
        Err(_) => {
            // If the variable isn't set, we can't proceed.
            // This is highly unlikely in a normal Cargo build.
            panic!("CARGO_CFG_TARGET_OS is not set, cannot determine target platform.");
        }
    };

    // Based on the operating system, we emit the appropriate `cfg` flag.
    // The `main.rs` and other modules will use these flags to include the
    // correct platform-specific backend.
    match target_os.as_str() {
        "linux" => {
            // On Linux, we enable the X11 backend.
            println!("cargo:rustc-cfg=use_x11_backend");

            // Define the list of required libraries for the X11 backend.
            let required_libs = ["x11", "xext", "xft", "fontconfig", "freetype2"];

            // Use a loop to probe for each library using pkg-config.
            // This is cleaner and easier to maintain than repeating the call.
            for lib in required_libs {
                if let Err(e) = pkg_config::probe_library(lib) {
                    // Panicking here is the correct behavior, as these are
                    // required dependencies for the build to succeed on Linux.
                    panic!(
                        "Failed to find required library `{}` using pkg-config: {}",
                        lib, e
                    );
                }
            }
        }
        "macos" => {
            // On macOS, we enable the macOS-specific backend.
            // No special linker flags are needed here by default, as the
            // system libraries are found automatically via frameworks.
            println!("cargo:rustc-cfg=use_macos_backend");

            // Create app bundle structure for keyboard input support
            create_macos_app_bundle();
        }
        _ => {
            // For any other operating system, we don't enable a specific backend.
        }
    }
}

#[cfg(target_os = "macos")]
fn create_macos_app_bundle() {
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bundle_dir = Path::new(&manifest_dir).join("CoreTerm.app/Contents");

    // Create bundle structure
    let _ = fs::remove_dir_all(Path::new(&manifest_dir).join("CoreTerm.app"));
    fs::create_dir_all(bundle_dir.join("MacOS")).expect("Failed to create MacOS directory");
    fs::create_dir_all(bundle_dir.join("Resources")).expect("Failed to create Resources directory");

    // Create Info.plist (required for keyboard input to work on macOS)
    let plist_content = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>CoreTerm</string>
    <key>CFBundleIdentifier</key>
    <string>com.core-term.terminal</string>
    <key>CFBundleName</key>
    <string>CoreTerm</string>
    <key>CFBundleDisplayName</key>
    <string>CoreTerm</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
"#;

    let plist_path = bundle_dir.join("Info.plist");
    let mut plist_file = fs::File::create(&plist_path).expect("Failed to create Info.plist");
    plist_file
        .write_all(plist_content.as_bytes())
        .expect("Failed to write Info.plist");
}

#[cfg(not(target_os = "macos"))]
fn create_macos_app_bundle() {
    // No-op on non-macOS platforms
}
