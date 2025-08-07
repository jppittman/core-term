// build.rs

/// This build script automatically detects the target operating system and sets
/// the appropriate configuration flags (`cfg`) for conditional compilation.
/// It uses the `pkg-config` utility to find system libraries in a robust
/// and portable way, which is the standard practice for Rust projects
/// with C dependencies.
fn main() {
    // This line tells Cargo to re-run this script only if the script itself changes.
    println!("cargo:rerun-if-changed=build.rs");

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
        }
        _ => {
            // For any other operating system, we don't enable a specific backend.
        }
    }
}
