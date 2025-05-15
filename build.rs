// build.rs

fn main() {
    // --- Link against X11 and its dependencies ---
    // We'll try to use pkg-config first, which is the standard way to find
    // library linking information on Unix-like systems.
    // If pkg-config fails (e.g., not installed, or the .pc file is missing/incorrect),
    // we'll fall back to manually specifying common linker flags.

    let libraries = ["x11", "xft", "fontconfig", "freetype2"]; // Libraries needed for X backend drawing

    let mut pkg_config_success = true;

    for lib in &libraries {
        // Try to use pkg-config for each library
        let result = pkg_config::probe_library(lib);

        if result.is_err() {
            // If probing fails for any library, we'll assume pkg-config isn't fully working
            // or the library isn't found via pkg-config. We'll then use manual linking.
            eprintln!(
                "pkg-config failed for library '{}'. Falling back to manual linking.",
                lib
            );
            pkg_config_success = false;
            // No need to print error from probe_library, it already prints
            break; // Stop probing and switch to manual mode
        }
    }

    if !pkg_config_success {
        // --- Manual Linking Fallback ---
        // If pkg-config failed, manually tell Cargo how to link.
        // This assumes libraries are in standard paths like /usr/lib or /usr/local/lib.
        // If your libraries are in non-standard locations, you might need to adjust
        // the -L path or rely on environment variables like LIBRARY_PATH for this fallback.

        println!("cargo:rustc-link-lib=X11");
        println!("cargo:rustc-link-lib=Xext"); // Often needed with X11
        println!("cargo:rustc-link-lib=Xft"); // For font rendering
        println!("cargo:rustc-link-lib=fontconfig"); // For font configuration
        println!("cargo:rustc-link-lib=freetype"); // For font rendering

        // Add a common library search path. /usr/lib is standard on many systems.
        // On some systems, /usr/lib64 or /usr/local/lib might also be needed.
        // You can add multiple search paths if necessary.
        println!("cargo:rustc-link-search=/usr/lib");
        // println!("cargo:rustc-link-search=/usr/lib64");
        // println!("cargo:rustc-link-search=/usr/local/lib");

        // For include paths, the 'x11' crate might still need help if pkg-config --cflags is empty.
        // While build.rs can't directly set CFLAGS for the main compilation,
        // the 'x11' crate's build script might respect environment variables
        // set *before* cargo runs. However, the manual linking above is often
        // sufficient if the headers are in a standard include path that cc already checks.
        // If you still get compilation errors related to missing headers (e.g., Xlib.h not found),
        // you might need to set CFLAGS="-I/usr/include/X11" when running cargo build.
        eprintln!(
            "Manual linking flags applied. Ensure X11, Xft, Fontconfig, and Freetype development libraries are installed."
        );
    } else {
        // If pkg-config succeeded for all, it has already printed the necessary flags.
        eprintln!("pkg-config successfully found libraries. Linking configured automatically.");
    }

    // --- Other build tasks (if any) ---
    // You can add other build-time logic here, like:
    // - Generating code from protocol definitions (e.g., Wayland)
    // - Compiling C helper files
    // - Embedding assets
}
