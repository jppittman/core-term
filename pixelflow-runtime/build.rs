fn has_x11_feature() -> bool {
    cfg!(feature = "display_x11") || std::env::var("CARGO_FEATURE_DISPLAY_X11").is_ok()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DISPLAY_DRIVER");

    // Declare custom cfg names to avoid warnings
    println!("cargo::rustc-check-cfg=cfg(use_cocoa_display)");
    println!("cargo::rustc-check-cfg=cfg(use_x11_display)");
    println!("cargo::rustc-check-cfg=cfg(use_headless_display)");
    println!("cargo::rustc-check-cfg=cfg(use_web_display)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS")
        .expect("CARGO_CFG_TARGET_OS is not set, cannot determine target platform.");

    // Determine which display driver to use
    let display_driver = determine_display_driver(&target_os);

    // Emit appropriate cfg flag for the selected display driver
    match display_driver.as_str() {
        "cocoa" => {
            println!("cargo:rustc-cfg=use_cocoa_display");
        }
        "x11" => {
            println!("cargo:rustc-cfg=use_x11_display");
            // Probe for X11 libraries using pkg-config
            // Only fail if X11 was explicitly requested via feature
            if has_x11_feature() {
                pkg_config::probe_library("x11")
                    .expect("X11 library not found. On Linux, install libx11-dev (Debian/Ubuntu) or libx11-devel (RHEL/Fedora)");
            } else if pkg_config::probe_library("x11").is_err() {
                eprintln!("Warning: X11 libraries not found. X11 features may not be available.");
            }
        }
        "headless" => {
            println!("cargo:rustc-cfg=use_headless_display");
        }
        "web" => {
            println!("cargo:rustc-cfg=use_web_display");
        }
        _ => {
            panic!("Unknown display driver: {}", display_driver);
        }
    }
}

fn determine_display_driver(target_os: &str) -> String {
    if let Ok(driver) = std::env::var("DISPLAY_DRIVER") {
        return driver.to_lowercase();
    }

    // Check features
    let has_cocoa =
        cfg!(feature = "display_cocoa") || std::env::var("CARGO_FEATURE_DISPLAY_COCOA").is_ok();
    let has_x11 = has_x11_feature();
    let has_headless = cfg!(feature = "display_headless")
        || std::env::var("CARGO_FEATURE_DISPLAY_HEADLESS").is_ok();
    let has_web =
        cfg!(feature = "display_web") || std::env::var("CARGO_FEATURE_DISPLAY_WEB").is_ok();

    // Check target architecture to allow disabling web driver on non-wasm targets
    // even if the feature is enabled (e.g. via --all-features)
    let is_wasm = std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|a| a == "wasm32");

    if has_web && is_wasm {
        return "web".to_string();
    }

    // Prioritize OS-specific drivers when multiple features are enabled
    if target_os == "macos" {
        if has_cocoa {
            return "cocoa".to_string();
        }
        if has_headless {
            return "headless".to_string();
        }
        return "cocoa".to_string();
    }

    if target_os == "linux" {
        if has_x11 {
            return "x11".to_string();
        }
        if has_headless {
            return "headless".to_string();
        }

        // On Linux, prefer X11 but fall back to headless if X11 is not available
        if pkg_config::probe_library("x11").is_ok() {
            return "x11".to_string();
        } else {
            eprintln!("Warning: X11 libraries not found. Falling back to headless display driver.");
            println!("cargo:warning=X11 libraries not found. Building headless driver instead.");
            println!("cargo:warning=To use X11, install libx11-dev (Debian/Ubuntu) or libx11-devel (RHEL/Fedora)");
            return "headless".to_string();
        }
    }

    // Fallbacks
    if has_headless {
        return "headless".to_string();
    }
    if has_x11 {
        return "x11".to_string();
    }
    if has_cocoa {
        return "cocoa".to_string();
    }

    match target_os {
        "macos" => "cocoa".to_string(),
        "linux" => "x11".to_string(),
        "unknown" => "web".to_string(),
        _ => "headless".to_string(),
    }
}
