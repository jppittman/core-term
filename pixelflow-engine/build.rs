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
            let required_libs = ["x11", "fontconfig", "freetype2"];
            for lib in required_libs {
                if let Err(e) = pkg_config::probe_library(lib) {
                    eprintln!("Warning: Failed to find library `{}`: {}", lib, e);
                }
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
    let has_x11 =
        cfg!(feature = "display_x11") || std::env::var("CARGO_FEATURE_DISPLAY_X11").is_ok();
    let has_headless = cfg!(feature = "display_headless")
        || std::env::var("CARGO_FEATURE_DISPLAY_HEADLESS").is_ok();
    let has_web =
        cfg!(feature = "display_web") || std::env::var("CARGO_FEATURE_DISPLAY_WEB").is_ok();

    if has_web {
        return "web".to_string();
    }
    if has_headless && !has_x11 && !has_cocoa {
        return "headless".to_string();
    }
    if has_x11 && !has_headless {
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
