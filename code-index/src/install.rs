use crate::config::Config;
use std::path::PathBuf;
use std::process::Command;

pub fn install(config: &Config) -> Result<(), String> {
    // 1. Ensure model is downloaded.
    crate::download::ensure_model(&config.model_dir())?;

    // 2. Write config if it doesn't exist.
    config.save()?;

    // 3. Copy binary to ~/.local/bin so hook commands and PATH users find it.
    let exe = std::env::current_exe()
        .map_err(|e| format!("cannot determine binary path: {e}"))?;
    install_binary(&exe)?;

    #[cfg(target_os = "macos")]
    setup_launchd(&exe, config)?;

    #[cfg(target_os = "linux")]
    setup_systemd(&exe, config)?;

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    eprintln!(
        "[code-index] Daemon setup not supported on this platform. \
         Run `code-index reindex` manually."
    );

    // 4. Run first reindex.
    eprintln!("[code-index] Running initial reindex...");
    crate::cmd_reindex(config)?;

    eprintln!("\n[code-index] Install complete.");
    eprintln!("  Config: ~/.code-index/config.toml");
    eprintln!("  Add workspaces to config, then run: code-index reindex");
    eprintln!(
        "  Hook: add 'code-index query' to UserPromptSubmit in .claude/settings.json"
    );

    Ok(())
}

fn install_binary(exe: &std::path::Path) -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let bin_dir = PathBuf::from(&home).join(".local/bin");
    std::fs::create_dir_all(&bin_dir)
        .map_err(|e| format!("failed to create {}: {e}", bin_dir.display()))?;
    let dest = bin_dir.join("code-index");
    std::fs::copy(exe, &dest)
        .map_err(|e| format!("failed to copy binary to {}: {e}", dest.display()))?;
    eprintln!("[code-index] Binary installed to {}", dest.display());
    eprintln!(
        "[code-index] Make sure {} is on your PATH.",
        bin_dir.display()
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn setup_launchd(exe: &std::path::Path, config: &Config) -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let plist_dir = PathBuf::from(&home).join("Library/LaunchAgents");
    std::fs::create_dir_all(&plist_dir)
        .map_err(|e| format!("failed to create LaunchAgents dir: {e}"))?;
    let plist_path = plist_dir.join("io.code-index.plist");
    let interval = config.index.reindex_interval_minutes * 60;
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>io.code-index</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>reindex</string>
    </array>
    <key>StartInterval</key><integer>{interval}</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardErrorPath</key>
    <string>{home}/.code-index/daemon.log</string>
</dict>
</plist>"#,
        exe = exe.display(),
        interval = interval,
        home = home,
    );
    std::fs::write(&plist_path, plist)
        .map_err(|e| format!("failed to write plist: {e}"))?;
    let status = Command::new("launchctl")
        .args(["load", plist_path.to_str().unwrap()])
        .status()
        .map_err(|e| format!("launchctl load failed: {e}"))?;
    if !status.success() {
        return Err(format!("launchctl load exited {status}"));
    }
    eprintln!("[code-index] launchd agent installed: io.code-index");
    Ok(())
}

#[cfg(target_os = "linux")]
fn setup_systemd(exe: &std::path::Path, config: &Config) -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let unit_dir = PathBuf::from(&home).join(".config/systemd/user");
    std::fs::create_dir_all(&unit_dir)
        .map_err(|e| format!("failed to create systemd user dir: {e}"))?;

    let service = format!(
        "[Unit]\nDescription=code-index reindexer\n\n\
         [Service]\nType=oneshot\nExecStart={exe} reindex\n\
         StandardError=append:{home}/.code-index/daemon.log\n",
        exe = exe.display(),
        home = home,
    );
    std::fs::write(unit_dir.join("code-index.service"), service)
        .map_err(|e| format!("failed to write service: {e}"))?;

    let interval_sec = config.index.reindex_interval_minutes * 60;
    let timer = format!(
        "[Unit]\nDescription=code-index reindex timer\n\n\
         [Timer]\nOnBootSec=60\nOnUnitActiveSec={interval_sec}\n\n\
         [Install]\nWantedBy=timers.target\n",
    );
    std::fs::write(unit_dir.join("code-index.timer"), timer)
        .map_err(|e| format!("failed to write timer: {e}"))?;

    for cmd in [
        vec!["daemon-reload"],
        vec!["enable", "--now", "code-index.timer"],
    ] {
        let status = Command::new("systemctl")
            .arg("--user")
            .args(&cmd)
            .status()
            .map_err(|e| format!("systemctl failed: {e}"))?;
        if !status.success() {
            return Err(format!(
                "systemctl --user {} exited {status}",
                cmd.join(" ")
            ));
        }
    }
    eprintln!("[code-index] systemd user timer installed: code-index.timer");
    Ok(())
}
