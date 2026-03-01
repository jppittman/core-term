use std::fs;
use std::io;
use std::path::Path;

const BASE: &str =
    "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main";
const FILES: &[&str] = &["model.onnx", "model.onnx_data", "tokenizer.json"];

/// Ensure all model files are present in model_dir.
/// Downloads from HuggingFace via HTTPS if any are missing.
/// Idempotent — no-op if all files already exist.
pub fn ensure_model(model_dir: &Path) -> Result<(), String> {
    if FILES.iter().all(|f| model_dir.join(f).exists()) {
        return Ok(());
    }
    eprintln!(
        "[code-index] Downloading EmbeddingGemma to {} ...",
        model_dir.display()
    );
    fs::create_dir_all(model_dir)
        .map_err(|e| format!("failed to create model dir: {e}"))?;
    for file in FILES {
        let dest_path = model_dir.join(file);
        if dest_path.exists() {
            // Partial download recovery: skip files already present.
            continue;
        }
        eprintln!("[code-index]   {file}");
        let url = format!("{BASE}/{file}");
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| format!("failed to download {file}: {e}"))?;
        let mut dest = fs::File::create(&dest_path)
            .map_err(|e| format!("failed to create {}: {e}", dest_path.display()))?;
        let mut reader = resp.into_body().into_reader();
        io::copy(&mut reader, &mut dest)
            .map_err(|e| format!("failed to write {file}: {e}"))?;
    }
    eprintln!("[code-index] Download complete.");
    Ok(())
}
