pub mod claude;
pub mod gemini;

use crate::tools::ScoredChunk;

/// Formats scored code chunks for injection into a specific LLM's hook system.
pub trait HookBackend {
    /// Consume the top-k scored results and produce the output written to stdout.
    fn format_results(&self, results: &[ScoredChunk], top_k: usize) -> String;
}

/// Resolve a backend name to a boxed `HookBackend`.
/// Returns an `Err` with the list of known names on unrecognised input.
pub fn from_name(name: &str) -> Result<Box<dyn HookBackend>, String> {
    match name {
        "claude" => Ok(Box::new(claude::ClaudeBackend)),
        "gemini" => Ok(Box::new(gemini::GeminiBackend)),
        other => Err(format!(
            "unknown backend '{other}' — supported: claude, gemini"
        )),
    }
}
