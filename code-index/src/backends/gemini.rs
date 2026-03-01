use super::HookBackend;
use crate::tools::ScoredChunk;
use serde_json::json;

pub struct GeminiBackend;

impl HookBackend for GeminiBackend {
    fn format_results(&self, results: &[ScoredChunk], top_k: usize) -> String {
        if results.is_empty() {
            let output = json!({
                "hookSpecificOutput": {
                    "hookEventName": "BeforeAgent",
                    "additionalContext": "Workspace index is empty. Run `code-index reindex` first."
                }
            });
            return serde_json::to_string_pretty(&output)
                .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}\n"));
        }

        let mut context = String::from("Relevant Workspace Context (RAG):\n\n");
        for sc in results.iter().take(top_k) {
            context.push_str(&format!(
                "--- {} `{}` in {} (Relevance: {:.2}) ---\n{}\n\n",
                sc.chunk.kind, sc.chunk.name, sc.chunk.source_path, sc.score, sc.chunk.source
            ));
        }

        let output = json!({
            "hookSpecificOutput": {
                "hookEventName": "BeforeAgent",
                "additionalContext": context
            }
        });

        serde_json::to_string_pretty(&output)
            .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}\n"))
    }
}
