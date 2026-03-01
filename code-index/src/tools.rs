use crate::backends::HookBackend;
use crate::config::Config;
use crate::vdb::{self, Hnsw};
use crate::{embed, mcp, parse, store};
use serde_json::Value;

// Temporary type used during reindex: chunk metadata + vector before they're split apart.
struct EmbeddedChunk {
    id: String,
    kind: String,
    name: String,
    source_path: String,
    source: String,
    vector: Vec<f32>,
}

// ─── MCP dispatch ─────────────────────────────────────────────────────────────

pub fn handle(
    embedder: &embed::Embedder,
    config: &Config,
    req: &serde_json::Value,
) -> Option<serde_json::Value> {
    let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");

    match method {
        "initialize" => Some(mcp::ok(
            id,
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "code-index", "version": "0.1.0"}
            }),
        )),

        // Notifications have no id and expect no response.
        "notifications/initialized" => None,

        "tools/list" => Some(mcp::ok(
            id,
            serde_json::json!({"tools": tool_schemas()}),
        )),

        "tools/call" => {
            let params = req
                .get("params")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let tool_name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("(missing)");
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::Object(Default::default()));

            let result_text = match tool_name {
                "index_workspace" => index_workspace(embedder, config, &args),
                "search_code" => search_code(embedder, config, &args),
                "note_to_self" => note_to_self(embedder, &args),
                "check_notes" => check_notes(embedder, config, &args),
                other => Err(format!("unknown tool: {other}")),
            };

            let result_value = match result_text {
                Ok(text) => mcp::tool_result(text),
                Err(e) => {
                    eprintln!("tool error [{tool_name}]: {e}");
                    mcp::tool_error_result(e)
                }
            };

            Some(mcp::ok(id, result_value))
        }

        other => {
            eprintln!("unhandled method: {other}");
            // For unknown methods with an id, return a JSON-RPC error.
            if req.get("id").is_some() {
                Some(mcp::error(
                    id,
                    -32601,
                    format!("method not found: {other}"),
                ))
            } else {
                None
            }
        }
    }
}

fn tool_schemas() -> serde_json::Value {
    serde_json::json!([
        {
            "name": "index_workspace",
            "description": "Walk the workspace, parse source files with tree-sitter, embed each chunk with EmbeddingGemma (in-process ONNX), and write to .code-index/chunks.jsonl. Must be run before search_code.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional explicit workspace root path. Defaults to the first workspace in config or the workspace containing the current directory."
                    }
                }
            }
        },
        {
            "name": "search_code",
            "description": "Semantic search over indexed code chunks. Returns the top-k most similar chunks by cosine similarity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or code query to search for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default from config)."
                    },
                    "crate_filter": {
                        "type": "string",
                        "description": "Optional prefix filter on source_path (e.g. 'pixelflow-core/')."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "note_to_self",
            "description": "Store a persistent note in .code-index/notes.jsonl, embedded for later semantic retrieval.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note content to store."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of string tags for the note."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "check_notes",
            "description": "Semantic search over stored notes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search notes."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default from config)."
                    }
                },
                "required": ["query"]
            }
        }
    ])
}

// ─── Public reindex/query entry points for CLI subcommands ────────────────────

/// Walk a single workspace, embed chunks, build HNSW, and persist both.
/// Used by both the `reindex` CLI subcommand and the `index_workspace` MCP tool.
pub fn reindex_workspace(
    embedder: &embed::Embedder,
    config: &Config,
    workspace_root: &std::path::Path,
) -> Result<(), String> {
    let index_dir = store::index_dir(workspace_root)?;

    let mut all_chunks: Vec<EmbeddedChunk> = Vec::new();
    let mut file_count: usize = 0;

    let skip_dirs = ["target", ".git", ".code-index"];

    // Build the set of supported extensions from config.
    let extensions: Vec<&str> = config.index.extensions.iter().map(|s| s.as_str()).collect();

    let mut stack: Vec<std::path::PathBuf> = vec![workspace_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .map_err(|e| format!("read_dir failed for {}: {e}", dir.display()))?;
        for entry in entries {
            let entry =
                entry.map_err(|e| format!("dir entry error in {}: {e}", dir.display()))?;
            let path = entry.path();
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();

            if path.is_dir() {
                if file_name.starts_with('.') || skip_dirs.contains(&file_name) {
                    continue;
                }
                stack.push(path);
                continue;
            }

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default();
            if ext == "json" || ext == "toml" {
                continue;
            }
            if !extensions.contains(&ext) {
                continue;
            }

            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("skipping {}: {e}", path.display());
                    continue;
                }
            };

            let raw_chunks = parse::parse_file(&path, &source);
            if raw_chunks.is_empty() {
                continue;
            }

            let rel_path = path
                .strip_prefix(workspace_root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            eprintln!("indexing {}: {} chunks", rel_path, raw_chunks.len());
            file_count += 1;

            for raw in raw_chunks {
                let vector = embedder.embed(&raw.embed_text).map_err(|e| {
                    format!("embedding failed for `{}` in {rel_path}: {e}", raw.name)
                })?;
                let id = store::chunk_id(&rel_path, &raw.name, &raw.kind);
                all_chunks.push(EmbeddedChunk {
                    id,
                    kind: raw.kind,
                    name: raw.name,
                    source_path: rel_path.clone(),
                    source: raw.source,
                    vector,
                });
            }
        }
    }

    let chunk_count = all_chunks.len();

    // Build HNSW from embedded vectors, then strip vectors before storing metadata.
    let mut hnsw = Hnsw::new(vdb::dot);
    let meta_chunks: Vec<store::Chunk> = all_chunks
        .into_iter()
        .map(|c| {
            hnsw.insert(&c.id, c.vector);
            store::Chunk {
                id: c.id,
                kind: c.kind,
                name: c.name,
                source_path: c.source_path,
                source: c.source,
            }
        })
        .collect();

    let mut db = store::Store::open(&index_dir)?;
    db.replace_chunks(&meta_chunks)?;
    db.save_hnsw(&hnsw)?;

    eprintln!(
        "[reindex] {}: {file_count} files, {chunk_count} chunks, {} HNSW nodes -> {}/.code-index/index.db",
        workspace_root.display(),
        hnsw.len(),
        workspace_root.display(),
    );
    Ok(())
}

/// A search result with both similarity score and full chunk metadata.
pub struct ScoredChunk {
    pub score: f32,
    pub chunk: store::Chunk,
}

/// Embed a query, run HNSW search across all configured workspaces, fetch
/// metadata for hits, and drop results below `config.search.min_score`.
/// Returns results sorted descending by score.
pub fn search_workspaces(
    embedder: &embed::Embedder,
    config: &Config,
    prompt: &str,
) -> Result<Vec<ScoredChunk>, String> {
    let query_vec = embedder.embed(prompt)?;
    let top_k = config.search.top_k;
    let min_score = config.search.min_score;
    // ef > top_k gives the ANN search more candidates to improve recall.
    let ef = (top_k * 4).max(64);

    let workspaces = config.expanded_workspaces();
    let mut all: Vec<ScoredChunk> = Vec::new();

    for ws in &workspaces {
        let index_dir = store::index_dir(ws)?;
        let db = store::Store::open(&index_dir)?;
        let hnsw = db.load_hnsw(vdb::dot)?;

        if hnsw.is_empty() {
            continue;
        }

        let hits = hnsw.search(&query_vec, top_k, ef);
        let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        let chunks = db.get_chunks_by_ids(&ids)?;

        // Build a lookup so we preserve HNSW's score order.
        let mut by_id: std::collections::HashMap<String, store::Chunk> =
            chunks.into_iter().map(|c| (c.id.clone(), c)).collect();

        for hit in hits {
            if hit.score < min_score {
                break; // hits are sorted descending; everything after is also below threshold
            }
            if let Some(chunk) = by_id.remove(&hit.id) {
                all.push(ScoredChunk { score: hit.score, chunk });
            }
        }
    }

    // Sort across workspaces (each workspace's hits are already sorted,
    // but we merge multiple workspaces here).
    all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(all)
}

/// Search all configured workspaces and format the results via the given backend.
/// Used by CLI subcommands for LLM hook injection.
pub fn query_all_workspaces(
    embedder: &embed::Embedder,
    config: &Config,
    prompt: &str,
    backend: &dyn HookBackend,
) -> Result<String, String> {
    let results = search_workspaces(embedder, config, prompt)?;
    Ok(backend.format_results(&results, config.search.top_k))
}

// ─── MCP tool implementations ─────────────────────────────────────────────────

/// Walk workspace, parse supported files, embed each chunk, write chunks.jsonl.
fn index_workspace(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> Result<String, String> {
    let workspace_root = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            // Use first configured workspace, or fall back to workspace root detection.
            match config.expanded_workspaces().into_iter().next() {
                Some(ws) => ws,
                None => find_workspace_root()?,
            }
        }
    };

    reindex_workspace(embedder, config, &workspace_root)?;
    Ok(format!(
        "Indexing complete. Written to {}/.code-index/index.db",
        workspace_root.display()
    ))
}

/// Embed query, search HNSW, return top_k results with optional path filter.
fn search_code(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "search_code requires a 'query' string argument".to_string())?;

    let top_k = args
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(config.search.top_k);

    let crate_filter = args
        .get("crate_filter")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let workspace_root = match config.expanded_workspaces().into_iter().next() {
        Some(ws) => ws,
        None => find_workspace_root()?,
    };
    let index_dir = store::index_dir(&workspace_root)?;
    let db = store::Store::open(&index_dir)?;
    let hnsw = db.load_hnsw(vdb::dot)?;

    if hnsw.is_empty() {
        return Ok("No chunks indexed yet. Run `index_workspace` first.".to_string());
    }

    let query_vec = embedder.embed(query)?;
    // Request more candidates when filtering so we have enough after the filter.
    let ef = (top_k * 8).max(64);
    let hits = hnsw.search(&query_vec, ef, ef);

    let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    let chunks = db.get_chunks_by_ids(&ids)?;
    let by_id: std::collections::HashMap<String, store::Chunk> =
        chunks.into_iter().map(|c| (c.id.clone(), c)).collect();

    let mut out = String::new();
    let mut count = 0;
    for hit in &hits {
        if count >= top_k {
            break;
        }
        let chunk = match by_id.get(&hit.id) {
            Some(c) => c,
            None => continue,
        };
        if crate_filter
            .as_deref()
            .map(|f| !chunk.source_path.starts_with(f))
            .unwrap_or(false)
        {
            continue;
        }
        let preview = if chunk.source.len() > 300 {
            &chunk.source[..300]
        } else {
            &chunk.source
        };
        out.push_str(&format!(
            "[{:.2}] {} `{}` — {}\n{}\n\n",
            hit.score, chunk.kind, chunk.name, chunk.source_path, preview
        ));
        count += 1;
    }

    if out.is_empty() {
        Ok("No matching chunks found.".to_string())
    } else {
        Ok(out.trim_end().to_string())
    }
}

/// Embed and append a note to notes.jsonl.
fn note_to_self(embedder: &embed::Embedder, args: &Value) -> Result<String, String> {
    let text = args
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "note_to_self requires a 'text' string argument".to_string())?;

    let tags: Vec<String> = args
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let vector = embedder.embed(text)?;

    let workspace_root = find_workspace_root()?;
    let index_dir = store::index_dir(&workspace_root)?;
    let db = store::Store::open(&index_dir)?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string());
    let id = store::chunk_id(text, &timestamp, "note");

    let note = store::Note {
        id: id.clone(),
        text: text.to_string(),
        tags,
        timestamp,
        vector,
    };

    db.upsert_note(&note)?;

    Ok(format!("Note saved (id: {id})"))
}

/// Embed query, score notes by cosine similarity, return top_k results.
fn check_notes(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "check_notes requires a 'query' string argument".to_string())?;

    let top_k = args
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(config.search.top_k);

    let workspace_root = find_workspace_root()?;
    let index_dir = store::index_dir(&workspace_root)?;
    let db = store::Store::open(&index_dir)?;
    let notes = db.load_notes()?;
    if notes.is_empty() {
        return Ok("No notes yet. Use `note_to_self` to add one.".to_string());
    }

    let query_vec = embedder.embed(query)?;

    let mut scored: Vec<(f32, &store::Note)> = notes
        .iter()
        .map(|n| (vdb::dot(&query_vec, &n.vector), n))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = String::new();
    for (score, note) in scored.iter().take(top_k) {
        let tag_str = if note.tags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", note.tags.join(", "))
        };
        out.push_str(&format!(
            "[{:.2}]{} (ts: {}) {}\n\n",
            score, tag_str, note.timestamp, note.text
        ));
    }

    Ok(out.trim_end().to_string())
}

// ─── Workspace detection ───────────────────────────────────────────────────────

/// Walk up from cwd to find the directory containing a Cargo.toml with [workspace].
fn find_workspace_root() -> Result<std::path::PathBuf, String> {
    let cwd = std::env::current_dir()
        .map_err(|e| format!("cannot determine current directory: {e}"))?;

    let mut dir = cwd.as_path();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = std::fs::read_to_string(&cargo_toml)
                .map_err(|e| format!("failed to read {}: {e}", cargo_toml.display()))?;
            if content.contains("[workspace]") {
                return Ok(dir.to_path_buf());
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => {
                return Err(
                    "could not find workspace root (no Cargo.toml with [workspace] found)"
                        .to_string(),
                );
            }
        }
    }
}
