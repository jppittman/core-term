// code-index: language-agnostic semantic code search.
//
// Subcommands:
//   serve    — MCP server (JSON-RPC 2.0 over stdio) for Claude/Gemini/etc.
//   reindex  — walk workspaces, embed chunks, write index
//   query    — read {"prompt":"..."} from stdin, print top-k chunks to stdout
//   install  — lazy model download, daemon setup (launchd/systemd)

mod backends;
mod config;
mod download;
mod embed;
mod install;
mod mcp;
mod parse;
mod store;
mod tools;
mod vdb;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "code-index", about = "Semantic code search with EmbeddingGemma")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the MCP server (JSON-RPC 2.0 over stdio)
    Serve,
    /// Walk workspaces and rebuild the search index
    Reindex,
    /// Read {"prompt":"..."} from stdin, print relevant chunks to stdout (for LLM hooks)
    Query,
    /// Search and format chunks using a specific LLM hook backend
    Hook {
        /// Backend to use: claude, gemini
        #[arg(long, default_value = "claude")]
        backend: String,
        /// The query/prompt to search for
        query: String,
    },
    /// Download model and set up daemon for periodic reindexing
    Install,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Cmd::Serve => cmd_serve(),
        Cmd::Reindex => {
            let config = config::Config::load().unwrap_or_else(|e| {
                eprintln!("Config error: {e}");
                std::process::exit(1);
            });
            cmd_reindex(&config)
        }
        Cmd::Query => {
            let config = config::Config::load().unwrap_or_else(|e| {
                eprintln!("Config error: {e}");
                std::process::exit(1);
            });
            cmd_query(&config)
        }
        Cmd::Hook { backend, query } => {
            let config = config::Config::load().unwrap_or_else(|e| {
                eprintln!("Config error: {e}");
                std::process::exit(1);
            });
            let embedder = embed::Embedder::load(&config.model_dir()).unwrap_or_else(|e| {
                eprintln!("Embedder error: {e}");
                std::process::exit(1);
            });
            let backend = backends::from_name(&backend).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
            cmd_hook(&embedder, &config, &*backend, &query)
        }
        Cmd::Install => {
            let config = config::Config::load().unwrap_or_else(|_| config::Config::default());
            install::install(&config)
        }
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn cmd_serve() -> Result<(), String> {
    use std::io::{BufRead, Write};
    let config = config::Config::load().unwrap_or_default();
    let embedder = embed::Embedder::load(&config.model_dir())?;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = std::io::BufReader::new(stdin.lock());
    let mut writer = std::io::BufWriter::new(stdout.lock());
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("stdin error: {e}");
                break;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("JSON error: {e}");
                continue;
            }
        };
        if let Some(resp) = tools::handle(&embedder, &config, &req) {
            let mut s = serde_json::to_string(&resp)
                .map_err(|e| format!("response serialization failed: {e}"))?;
            s.push('\n');
            writer
                .write_all(s.as_bytes())
                .map_err(|e| format!("stdout write failed: {e}"))?;
            writer
                .flush()
                .map_err(|e| format!("stdout flush failed: {e}"))?;
        }
    }
    Ok(())
}

pub fn cmd_reindex(config: &config::Config) -> Result<(), String> {
    let embedder = embed::Embedder::load(&config.model_dir())?;
    let workspaces = config.expanded_workspaces();
    if workspaces.is_empty() {
        eprintln!("[reindex] No workspaces configured. Edit ~/.code-index/config.toml");
        return Ok(());
    }
    for ws in &workspaces {
        eprintln!("[reindex] Indexing {}", ws.display());
        tools::reindex_workspace(&embedder, config, ws)?;
    }
    Ok(())
}

fn cmd_query(config: &config::Config) -> Result<(), String> {
    use std::io::Read;
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("failed to read stdin: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(input.trim())
        .map_err(|e| format!("invalid JSON on stdin: {e}"))?;
    let prompt = json["prompt"]
        .as_str()
        .ok_or("missing 'prompt' field in stdin JSON")?;
    if prompt.trim().is_empty() {
        return Ok(());
    }
    let embedder = embed::Embedder::load(&config.model_dir())?;
    let backend = backends::claude::ClaudeBackend;
    let results = tools::query_all_workspaces(&embedder, config, prompt, &backend)?;
    if !results.is_empty() {
        println!("--- Relevant code context ---");
        println!("{results}");
        println!("--- End context ---");
    }
    Ok(())
}

fn cmd_hook(
    embedder: &embed::Embedder,
    config: &config::Config,
    backend: &dyn backends::HookBackend,
    query: &str,
) -> Result<(), String> {
    let scored = tools::search_workspaces(embedder, config, query)?;
    let output = backend.format_results(&scored, config.search.top_k);
    if !output.is_empty() {
        println!("{output}");
    }
    Ok(())
}
