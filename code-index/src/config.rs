use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub search: SearchConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub workspaces: Vec<String>,
    pub reindex_interval_minutes: u64,
    pub extensions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub top_k: usize,
    /// Minimum cosine similarity to inject. Results below this are dropped.
    /// 0.65 keeps clearly on-topic chunks and discards noise from unrelated prompts.
    pub min_score: f32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            workspaces: vec![],
            reindex_interval_minutes: 10,
            extensions: vec![
                "rs".into(),
                "py".into(),
                "ts".into(),
                "go".into(),
                "c".into(),
                "h".into(),
            ],
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Self {
            dir: format!("{home}/.code-index/models/embeddinggemma"),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.65,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            index: IndexConfig::default(),
            model: ModelConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, String> {
        let path = config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        toml::from_str(&text)
            .map_err(|e| format!("invalid config at {}: {e}", path.display()))
    }

    pub fn save(&self) -> Result<(), String> {
        let path = config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create config dir: {e}"))?;
        }
        let text = toml::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize config: {e}"))?;
        std::fs::write(&path, text)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    pub fn expanded_workspaces(&self) -> Vec<PathBuf> {
        self.index
            .workspaces
            .iter()
            .map(|w| expand_tilde(w))
            .collect()
    }

    pub fn model_dir(&self) -> PathBuf {
        expand_tilde(&self.model.dir)
    }
}

fn config_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".code-index").join("config.toml")
}

fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(&path[2..])
    } else {
        PathBuf::from(path)
    }
}
