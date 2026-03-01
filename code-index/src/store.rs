use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};

use crate::vdb::{Hnsw, HnswNode, SimFn};

// ─── Data types ───────────────────────────────────────────────────────────────

/// Chunk metadata. Vectors live in the HNSW index, not here.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub source_path: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct Note {
    pub id: String,
    pub text: String,
    pub tags: Vec<String>,
    pub timestamp: String,
    pub vector: Vec<f32>, // Notes are few enough for brute-force; no HNSW needed.
}

// ─── Store ────────────────────────────────────────────────────────────────────

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(index_dir: &Path) -> Result<Self, String> {
        let db_path = index_dir.join("index.db");
        let conn = Connection::open(&db_path)
            .map_err(|e| format!("failed to open {}: {e}", db_path.display()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| format!("WAL pragma failed: {e}"))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                kind        TEXT NOT NULL,
                name        TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id        TEXT PRIMARY KEY,
                text      TEXT NOT NULL,
                tags      TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                vector    BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hnsw_nodes (
                idx    INTEGER PRIMARY KEY,
                id     TEXT NOT NULL,
                vector BLOB NOT NULL,
                level  INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hnsw_edges (
                node_idx     INTEGER NOT NULL,
                layer        INTEGER NOT NULL,
                neighbor_idx INTEGER NOT NULL,
                PRIMARY KEY (node_idx, layer, neighbor_idx)
            );
            CREATE TABLE IF NOT EXISTS hnsw_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )
        .map_err(|e| format!("schema init failed: {e}"))?;
        Ok(Self { conn })
    }

    // ─── Chunks ───────────────────────────────────────────────────────────────

    /// Atomically replace all chunk metadata (used by reindex).
    pub fn replace_chunks(&mut self, chunks: &[Chunk]) -> Result<(), String> {
        let tx = self
            .conn
            .transaction()
            .map_err(|e| format!("transaction start failed: {e}"))?;
        tx.execute("DELETE FROM chunks", [])
            .map_err(|e| format!("DELETE chunks failed: {e}"))?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO chunks (id, kind, name, source_path, source)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(|e| format!("prepare insert failed: {e}"))?;
            for c in chunks {
                stmt.execute(params![c.id, c.kind, c.name, c.source_path, c.source])
                    .map_err(|e| format!("insert chunk '{}' failed: {e}", c.id))?;
            }
        }
        tx.commit()
            .map_err(|e| format!("transaction commit failed: {e}"))?;
        Ok(())
    }

    /// Fetch specific chunks by id. Order is not guaranteed.
    pub fn get_chunks_by_ids(&self, ids: &[String]) -> Result<Vec<Chunk>, String> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        // Build parameterised IN clause.
        let placeholders: String = ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT id, kind, name, source_path, source FROM chunks WHERE id IN ({placeholders})"
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| format!("prepare failed: {e}"))?;

        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

        let rows = stmt
            .query_map(params.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .map_err(|e| format!("query failed: {e}"))?;

        rows.map(|r| {
            let (id, kind, name, source_path, source) =
                r.map_err(|e| format!("row read failed: {e}"))?;
            Ok(Chunk { id, kind, name, source_path, source })
        })
        .collect()
    }

    // ─── Notes ────────────────────────────────────────────────────────────────

    pub fn upsert_note(&self, note: &Note) -> Result<(), String> {
        let tags_json = serde_json::to_string(&note.tags)
            .map_err(|e| format!("tags serialization failed: {e}"))?;
        self.conn
            .execute(
                "INSERT OR REPLACE INTO notes (id, text, tags, timestamp, vector)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    note.id,
                    note.text,
                    tags_json,
                    note.timestamp,
                    encode_vector(&note.vector),
                ],
            )
            .map_err(|e| format!("upsert note failed: {e}"))?;
        Ok(())
    }

    pub fn load_notes(&self) -> Result<Vec<Note>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, text, tags, timestamp, vector FROM notes")
            .map_err(|e| format!("prepare failed: {e}"))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                ))
            })
            .map_err(|e| format!("query notes failed: {e}"))?;
        rows.map(|r| {
            let (id, text, tags_json, timestamp, vec_bytes) =
                r.map_err(|e| format!("row read failed: {e}"))?;
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            Ok(Note {
                id,
                text,
                tags,
                timestamp,
                vector: decode_vector(&vec_bytes),
            })
        })
        .collect()
    }

    // ─── HNSW persistence ─────────────────────────────────────────────────────

    /// Persist the entire HNSW graph atomically.
    pub fn save_hnsw(&mut self, hnsw: &Hnsw) -> Result<(), String> {
        let tx = self
            .conn
            .transaction()
            .map_err(|e| format!("transaction start failed: {e}"))?;

        tx.execute("DELETE FROM hnsw_nodes", [])
            .map_err(|e| format!("DELETE hnsw_nodes failed: {e}"))?;
        tx.execute("DELETE FROM hnsw_edges", [])
            .map_err(|e| format!("DELETE hnsw_edges failed: {e}"))?;
        tx.execute("DELETE FROM hnsw_meta", [])
            .map_err(|e| format!("DELETE hnsw_meta failed: {e}"))?;

        {
            let mut node_stmt = tx
                .prepare(
                    "INSERT INTO hnsw_nodes (idx, id, vector, level) VALUES (?1, ?2, ?3, ?4)",
                )
                .map_err(|e| format!("prepare node insert failed: {e}"))?;
            let mut edge_stmt = tx
                .prepare(
                    "INSERT INTO hnsw_edges (node_idx, layer, neighbor_idx) VALUES (?1, ?2, ?3)",
                )
                .map_err(|e| format!("prepare edge insert failed: {e}"))?;

            for (idx, node) in hnsw.nodes.iter().enumerate() {
                node_stmt
                    .execute(params![
                        idx as i64,
                        node.id,
                        encode_vector(&node.vector),
                        node.level as i64,
                    ])
                    .map_err(|e| format!("insert node {idx} failed: {e}"))?;

                for (layer, neighbors) in node.neighbors.iter().enumerate() {
                    for &ni in neighbors {
                        edge_stmt
                            .execute(params![idx as i64, layer as i64, ni as i64])
                            .map_err(|e| {
                                format!("insert edge ({idx},{layer},{ni}) failed: {e}")
                            })?;
                    }
                }
            }
        }

        // Store scalar metadata.
        let ep_val = hnsw
            .entry_point
            .map(|i| i.to_string())
            .unwrap_or_else(|| "null".to_string());
        for (k, v) in [
            ("entry_point", ep_val.as_str()),
            ("max_level", &hnsw.max_level.to_string()),
            ("m", &hnsw.m.to_string()),
            ("m_max0", &hnsw.m_max0.to_string()),
            ("ef_construction", &hnsw.ef_construction.to_string()),
        ] {
            tx.execute(
                "INSERT INTO hnsw_meta (key, value) VALUES (?1, ?2)",
                params![k, v],
            )
            .map_err(|e| format!("insert meta '{k}' failed: {e}"))?;
        }

        tx.commit()
            .map_err(|e| format!("transaction commit failed: {e}"))?;
        Ok(())
    }

    /// Load the HNSW graph from SQLite. Returns an empty index if nothing is saved.
    pub fn load_hnsw(&self, sim: SimFn) -> Result<Hnsw, String> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM hnsw_nodes", [], |r| r.get(0))
            .map_err(|e| format!("count hnsw_nodes failed: {e}"))?;

        if count == 0 {
            return Ok(Hnsw::new(sim));
        }

        // Load scalar metadata.
        let meta = |key: &str| -> Result<String, String> {
            self.conn
                .query_row(
                    "SELECT value FROM hnsw_meta WHERE key = ?1",
                    [key],
                    |r| r.get::<_, String>(0),
                )
                .map_err(|e| format!("load meta '{key}' failed: {e}"))
        };

        let entry_point: Option<usize> = match meta("entry_point")?.as_str() {
            "null" => None,
            s => Some(
                s.parse::<usize>()
                    .map_err(|e| format!("parse entry_point failed: {e}"))?,
            ),
        };
        let max_level = meta("max_level")?
            .parse::<usize>()
            .map_err(|e| format!("parse max_level failed: {e}"))?;
        let m = meta("m")?
            .parse::<usize>()
            .map_err(|e| format!("parse m failed: {e}"))?;
        let ef_construction = meta("ef_construction")?
            .parse::<usize>()
            .map_err(|e| format!("parse ef_construction failed: {e}"))?;

        // Load nodes ordered by idx.
        let mut stmt = self
            .conn
            .prepare("SELECT idx, id, vector, level FROM hnsw_nodes ORDER BY idx")
            .map_err(|e| format!("prepare node select failed: {e}"))?;
        let node_rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })
            .map_err(|e| format!("query nodes failed: {e}"))?;

        let mut nodes: Vec<HnswNode> = node_rows
            .map(|r| {
                let (_idx, id, vec_bytes, level) =
                    r.map_err(|e| format!("node row failed: {e}"))?;
                let level = level as usize;
                Ok(HnswNode {
                    id,
                    vector: decode_vector(&vec_bytes),
                    level,
                    neighbors: vec![Vec::new(); level + 1],
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Load edges and populate neighbor lists.
        let mut edge_stmt = self
            .conn
            .prepare(
                "SELECT node_idx, layer, neighbor_idx FROM hnsw_edges
                 ORDER BY node_idx, layer, neighbor_idx",
            )
            .map_err(|e| format!("prepare edge select failed: {e}"))?;
        let edge_rows = edge_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(|e| format!("query edges failed: {e}"))?;

        for r in edge_rows {
            let (node_idx, layer, neighbor_idx) =
                r.map_err(|e| format!("edge row failed: {e}"))?;
            let (ni, l, nn) = (node_idx as usize, layer as usize, neighbor_idx as usize);
            if ni < nodes.len() && l < nodes[ni].neighbors.len() {
                nodes[ni].neighbors[l].push(nn);
            } else {
                return Err(format!(
                    "edge references out-of-bounds: node {ni} layer {l}"
                ));
            }
        }

        Ok(Hnsw::from_saved(nodes, entry_point, max_level, m, ef_construction, sim))
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// djb2 hash over concatenated bytes, formatted as 16 hex chars.
pub fn chunk_id(path: &str, name: &str, kind: &str) -> String {
    let mut h: u64 = 5381;
    for b in path.bytes().chain(name.bytes()).chain(kind.bytes()) {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    format!("{:016x}", h)
}

/// Returns the .code-index/ directory for a workspace, creating it if needed.
pub fn index_dir(workspace_root: &Path) -> Result<PathBuf, String> {
    let dir = workspace_root.join(".code-index");
    if !dir.exists() {
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
    }
    Ok(dir)
}

fn encode_vector(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
