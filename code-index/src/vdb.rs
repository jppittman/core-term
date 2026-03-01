//! HNSW — Hierarchical Navigable Small World
//!
//! An approximate nearest-neighbor index on unit-normalized f32 vectors.
//! Uses dot product as the similarity function (= cosine on normalized vectors).
//!
//! Parameters (defaults tuned for code-chunk retrieval):
//!   M              = 16   max neighbors per layer (except layer 0)
//!   M_max0         = 32   max neighbors at layer 0
//!   ef_construction = 200 candidate set size while building
//!   mL             = 1/ln(M) ≈ 0.366  level normalization
//!
//! Complexity:
//!   insert   O(log n) expected
//!   search   O(log n) expected

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Similarity function: higher = more similar. Must be in [0, 1] for this impl.
/// Use `dot` for pre-normalized vectors (= cosine similarity).
pub type SimFn = fn(&[f32], &[f32]) -> f32;

/// A single search result.
pub struct Hit {
    pub id: String,
    pub score: f32, // similarity in [0, 1], higher = more similar
}

// ─── Internal types ───────────────────────────────────────────────────────────

/// NaN-safe f32 wrapper for BinaryHeap (treats NaN as +infinity / worst).
#[derive(Clone, Copy, PartialEq)]
struct D(f32);

impl Eq for D {}

impl PartialOrd for D {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for D {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Greater) // NaN → biggest (worst)
    }
}

pub(crate) struct HnswNode {
    pub(crate) id: String,
    pub(crate) vector: Vec<f32>,
    pub(crate) level: usize,
    /// `neighbors[l]` = neighbor node indices at layer l (0 ≤ l ≤ level).
    pub(crate) neighbors: Vec<Vec<usize>>,
}

// ─── Hnsw ─────────────────────────────────────────────────────────────────────

pub struct Hnsw {
    pub(crate) nodes: Vec<HnswNode>,
    pub(crate) entry_point: Option<usize>,
    pub(crate) max_level: usize,
    pub(crate) m: usize,
    pub(crate) m_max0: usize,
    pub(crate) ef_construction: usize,
    ml: f64,  // = 1/ln(m), level normalization
    rng: u64, // xorshift64 state for level assignment
    sim: SimFn,
}

impl Hnsw {
    pub fn new(sim: SimFn) -> Self {
        Self::with_params(16, 200, sim)
    }

    /// `m`: max neighbors per layer. `ef_construction`: candidate list during build.
    /// Larger m → better recall, more memory. Larger ef_c → better recall, slower build.
    pub fn with_params(m: usize, ef_construction: usize, sim: SimFn) -> Self {
        let m = m.max(2);
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            m,
            m_max0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            rng: 0xdeadbeef_cafebabe,
            sim,
        }
    }

    /// Reconstruct an Hnsw from persisted data (used by Store::load_hnsw).
    pub(crate) fn from_saved(
        nodes: Vec<HnswNode>,
        entry_point: Option<usize>,
        max_level: usize,
        m: usize,
        ef_construction: usize,
        sim: SimFn,
    ) -> Self {
        let m = m.max(2);
        Self {
            nodes,
            entry_point,
            max_level,
            m,
            m_max0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            rng: 0xdeadbeef_cafebabe,
            sim,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ─── Insert ───────────────────────────────────────────────────────────────

    /// Add a vector to the index. `id` is the external identifier (chunk id).
    /// Vectors should be L2-normalized so that `sim` = dot product = cosine.
    pub fn insert(&mut self, id: &str, vector: Vec<f32>) {
        let new_idx = self.nodes.len();
        let level = self.random_level();

        // Push the node first so bidirectional trim can access its vector.
        self.nodes.push(HnswNode {
            id: id.to_string(),
            vector,
            level,
            neighbors: vec![Vec::new(); level + 1],
        });

        // First node: becomes the entry point and we're done.
        let ep = match self.entry_point {
            None => {
                self.entry_point = Some(new_idx);
                self.max_level = level;
                return;
            }
            Some(ep) => ep,
        };

        let mut curr_eps = vec![ep];

        // Phase 1: greedy descent from max_level down to level+1 (ef=1 per layer).
        for l in (level + 1..=self.max_level).rev() {
            let w = self.search_layer_idx(new_idx, &curr_eps, 1, l);
            curr_eps = vec![w[0].1];
        }

        // Phase 2: connect from min(max_level, level) down to 0.
        for l in (0..=level.min(self.max_level)).rev() {
            let m_l = if l == 0 { self.m_max0 } else { self.m };
            let w = self.search_layer_idx(new_idx, &curr_eps, self.ef_construction, l);

            // Select m_l nearest as neighbors for the new node.
            let selected: Vec<usize> = w.iter().take(m_l).map(|(_, ni)| *ni).collect();
            self.nodes[new_idx].neighbors[l] = selected.clone();

            // Add bidirectional edges and trim if over limit.
            for &ni in &selected {
                // Guard: ni must exist at layer l.
                if l < self.nodes[ni].neighbors.len() {
                    self.nodes[ni].neighbors[l].push(new_idx);
                    if self.nodes[ni].neighbors[l].len() > m_l {
                        self.trim_neighbors(ni, l, m_l);
                    }
                }
            }

            curr_eps = w.into_iter().map(|(_, ni)| ni).collect();
        }

        // Promote entry point if this node reaches a new maximum level.
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(new_idx);
        }
    }

    // ─── Search ───────────────────────────────────────────────────────────────

    /// Find the `k` most similar vectors. `ef` controls recall vs speed (ef ≥ k).
    /// Returns hits sorted descending by similarity score.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Hit> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let mut curr_eps = vec![ep];

        // Greedy descent to layer 1.
        for l in (1..=self.max_level).rev() {
            let w = self.search_layer(query, &curr_eps, 1, l);
            curr_eps = vec![w[0].1];
        }

        // Full beam search at layer 0.
        let w = self.search_layer(query, &curr_eps, ef.max(k), 0);

        w.into_iter()
            .take(k)
            .map(|(dist, idx)| Hit {
                id: self.nodes[idx].id.clone(),
                score: 1.0 - dist, // convert distance back to similarity
            })
            .collect()
    }

    // ─── Core layer search ────────────────────────────────────────────────────

    /// Greedy beam search at a single layer. Returns candidates sorted ascending
    /// by distance (closest first). Distance = 1 - sim(q, node).
    fn search_layer(
        &self,
        q: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, usize)> {
        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();

        // C: min-heap on distance — we explore the closest candidate first.
        let mut candidates: BinaryHeap<Reverse<(D, usize)>> = BinaryHeap::new();
        // W: max-heap on distance — we prune the furthest when |W| > ef.
        let mut found: BinaryHeap<(D, usize)> = BinaryHeap::new();

        for &ep in entry_points {
            let d = D(self.dist(q, &self.nodes[ep].vector));
            candidates.push(Reverse((d, ep)));
            found.push((d, ep));
        }

        while let Some(Reverse((d_c, c))) = candidates.pop() {
            let d_f = found.peek().map(|(d, _)| *d).unwrap_or(D(f32::MAX));
            if d_c > d_f {
                break; // every remaining candidate is worse than the current worst found
            }

            let neighbors: Vec<usize> = self.nodes[c]
                .neighbors
                .get(layer)
                .cloned()
                .unwrap_or_default();

            for e in neighbors {
                if !visited.insert(e) {
                    continue;
                }
                let d_e = D(self.dist(q, &self.nodes[e].vector));
                let d_f = found.peek().map(|(d, _)| *d).unwrap_or(D(f32::MAX));

                if d_e < d_f || found.len() < ef {
                    candidates.push(Reverse((d_e, e)));
                    found.push((d_e, e));
                    if found.len() > ef {
                        found.pop();
                    }
                }
            }
        }

        let mut result: Vec<(f32, usize)> =
            found.into_iter().map(|(d, idx)| (d.0, idx)).collect();
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Wrapper that uses `new_idx`'s own vector as the query.
    /// Avoids borrowing the node's vector while we still need to mutate nodes.
    fn search_layer_idx(
        &self,
        query_idx: usize,
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, usize)> {
        // Clone the query vector so search_layer can borrow self freely.
        let q = self.nodes[query_idx].vector.clone();
        self.search_layer(&q, entry_points, ef, layer)
    }

    // ─── Neighbor management ──────────────────────────────────────────────────

    /// Shrink `node_idx`'s neighbor list at `layer` to `max_m` by keeping the closest.
    fn trim_neighbors(&mut self, node_idx: usize, layer: usize, max_m: usize) {
        let base = self.nodes[node_idx].vector.clone();
        let all: Vec<usize> = self.nodes[node_idx].neighbors[layer].clone();

        let mut scored: Vec<(f32, usize)> = all
            .iter()
            .map(|&ni| (self.dist(&base, &self.nodes[ni].vector), ni))
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.nodes[node_idx].neighbors[layer] =
            scored.into_iter().take(max_m).map(|(_, ni)| ni).collect();
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    /// Distance = 1 - similarity. In [0, 2] for cosine, [0, 1] for dot on unit vecs.
    #[inline(always)]
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - (self.sim)(a, b)
    }

    /// Sample a level from a geometric distribution truncated to reasonable depth.
    fn random_level(&mut self) -> usize {
        // xorshift64
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        // Map to [0, 1)
        let u = (self.rng >> 11) as f64 / (1u64 << 53) as f64;
        let u = u.max(f64::MIN_POSITIVE); // avoid ln(0)
        (-u.ln() * self.ml).floor() as usize
    }
}

// ─── Similarity functions ─────────────────────────────────────────────────────

/// Dot product. Equals cosine similarity on L2-normalized vectors.
/// This is the fast path: no sqrt, no division.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Full cosine similarity. Use when vectors are not pre-normalized.
/// Ready for when pixelflow's map/reduce lands and we benchmark the two paths.
#[allow(dead_code)]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut d = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}
