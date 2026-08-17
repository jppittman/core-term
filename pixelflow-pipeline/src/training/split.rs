//! Split manifest for the tiered benchmark corpus (plan 0.2, audit L4).
//!
//! The corpus is split TRAIN / DEV / FINAL **by family** — a family is one
//! `(band, seed)` generator stream — never by random split. Random splits
//! over a single generator leak: the generator reuses structural motifs
//! across draws, so near-duplicates land on both sides and the "held-out"
//! score measures memorization. The checked-in manifest
//! (`pixelflow-pipeline/corpus_split.toml`) is the single source of truth
//! for which families belong to which tier; this module loads it, validates
//! it loudly (overlaps and empty tiers are errors at load, not surprises at
//! training time), and answers tier-assignment queries.
//!
//! The manifest is a deliberately tiny TOML subset — `[section]` headers,
//! integer arrays, `[start, end]` range pairs, and string arrays — parsed
//! here strictly (unknown keys, duplicate keys, and multi-line values are
//! errors) rather than through a TOML dependency the workspace doesn't
//! otherwise need.

use std::collections::HashSet;
use std::fmt;
use std::path::Path;

/// Corpus tier. Ordered by holdout sacredness: `Train < Dev < Final`, so the
/// corpus build can assert it admits more-sacred tiers first and cross-tier
/// duplicates are always dropped from the less sacred side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// Trained on; grows with mined adversarial expressions.
    Train,
    /// Held out by family; drives per-round SELECT decisions.
    Dev,
    /// Held out by family; touched only for publication claims.
    Final,
}

impl Tier {
    /// All tiers, in ascending sacredness order.
    pub const ALL: [Tier; 3] = [Tier::Train, Tier::Dev, Tier::Final];

    /// Lowercase tier name, used in corpus file names (`corpus_train.bin`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tier::Train => "train",
            Tier::Dev => "dev",
            Tier::Final => "final",
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A manifest load/parse/validation failure. Always fatal for the caller —
/// a corpus built from a half-understood manifest is worse than no corpus.
#[derive(Debug)]
pub struct SplitError {
    /// 1-based manifest line the error was detected on, when known.
    line: Option<usize>,
    msg: String,
}

impl SplitError {
    fn new(msg: impl Into<String>) -> Self {
        Self {
            line: None,
            msg: msg.into(),
        }
    }

    fn at_line(line: usize, msg: impl Into<String>) -> Self {
        Self {
            line: Some(line),
            msg: msg.into(),
        }
    }
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "split manifest line {line}: {}", self.msg),
            None => write!(f, "split manifest: {}", self.msg),
        }
    }
}

impl std::error::Error for SplitError {}

/// Inclusive family-seed range `[start, end]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedRange {
    pub start: u64,
    pub end: u64,
}

impl SeedRange {
    #[must_use]
    pub fn contains(self, seed: u64) -> bool {
        (self.start..=self.end).contains(&seed)
    }

    #[must_use]
    pub fn overlaps(self, other: SeedRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Number of seeds in the range. Inclusive, so never zero for a valid
    /// (non-inverted) range.
    #[must_use]
    pub fn count(self) -> u64 {
        self.end - self.start + 1
    }
}

/// One tier's family set: the cartesian product of `bands` and the seeds in
/// `seed_ranges`.
#[derive(Clone, Debug, Default)]
pub struct TierSpec {
    /// Indices into the generator's band table.
    pub bands: Vec<usize>,
    /// Inclusive family-seed ranges, applying to every band of this tier.
    pub seed_ranges: Vec<SeedRange>,
}

impl TierSpec {
    /// Whether the family `(band, seed)` belongs to this tier.
    #[must_use]
    pub fn contains(&self, band: usize, seed: u64) -> bool {
        self.bands.contains(&band) && self.seed_ranges.iter().any(|r| r.contains(seed))
    }

    /// Total number of `(band, seed)` families in this tier.
    ///
    /// # Panics
    ///
    /// Panics if the count overflows `usize` — a manifest describing more
    /// families than addressable memory is a manifest bug.
    #[must_use]
    pub fn family_count(&self) -> usize {
        let seeds: u128 = self.seed_ranges.iter().map(|r| u128::from(r.count())).sum();
        let total = seeds * self.bands.len() as u128;
        usize::try_from(total)
            .unwrap_or_else(|_| panic!("TierSpec::family_count overflows usize: {total} families"))
    }

    /// All `(band, seed)` families of this tier, band-major, in manifest
    /// order — the deterministic generation order for the corpus build.
    pub fn families(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.bands.iter().flat_map(move |&band| {
            self.seed_ranges
                .iter()
                .flat_map(move |r| (r.start..=r.end).map(move |seed| (band, seed)))
        })
    }
}

/// The parsed, validated three-tier split.
#[derive(Clone, Debug)]
pub struct SplitManifest {
    pub train: TierSpec,
    pub dev: TierSpec,
    pub final_tier: TierSpec,
    /// Names of the real production kernels in FINAL (resolved to arena
    /// builders by the corpus build; an unknown name is a build-time panic).
    pub final_kernels: Vec<String>,
}

impl SplitManifest {
    /// Load and validate a manifest file. Any I/O, parse, or validation
    /// failure is an error — there is no partial or default manifest.
    pub fn load(path: &Path) -> Result<Self, SplitError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| SplitError::new(format!("failed to read {}: {e}", path.display())))?;
        Self::parse(&text)
    }

    /// Parse and validate manifest text.
    pub fn parse(text: &str) -> Result<Self, SplitError> {
        let raw = parse_sections(text)?;
        let manifest = Self {
            train: raw.tier("train")?,
            dev: raw.tier("dev")?,
            final_tier: raw.tier("final")?,
            final_kernels: raw.final_kernels()?,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// The spec for one tier.
    #[must_use]
    pub fn spec(&self, tier: Tier) -> &TierSpec {
        match tier {
            Tier::Train => &self.train,
            Tier::Dev => &self.dev,
            Tier::Final => &self.final_tier,
        }
    }

    /// Which tier the family `(band, seed)` belongs to, if any. Validation
    /// guarantees at most one tier matches.
    #[must_use]
    pub fn tier_of(&self, band: usize, seed: u64) -> Option<Tier> {
        Tier::ALL
            .into_iter()
            .find(|&tier| self.spec(tier).contains(band, seed))
    }

    /// Largest band index referenced anywhere in the manifest, so the corpus
    /// build can assert it against the generator's band table.
    #[must_use]
    pub fn max_band_index(&self) -> usize {
        Tier::ALL
            .iter()
            .flat_map(|&t| self.spec(t).bands.iter().copied())
            .max()
            .expect("validated manifest has at least one band per tier")
    }

    fn validate(&self) -> Result<(), SplitError> {
        for &tier in &Tier::ALL {
            let spec = self.spec(tier);
            if spec.bands.is_empty() {
                return Err(SplitError::new(format!(
                    "tier [{tier}] has no bands — an empty tier cannot serve its role"
                )));
            }
            if spec.seed_ranges.is_empty() {
                return Err(SplitError::new(format!(
                    "tier [{tier}] has no seed ranges — an empty tier cannot serve its role"
                )));
            }
            let mut seen_bands = HashSet::new();
            for &band in &spec.bands {
                if !seen_bands.insert(band) {
                    return Err(SplitError::new(format!(
                        "tier [{tier}] lists band {band} twice"
                    )));
                }
            }
            for r in &spec.seed_ranges {
                if r.start > r.end {
                    return Err(SplitError::new(format!(
                        "tier [{tier}] has inverted seed range [{}, {}]",
                        r.start, r.end
                    )));
                }
            }
            for (i, a) in spec.seed_ranges.iter().enumerate() {
                for b in &spec.seed_ranges[i + 1..] {
                    if a.overlaps(*b) {
                        return Err(SplitError::new(format!(
                            "tier [{tier}] has overlapping seed ranges [{}, {}] and [{}, {}]",
                            a.start, a.end, b.start, b.end
                        )));
                    }
                }
            }
        }

        if self.final_kernels.is_empty() {
            return Err(SplitError::new(
                "tier [final] lists no kernels — the FINAL tier must contain \
                 the named production kernels",
            ));
        }
        let mut seen_kernels = HashSet::new();
        for name in &self.final_kernels {
            if !seen_kernels.insert(name.as_str()) {
                return Err(SplitError::new(format!(
                    "tier [final] lists kernel \"{name}\" twice"
                )));
            }
        }

        // Cross-tier disjointness: a family claimable by two tiers is
        // leakage by construction.
        for (i, &a) in Tier::ALL.iter().enumerate() {
            for &b in &Tier::ALL[i + 1..] {
                let (sa, sb) = (self.spec(a), self.spec(b));
                for &band in &sa.bands {
                    if !sb.bands.contains(&band) {
                        continue;
                    }
                    for ra in &sa.seed_ranges {
                        for rb in &sb.seed_ranges {
                            if ra.overlaps(*rb) {
                                return Err(SplitError::new(format!(
                                    "tiers [{a}] and [{b}] overlap on band {band}: seed ranges \
                                     [{}, {}] and [{}, {}] intersect — a family may belong to \
                                     exactly one tier",
                                    ra.start, ra.end, rb.start, rb.end
                                )));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ── Strict TOML-subset parsing ──────────────────────────────────────────────

/// Raw per-section key/value storage before assembly into [`SplitManifest`].
#[derive(Default)]
struct RawSection {
    bands: Option<Vec<usize>>,
    seeds: Option<Vec<SeedRange>>,
    kernels: Option<Vec<String>>,
}

struct RawManifest {
    train: RawSection,
    dev: RawSection,
    final_section: RawSection,
}

impl RawManifest {
    fn section(&mut self, name: &str) -> Option<&mut RawSection> {
        match name {
            "train" => Some(&mut self.train),
            "dev" => Some(&mut self.dev),
            "final" => Some(&mut self.final_section),
            _ => None,
        }
    }

    fn section_ref(&self, name: &str) -> &RawSection {
        match name {
            "train" => &self.train,
            "dev" => &self.dev,
            "final" => &self.final_section,
            other => panic!("RawManifest::section_ref: unknown section {other}"),
        }
    }

    fn tier(&self, name: &str) -> Result<TierSpec, SplitError> {
        let section = self.section_ref(name);
        let bands = section
            .bands
            .clone()
            .ok_or_else(|| SplitError::new(format!("section [{name}] is missing `bands`")))?;
        let seed_ranges = section
            .seeds
            .clone()
            .ok_or_else(|| SplitError::new(format!("section [{name}] is missing `seeds`")))?;
        Ok(TierSpec { bands, seed_ranges })
    }

    fn final_kernels(&self) -> Result<Vec<String>, SplitError> {
        self.final_section
            .kernels
            .clone()
            .ok_or_else(|| SplitError::new("section [final] is missing `kernels`"))
    }
}

fn parse_sections(text: &str) -> Result<RawManifest, SplitError> {
    let mut raw = RawManifest {
        train: RawSection::default(),
        dev: RawSection::default(),
        final_section: RawSection::default(),
    };
    let mut sections_seen: HashSet<String> = HashSet::new();
    let mut current: Option<String> = None;

    for (idx, raw_line) in text.lines().enumerate() {
        let lineno = idx + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('[') {
            // Distinguish a section header `[name]` from a value line — value
            // lines always contain `=` before their brackets.
            let name = rest.strip_suffix(']').ok_or_else(|| {
                SplitError::at_line(lineno, format!("malformed section header `{line}`"))
            })?;
            let name = name.trim().to_string();
            if raw.section(&name).is_none() {
                return Err(SplitError::at_line(
                    lineno,
                    format!("unknown section [{name}] (expected [train], [dev], or [final])"),
                ));
            }
            if !sections_seen.insert(name.clone()) {
                return Err(SplitError::at_line(
                    lineno,
                    format!("duplicate section [{name}]"),
                ));
            }
            current = Some(name);
            continue;
        }

        let (key, value) = line.split_once('=').ok_or_else(|| {
            SplitError::at_line(
                lineno,
                format!("expected `key = value` or `[section]`, got `{line}`"),
            )
        })?;
        let key = key.trim();
        let value = value.trim();
        let section_name = current.clone().ok_or_else(|| {
            SplitError::at_line(lineno, format!("key `{key}` appears before any [section]"))
        })?;
        let section = raw
            .section(&section_name)
            .expect("current section validated at header");

        match key {
            "bands" => {
                if section.bands.is_some() {
                    return Err(SplitError::at_line(
                        lineno,
                        format!("duplicate key `bands` in [{section_name}]"),
                    ));
                }
                section.bands = Some(
                    parse_usize_list(value)
                        .map_err(|e| SplitError::at_line(lineno, format!("bands: {e}")))?,
                );
            }
            "seeds" => {
                if section.seeds.is_some() {
                    return Err(SplitError::at_line(
                        lineno,
                        format!("duplicate key `seeds` in [{section_name}]"),
                    ));
                }
                section.seeds = Some(
                    parse_seed_ranges(value)
                        .map_err(|e| SplitError::at_line(lineno, format!("seeds: {e}")))?,
                );
            }
            "kernels" if section_name == "final" => {
                if section.kernels.is_some() {
                    return Err(SplitError::at_line(
                        lineno,
                        "duplicate key `kernels` in [final]".to_string(),
                    ));
                }
                section.kernels = Some(
                    parse_string_list(value)
                        .map_err(|e| SplitError::at_line(lineno, format!("kernels: {e}")))?,
                );
            }
            other => {
                return Err(SplitError::at_line(
                    lineno,
                    format!("unknown key `{other}` in [{section_name}]"),
                ));
            }
        }
    }

    Ok(raw)
}

/// Cut a line at the first `#` that is not inside a double-quoted string.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..i],
            _ => {}
        }
    }
    line
}

/// `value` must be a single-line bracketed array; returns its inner text.
fn strip_brackets(value: &str) -> Result<&str, String> {
    let value = value.trim();
    value
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .ok_or_else(|| format!("expected a single-line `[...]` array, got `{value}`"))
}

/// Split array-inner text on commas at bracket depth 0. Allows one trailing
/// comma; rejects empty elements otherwise.
fn split_top_level(inner: &str) -> Result<Vec<&str>, String> {
    let mut parts = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth < 0 {
                    return Err(format!("unbalanced `]` in `{inner}`"));
                }
            }
            ',' if depth == 0 => {
                parts.push(inner[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(format!("unbalanced `[` in `{inner}`"));
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    } else if !parts.is_empty() && inner.trim().ends_with(',') {
        // Trailing comma: fine, nothing to push.
    }
    if parts.iter().any(|p| p.is_empty()) {
        return Err(format!("empty element in `{inner}`"));
    }
    Ok(parts)
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, String> {
    let inner = strip_brackets(value)?;
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner)?
        .into_iter()
        .map(|p| {
            p.parse::<usize>()
                .map_err(|e| format!("`{p}` is not an unsigned integer: {e}"))
        })
        .collect()
}

fn parse_seed_ranges(value: &str) -> Result<Vec<SeedRange>, String> {
    let inner = strip_brackets(value)?;
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner)?
        .into_iter()
        .map(|pair| {
            let pair_inner = strip_brackets(pair)
                .map_err(|_| format!("expected `[start, end]` pair, got `{pair}`"))?;
            let ends = split_top_level(pair_inner)?;
            let [start, end] = ends.as_slice() else {
                return Err(format!(
                    "expected exactly two values in `[start, end]`, got {} in `{pair}`",
                    ends.len()
                ));
            };
            let start = start
                .parse::<u64>()
                .map_err(|e| format!("`{start}` is not an unsigned integer: {e}"))?;
            let end = end
                .parse::<u64>()
                .map_err(|e| format!("`{end}` is not an unsigned integer: {e}"))?;
            Ok(SeedRange { start, end })
        })
        .collect()
}

fn parse_string_list(value: &str) -> Result<Vec<String>, String> {
    let inner = strip_brackets(value)?;
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner)?
        .into_iter()
        .map(|p| {
            let unquoted = p
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .ok_or_else(|| format!("expected a double-quoted string, got `{p}`"))?;
            if unquoted.is_empty() {
                return Err("empty string element".to_string());
            }
            if unquoted.contains(['"', '\\']) {
                return Err(format!(
                    "escapes and embedded quotes are not supported: `{p}`"
                ));
            }
            Ok(unquoted.to_string())
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
# leading comment
[train]
bands = [0, 2]  # trailing comment
seeds = [[0, 7]]

[dev]
bands = [1]
seeds = [[0, 7]]

[final]
bands = [3]
seeds = [[0, 3], [5, 6]]
kernels = ["swirl", "poly"]
"#;

    #[test]
    fn parses_valid_manifest_and_assigns_tiers() {
        let m = SplitManifest::parse(VALID).expect("valid manifest must parse");
        assert_eq!(m.tier_of(0, 0), Some(Tier::Train));
        assert_eq!(m.tier_of(2, 7), Some(Tier::Train));
        assert_eq!(m.tier_of(1, 3), Some(Tier::Dev));
        assert_eq!(m.tier_of(3, 2), Some(Tier::Final));
        assert_eq!(m.tier_of(3, 5), Some(Tier::Final));
        // Seed 4 falls in the gap between final's ranges.
        assert_eq!(m.tier_of(3, 4), None);
        // Band 9 is in no tier; seed 8 is outside every range.
        assert_eq!(m.tier_of(9, 0), None);
        assert_eq!(m.tier_of(0, 8), None);
        assert_eq!(m.final_kernels, vec!["swirl", "poly"]);
        assert_eq!(m.max_band_index(), 3);
        // train: 2 bands x 8 seeds; final: 1 band x (4 + 2) seeds.
        assert_eq!(m.train.family_count(), 16);
        assert_eq!(m.final_tier.family_count(), 6);
        let fams: Vec<(usize, u64)> = m.final_tier.families().collect();
        assert_eq!(fams, vec![(3, 0), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6)]);
    }

    #[test]
    fn checked_in_manifest_is_valid() {
        let text = include_str!("../../corpus_split.toml");
        let m = SplitManifest::parse(text).expect("checked-in corpus_split.toml must be valid");
        assert_eq!(
            m.final_kernels.len(),
            5,
            "the five named production kernels"
        );
        assert_eq!(m.max_band_index(), 37, "band table has 38 bands (0..=37)");
        // Spot-check the by-family holdout: dev band 22 is dev at every seed,
        // and no train band collides with it.
        assert_eq!(m.tier_of(22, 0), Some(Tier::Dev));
        assert_eq!(m.tier_of(22, 7), Some(Tier::Dev));
        assert_eq!(m.tier_of(12, 3), Some(Tier::Final));
        assert_eq!(m.tier_of(0, 3), Some(Tier::Train));
    }

    #[test]
    fn shared_band_with_disjoint_seed_ranges_is_valid() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = [0]
seeds = [[8, 9]]
[final]
bands = [1]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let m = SplitManifest::parse(text).expect("disjoint seed ranges on a shared band are fine");
        assert_eq!(m.tier_of(0, 7), Some(Tier::Train));
        assert_eq!(m.tier_of(0, 8), Some(Tier::Dev));
    }

    #[test]
    fn cross_tier_overlap_is_rejected() {
        let text = r#"
[train]
bands = [0, 1]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[5, 9]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("overlap must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("overlap"), "unexpected error: {msg}");
        assert!(msg.contains("band 1"), "unexpected error: {msg}");
    }

    #[test]
    fn within_tier_seed_range_overlap_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 4], [4, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("within-tier overlap must be rejected");
        assert!(
            err.to_string().contains("overlapping seed ranges"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_tier_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = []
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("empty tier must be rejected");
        assert!(
            err.to_string().contains("[dev] has no bands"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn inverted_seed_range_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[7, 0]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("inverted range must be rejected");
        assert!(
            err.to_string().contains("inverted seed range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_kernels_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
"#;
        let err = SplitManifest::parse(text).expect_err("missing kernels must be rejected");
        assert!(
            err.to_string().contains("missing `kernels`"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn duplicate_band_is_rejected() {
        let text = r#"
[train]
bands = [0, 0]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("duplicate band must be rejected");
        assert!(
            err.to_string().contains("band 0 twice"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unknown_key_and_unknown_section_are_rejected() {
        let err = SplitManifest::parse("[train]\nbandz = [0]\n").expect_err("unknown key");
        assert!(err.to_string().contains("unknown key"), "got: {err}");

        let err = SplitManifest::parse("[trainn]\n").expect_err("unknown section");
        assert!(err.to_string().contains("unknown section"), "got: {err}");
    }

    #[test]
    fn kernels_outside_final_is_rejected() {
        let err = SplitManifest::parse("[train]\nkernels = [\"swirl\"]\n")
            .expect_err("kernels key belongs to [final] only");
        assert!(err.to_string().contains("unknown key"), "got: {err}");
    }

    #[test]
    fn malformed_values_are_rejected_with_line_numbers() {
        let err = SplitManifest::parse("[train]\nbands = [0, x]\n").expect_err("bad integer");
        let msg = err.to_string();
        assert!(msg.contains("line 2"), "got: {msg}");

        let err =
            SplitManifest::parse("[train]\nseeds = [[0]]\n").expect_err("range needs two values");
        assert!(err.to_string().contains("exactly two"), "got: {err}");

        let err = SplitManifest::parse("[train]\nbands = [0\n").expect_err("unbalanced bracket");
        assert!(err.to_string().contains("array"), "got: {err}");
    }

    #[test]
    fn seed_range_contains_count_and_overlaps_match_their_definitions() {
        let r = SeedRange { start: 2, end: 5 };
        assert!(r.contains(2) && r.contains(5) && !r.contains(6));
        assert_eq!(r.count(), 4);
        assert!(r.overlaps(SeedRange { start: 5, end: 9 }));
        assert!(!r.overlaps(SeedRange { start: 6, end: 9 }));
    }

    #[test]
    fn tier_ordering_reflects_sacredness() {
        assert!(Tier::Train < Tier::Dev);
        assert!(Tier::Dev < Tier::Final);
    }
}
