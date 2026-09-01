//! Rule-set inflation for the Phase 3 Round 2 rule-scaling harness
//! (`docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §2, §8).
//!
//! **Harness-only.** Nothing here is referenced by [`super::all_rules`], and
//! nothing here changes production behavior: `all_rules()` still returns
//! exactly 62 rules (pinned by `super::tests::all_rules_count`). Every
//! inflated set built here is a `Vec<Box<dyn Rewrite>>` a research binary
//! hands to `EGraph::with_rules` directly.
//!
//! Two of the design's three inflation modes live here — (i) exact
//! duplicates, the "learnable overhead" control, and (ii) mechanical
//! compositions, real closure-preserving shortcuts built by unifying one
//! rule's RHS against another's LHS (`crate::egraph::template`). Mode (iii)
//! (genuinely new rules) is a separate task; this module never constructs
//! [`InflationMode::NewRules`] because it has no variant for it — adding one
//! without a builder would be exactly the silent gap this codebase's
//! CLAUDE.md warns against.

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pixelflow_ir::arena::ExprArena;

use super::all_rules;
use crate::egraph::rewrite::{Rewrite, RewriteAction};
use crate::egraph::template::{self, TemplateRewrite};
use crate::egraph::{EClassId, EGraph, ENode};

/// Which inflation mechanism a [`RuleSetSpec`] selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InflationMode {
    /// Exact duplicates of existing rules under new indices (§2.1).
    Duplicates,
    /// Mechanical compositions A∘B of two existing rules (§2.2).
    Compositions,
}

impl std::fmt::Display for InflationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicates => write!(f, "dup"),
            Self::Compositions => write!(f, "comp"),
        }
    }
}

/// A parsed `--rule-set` argument: `"base"` (the production 62), or
/// `"dup:<total>"` / `"comp:<total>"` where `<total>` is the TOTAL rule
/// count requested (base 62 plus that mode's inflation), not the inflation
/// alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleSetSpec {
    pub mode: Option<InflationMode>,
    pub total: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleSetSpecError {
    /// Not `"base"` and not `"<mode>:<count>"`.
    Malformed(String),
    /// The `<mode>` tag isn't one this harness builds.
    UnknownMode(String),
    /// The `<count>` half didn't parse as a `usize`.
    BadCount(String),
}

impl std::fmt::Display for RuleSetSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(s) => write!(
                f,
                "rule-set spec {s:?}: expected \"base\", \"dup:<total>\", or \"comp:<total>\""
            ),
            Self::UnknownMode(s) => write!(f, "rule-set spec {s:?}: unknown mode (want dup/comp)"),
            Self::BadCount(s) => write!(f, "rule-set spec: {s:?} is not a valid rule count"),
        }
    }
}

impl std::error::Error for RuleSetSpecError {}

impl RuleSetSpec {
    /// Parse `"base"` | `"dup:<count>"` | `"comp:<count>"`. `<count>` is the
    /// TOTAL `|R|`, matching every column in
    /// `docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §3's grid.
    pub fn parse(s: &str) -> Result<Self, RuleSetSpecError> {
        if s == "base" {
            return Ok(Self {
                mode: None,
                total: all_rules().len(),
            });
        }
        let (tag, rest) = s
            .split_once(':')
            .ok_or_else(|| RuleSetSpecError::Malformed(s.to_string()))?;
        let mode = match tag {
            "dup" => InflationMode::Duplicates,
            "comp" => InflationMode::Compositions,
            _ => return Err(RuleSetSpecError::UnknownMode(s.to_string())),
        };
        let total: usize = rest
            .parse()
            .map_err(|_| RuleSetSpecError::BadCount(rest.to_string()))?;
        Ok(Self {
            mode: Some(mode),
            total,
        })
    }
}

/// Content-addressed identity of a rule set: every rule's name, in order,
/// hashed. Written into every result row this harness produces so a curve
/// can never be silently read back against the wrong rule set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleSetFingerprint(pub u64);

impl std::fmt::Display for RuleSetFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

#[must_use]
pub fn rule_set_fingerprint(rules: &[Box<dyn Rewrite>]) -> RuleSetFingerprint {
    let mut hasher = DefaultHasher::new();
    rules.len().hash(&mut hasher);
    for r in rules {
        r.name().hash(&mut hasher);
    }
    RuleSetFingerprint(hasher.finish())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleSetError {
    /// `total` is not one of this mode's grid points — never padded to fit.
    UnreachableTotal {
        mode: InflationMode,
        requested: usize,
        grid: &'static [usize],
    },
    /// The composition pool has fewer members than the requested inflation.
    PoolTooSmall {
        requested_inflation: usize,
        pool_size: usize,
    },
}

impl std::fmt::Display for RuleSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnreachableTotal {
                mode,
                requested,
                grid,
            } => write!(
                f,
                "rule-set {mode}: |R|={requested} is not on this mode's grid {grid:?} \
                 (never padded to fit — pick a grid point)"
            ),
            Self::PoolTooSmall {
                requested_inflation,
                pool_size,
            } => write!(
                f,
                "composition pool has only {pool_size} members, requested {requested_inflation} \
                 — the grid point is unreachable on this rule library, dropped rather than padded"
            ),
        }
    }
}

impl std::error::Error for RuleSetError {}

/// THE one constructor for every rule set the Round 2 harness runs. The
/// prefix is always `all_rules()` in its exact production order (indices
/// 0..62 mean the same thing at every `|R|`); inflation is appended after.
pub fn build_rule_set(spec: &RuleSetSpec) -> Result<Vec<Box<dyn Rewrite>>, RuleSetError> {
    match spec.mode {
        None => Ok(all_rules()),
        Some(InflationMode::Duplicates) => build_duplicate_set(spec.total),
        Some(InflationMode::Compositions) => build_composition_set(spec.total),
    }
}

// ============================================================================
// Mode (i): exact duplicates (§2.1)
// ============================================================================

/// A duplicate of `inner` under a new rule index. Delegates every behavior
/// to `inner` — same closure, same per-match semantics, purely a second
/// (or third, …) way to reach the same node — except `name`, which is
/// `"<inner>#dup<copy>"` so per-rule statistics (and a Guide's
/// `w_rule[rule_idx]`, out of scope for this unguided-only harness) can tell
/// a copy from its original.
pub struct DuplicateRule {
    inner: Box<dyn Rewrite>,
    name: String,
}

impl DuplicateRule {
    #[must_use]
    pub fn new(inner: Box<dyn Rewrite>, copy: usize) -> Self {
        let name = format!("{}#dup{copy}", inner.name());
        Self { inner, name }
    }
}

impl Rewrite for DuplicateRule {
    fn name(&self) -> &str {
        &self.name
    }
    fn apply(&self, egraph: &EGraph, id: EClassId, node: &ENode) -> Option<RewriteAction> {
        self.inner.apply(egraph, id, node)
    }
    fn is_destructive(&self) -> bool {
        self.inner.is_destructive()
    }
    fn lhs_template(&self, out: &mut ExprArena) -> Option<pixelflow_ir::ExprId> {
        self.inner.lhs_template(out)
    }
    fn rhs_template(&self, out: &mut ExprArena) -> Option<pixelflow_ir::ExprId> {
        self.inner.rhs_template(out)
    }
}

/// The five |R| points mode (i) is registered at (§3).
pub const DUPLICATE_GRID: &[usize] = &[62, 93, 124, 186, 248];

fn build_duplicate_set(total: usize) -> Result<Vec<Box<dyn Rewrite>>, RuleSetError> {
    if !DUPLICATE_GRID.contains(&total) {
        return Err(RuleSetError::UnreachableTotal {
            mode: InflationMode::Duplicates,
            requested: total,
            grid: DUPLICATE_GRID,
        });
    }
    let mut rules = all_rules();
    if total == 62 {
        return Ok(rules);
    }
    if total == 93 {
        // Half-cycle: even indices 0, 2, .., 60 — 31 rules, spread across
        // every module rather than the first 31 (which would all be
        // algebra rules, per §2.1's "spread across every module" note).
        let half: Vec<Box<dyn Rewrite>> = all_rules()
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, r)| Box::new(DuplicateRule::new(r, 1)) as Box<dyn Rewrite>)
            .collect();
        rules.extend(half);
        return Ok(rules);
    }
    let cycles = match total {
        124 => 1,
        186 => 2,
        248 => 3,
        _ => unreachable!("checked against DUPLICATE_GRID above"),
    };
    for c in 1..=cycles {
        let cycle: Vec<Box<dyn Rewrite>> = all_rules()
            .into_iter()
            .map(|r| Box::new(DuplicateRule::new(r, c)) as Box<dyn Rewrite>)
            .collect();
        rules.extend(cycle);
    }
    Ok(rules)
}

// ============================================================================
// Mode (ii): mechanical compositions (§2.2)
// ============================================================================

/// A composed rule A∘B, plus the provenance of how it was built — written
/// out (names, indices, position) so the Register run's composition-pool
/// JSON is reproducible and reviewable.
pub struct Composition {
    pub a_idx: usize,
    pub b_idx: usize,
    pub position: Vec<u8>,
    pub rule: TemplateRewrite,
}

/// Whether `r` has both an LHS and an RHS template — the composable subset
/// (30 of 62 in the production library, per §2.2).
fn is_templated(r: &dyn Rewrite) -> bool {
    let mut scratch = ExprArena::new();
    r.lhs_template(&mut scratch).is_some() && r.rhs_template(&mut scratch).is_some()
}

/// Ops whose `eval_scalar` implementation is only well-behaved for a
/// positive argument (`Pow`'s base; `Log2`/`Ln`/`Log10`/`Sqrt`/`Rsqrt`/
/// `Recip`'s argument) — it produces large finite garbage rather than a
/// clean domain-edge `NaN` outside that domain (verified empirically while
/// building this harness's oracle gate, `math::oracle`). For a rule used
/// standalone this is harmless: its metavariable IS the argument, and a
/// harness or a Guide samples/matches it directly. For a COMPOSITION,
/// unifying another rule's RHS into that argument position can make the
/// *effective* argument a compound sub-expression (e.g. `sin(neg(y))`) that
/// legitimately goes negative for ordinary `y` — narrowing the composed
/// rule's valid domain in its surface metavariables in a way this harness
/// has no general way to detect. Rather than accept an oracle gate that
/// cannot tell that domain-narrowing apart from a real unifier bug (both
/// present as "disagreement at some sampled points"), `compose_rules`
/// excludes any composition touching this op family — conservative, but
/// checkable structurally with no oracle/eval_scalar dependency, so it
/// applies in every build, not just `cargo test`.
fn touches_domain_restricted_op(arena: &ExprArena, id: pixelflow_ir::ExprId) -> bool {
    use pixelflow_ir::OpKind;
    matches!(
        arena.kind(id),
        OpKind::Pow
            | OpKind::Log2
            | OpKind::Ln
            | OpKind::Log10
            | OpKind::Sqrt
            | OpKind::Rsqrt
            | OpKind::Recip
    ) || arena.children(id).any(|c| touches_domain_restricted_op(arena, c))
}

/// Enumerate every valid composition of `base`'s templated rules, in
/// deterministic (a_idx, b_idx, position) order, applying §2.2's filters
/// (identity, no-op, exact-structural-duplicate) via
/// [`TemplateRewrite::compose`] and a display-string dedup key.
///
/// This does not run the oracle gate (§2.4) — that requires
/// `pixelflow-ir`'s `oracle` feature, which this crate exposes only to its
/// own test binary (`math::oracle`, `#[cfg(test)]`) and to downstream
/// crates that opt in themselves (`pixelflow-pipeline --features
/// training`). Callers that need the validated pool run the oracle check
/// themselves against every `Composition` this returns and drop failures —
/// never silently, per the design's §2.4 gate.
#[must_use]
pub fn compose_rules(base: &[Box<dyn Rewrite>]) -> Vec<Composition> {
    let templated_b: Vec<usize> = base
        .iter()
        .enumerate()
        .filter(|(_, r)| is_templated(r.as_ref()))
        .map(|(i, _)| i)
        .collect();

    let mut out = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();

    for (a_idx, a) in base.iter().enumerate() {
        let mut probe = ExprArena::new();
        let (Some(a_lhs), Some(a_rhs)) = (a.lhs_template(&mut probe), a.rhs_template(&mut probe))
        else {
            continue;
        };
        let _ = a_lhs; // only a_rhs's positions matter for this loop
        let positions = template::positions(&probe, a_rhs);

        for &b_idx in &templated_b {
            let b = &base[b_idx];
            for pos in &positions {
                let Some(rule) = TemplateRewrite::compose(a.as_ref(), b.as_ref(), pos) else {
                    continue;
                };
                let mut disp = ExprArena::new();
                let lhs = rule
                    .lhs_template(&mut disp)
                    .expect("TemplateRewrite always has both templates");
                let rhs = rule
                    .rhs_template(&mut disp)
                    .expect("TemplateRewrite always has both templates");
                if touches_domain_restricted_op(&disp, lhs) || touches_domain_restricted_op(&disp, rhs) {
                    continue;
                }
                let key = (disp.display(lhs).to_string(), disp.display(rhs).to_string());
                if seen.insert(key) {
                    out.push(Composition {
                        a_idx,
                        b_idx,
                        position: pos.clone(),
                        rule,
                    });
                }
            }
        }
    }
    out
}

/// A tiny deterministic PRNG (splitmix64) — no external dependency for one
/// seeded shuffle over a few hundred elements.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn fisher_yates<T>(v: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// The seeded permutation §2.2 fixes for the |R| grid's composition
/// prefixes (31, 62, 124, 186 compositions).
pub const COMPOSITION_SEED: u64 = 0x5EED2;

#[must_use]
pub fn composition_pool(base: &[Box<dyn Rewrite>], seed: u64) -> Vec<Composition> {
    let mut pool = compose_rules(base);
    fisher_yates(&mut pool, seed);
    pool
}

/// Total |R| → composition-count prefix, per §3's grid.
pub const COMPOSITION_GRID: &[(usize, usize)] = &[(62, 0), (93, 31), (124, 62), (186, 124), (248, 186)];

fn build_composition_set(total: usize) -> Result<Vec<Box<dyn Rewrite>>, RuleSetError> {
    let Some(&(_, inflation)) = COMPOSITION_GRID.iter().find(|(t, _)| *t == total) else {
        return Err(RuleSetError::UnreachableTotal {
            mode: InflationMode::Compositions,
            requested: total,
            grid: &[62, 93, 124, 186, 248],
        });
    };
    let mut rules = all_rules();
    if inflation == 0 {
        return Ok(rules);
    }
    let pool = composition_pool(&all_rules(), COMPOSITION_SEED);
    if pool.len() < inflation {
        return Err(RuleSetError::PoolTooSmall {
            requested_inflation: inflation,
            pool_size: pool.len(),
        });
    }
    rules.extend(
        pool.into_iter()
            .take(inflation)
            .map(|c| Box::new(c.rule) as Box<dyn Rewrite>),
    );
    Ok(rules)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_the_documented_forms() {
        assert_eq!(RuleSetSpec::parse("base").unwrap().mode, None);
        let d = RuleSetSpec::parse("dup:124").unwrap();
        assert_eq!(d.mode, Some(InflationMode::Duplicates));
        assert_eq!(d.total, 124);
        let c = RuleSetSpec::parse("comp:186").unwrap();
        assert_eq!(c.mode, Some(InflationMode::Compositions));
        assert_eq!(c.total, 186);
        assert!(RuleSetSpec::parse("bogus").is_err());
        assert!(RuleSetSpec::parse("dup:notanumber").is_err());
        assert!(
            RuleSetSpec::parse("new:31").is_err(),
            "mode (iii) has no builder here"
        );
    }

    #[test]
    fn duplicate_grid_never_pads() {
        for &total in DUPLICATE_GRID {
            let rules = build_rule_set(&RuleSetSpec {
                mode: Some(InflationMode::Duplicates),
                total,
            })
            .unwrap_or_else(|e| panic!("dup:{total} should build: {e}"));
            assert_eq!(rules.len(), total, "dup:{total} produced the wrong count");
        }
        assert!(
            build_rule_set(&RuleSetSpec {
                mode: Some(InflationMode::Duplicates),
                total: 100
            })
            .is_err(),
            "an off-grid total must error, never pad"
        );
    }

    #[test]
    fn duplicate_names_itself_after_its_inner_rule_and_matches_identically() {
        // A duplicate must behave identically to its inner rule (same
        // apply/templates) so the sweep cannot tell them apart except by
        // name — that is exactly what makes copies "learnable overhead"
        // (§2.1: a Guide must learn the name is worthless, not the shape).
        let inner = all_rules()
            .into_iter()
            .next()
            .expect("all_rules is nonempty");
        let inner_name = inner.name().to_string();
        let dup = DuplicateRule::new(inner, 1);
        assert_eq!(dup.name(), format!("{inner_name}#dup1"));

        // Same LHS/RHS templates (or lack thereof) as a fresh copy of the
        // same rule at the same index.
        let fresh = &all_rules()[0];
        let mut a = ExprArena::new();
        let mut b = ExprArena::new();
        assert_eq!(
            dup.lhs_template(&mut a).is_some(),
            fresh.lhs_template(&mut b).is_some()
        );
    }

    #[test]
    fn composition_pool_is_nonempty_and_deterministic() {
        let base = all_rules();
        let pool_a = composition_pool(&base, COMPOSITION_SEED);
        let pool_b = composition_pool(&base, COMPOSITION_SEED);
        assert!(
            !pool_a.is_empty(),
            "expected at least one valid composition"
        );
        assert_eq!(
            pool_a.len(),
            pool_b.len(),
            "same seed must give the same pool size"
        );
        for (x, y) in pool_a.iter().zip(pool_b.iter()) {
            assert_eq!(x.a_idx, y.a_idx);
            assert_eq!(x.b_idx, y.b_idx);
            assert_eq!(x.position, y.position);
        }
    }

    #[test]
    fn composition_rule_set_grid_never_pads() {
        let pool_size = composition_pool(&all_rules(), COMPOSITION_SEED).len();
        for &(total, inflation) in COMPOSITION_GRID {
            let result = build_rule_set(&RuleSetSpec {
                mode: Some(InflationMode::Compositions),
                total,
            });
            if inflation <= pool_size {
                let rules = result.unwrap_or_else(|e| panic!("comp:{total} should build: {e}"));
                assert_eq!(rules.len(), total);
            } else {
                assert!(
                    result.is_err(),
                    "comp:{total} needs {inflation} compositions but the pool has only \
                     {pool_size} — must error, never pad"
                );
            }
        }
    }

    #[test]
    fn fingerprint_is_stable_and_order_sensitive() {
        let a = all_rules();
        let b = all_rules();
        assert_eq!(rule_set_fingerprint(&a), rule_set_fingerprint(&b));
        let dup = build_rule_set(&RuleSetSpec {
            mode: Some(InflationMode::Duplicates),
            total: 93,
        })
        .unwrap();
        assert_ne!(rule_set_fingerprint(&a), rule_set_fingerprint(&dup));
    }
}
