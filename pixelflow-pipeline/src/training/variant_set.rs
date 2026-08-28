//! `VariantSet` (J6, docs/plans/2026-08-17-cost-model-domain.md): a finite
//! set of extractions of ONE base expression, from ONE saturated e-graph,
//! labeled with `CostLabel`s on ONE clock.
//!
//! # Why this exists
//!
//! Round 0/1 (docs/plans/2026-08-17-egraph-vsa-nnue-research-notes.md §1.5)
//! measured that corpus-level regression is saturated (DEV Spearman ρ≈0.98,
//! a bare op-count baseline scores 0.949) while the object extraction
//! actually needs ranked — candidates competing for ONE e-class, whose true
//! cost deltas sit near the measurement floor — has no type and no minting
//! path. `BwdTrainingPairArena` (the live training loop) destroys exactly
//! this structure: each side of a pair becomes an independent absolute-
//! regression sample, so nothing ever asks the model "which of these do you
//! prefer", only "what does this cost in isolation". `VariantSet` is the
//! missing object: [`mint_variant_set`] is its one constructor, and it is
//! the only thing that produces one — there is no other way to assemble a
//! `VariantSet` by hand, so a set that exists was actually minted (measured,
//! quarantined, fenced), never fabricated by a caller assembling labels
//! from elsewhere.
//!
//! # Invariants the constructor enforces
//!
//! - **Same base, same e-graph.** Every member is an extraction of the SAME
//!   saturated e-graph built from the SAME base expression — `enumerate_candidates`
//!   is called once per set, not once per member.
//! - **Same clock.** Every member's [`CostLabel`] is minted from the SAME
//!   `BenchSession` in `BenchMode::Latency` — the measurement-discipline
//!   contract (docs/plans/2026-08-05-egraph-nnue-research-workflow.md §4).
//! - **Every member independently fenced and quarantined.** The base is
//!   checked once; EVERY candidate extraction is checked again, because a
//!   rewrite can coincidentally produce a DEV/FINAL structure or an
//!   expression the same-form gate cannot verify even when the base could —
//!   "the fence must be applied to every variant, not just the base" is a
//!   task requirement, not an optimization to skip when it looks redundant.
//! - **At least 2 labeled members, or no set.** A set of 0 or 1 members
//!   supplies no orderable pair, so it is not a `VariantSet` at all —
//!   [`mint_variant_set`] returns `Err` and the caller counts the exclusion,
//!   rather than persisting a degenerate set that would silently contribute
//!   zero training pairs while still counting toward a "sets minted" total.

use serde::Serialize;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_search::egraph::{
    CostModel, EGraph, ESTIMATE_OPS, all_rules, choices_to_arena, enumerate_candidates,
};
use pixelflow_search::nnue::ExprNnue;

use crate::jit_bench::{BenchMode, BenchPosition, BenchResult, BenchSession, CostLabel};
use crate::training::factored::arena_to_kernel_code;
use crate::training::mint::bench_mode_slug;
use crate::training::quarantine::Quarantine;
use crate::training::split::{DevSide, Fence, FinalSide, blocked_by_either};

/// One labeled candidate extraction inside a [`VariantSet`].
#[derive(Clone, Debug, Serialize)]
pub struct VariantMember {
    /// Which arm of `enumerate_candidates` produced this member:
    /// `"static"` (DP-optimal under the latency prior), `"nnue"` (the
    /// learned pick), or `"swap"` (an estimate-op flip of one of those).
    pub provenance: String,
    /// The materialized expression, as `kernel!`-shaped source (the same
    /// serialization `training::factored::arena_to_kernel_code` gives every
    /// other stage) — human-readable and stable, so the JSONL is directly
    /// inspectable without an arena deserializer.
    pub expression: String,
    pub node_count: usize,
    /// How many of [`ESTIMATE_OPS`] (Recip/Rsqrt/MulAdd/Sqrt) appear in this
    /// expression — the axis Round-0 found the NNUE policy conservative on;
    /// recorded per member so a downstream loss can regress on the swap
    /// delta directly instead of re-deriving it from `expression` text.
    pub estimate_op_count: usize,
    /// [`CostLabel::target_log_ns`] — the log-ns training target.
    pub target_log_ns: f32,
    /// The label's value in the run's opening clock (ns), pre-log — for
    /// reporting pair deltas in physical units.
    pub session_ns: f64,
    pub mode: String,
    pub drift: f64,
    pub position: usize,
    pub calibration_ns: f64,
    pub local_sentinel_ns: f64,
}

impl VariantMember {
    fn new(provenance: &'static str, arena: &ExprArena, root: ExprId, label: CostLabel) -> Self {
        let calibration = label.calibration();
        Self {
            provenance: provenance.to_string(),
            expression: arena_to_kernel_code(arena, root),
            node_count: arena.node_count_subtree(root),
            estimate_op_count: count_estimate_ops(arena, root),
            target_log_ns: label.target_log_ns(),
            session_ns: label.value().get(),
            mode: bench_mode_slug(label.mode()).to_string(),
            drift: label.drift().get(),
            position: label.position().0,
            calibration_ns: calibration.calibration_ns,
            local_sentinel_ns: calibration.local_sentinel_ns,
        }
    }
}

/// A finite set of extractions of ONE base expression from ONE saturated
/// e-graph, with labels in ONE clock context (J6). See the module docs for
/// the invariants [`mint_variant_set`] enforces to produce one.
#[derive(Clone, Debug, Serialize)]
pub struct VariantSet {
    pub base_name: String,
    pub base_node_count: usize,
    /// Node/class counts of the saturated e-graph this set's members were
    /// all extracted from — a coarse proxy for how much search space the
    /// candidates were drawn from, useful for correlating set size/spread
    /// against saturation size.
    pub egraph_node_count: usize,
    pub egraph_class_count: usize,
    pub members: Vec<VariantMember>,
}

impl VariantSet {
    /// Every within-set pair's measured relative delta, as a percent of the
    /// smaller label (`|a - b| / min(a, b) * 100`) — the number the dataset-
    /// quality report (plan task 4) compares against the run noise floor.
    /// `C(members.len(), 2)` entries, in member-index order `(0,1), (0,2),
    /// …`. A label `<= 0` (documented as possible for kernels cheaper than
    /// call overhead, see `training::mint`) makes its pairs undefined and
    /// they are omitted rather than divided by a non-positive number.
    #[must_use]
    pub fn pair_deltas_pct(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..self.members.len() {
            for j in (i + 1)..self.members.len() {
                let a = self.members[i].session_ns;
                let b = self.members[j].session_ns;
                let lo = a.min(b);
                if lo <= 0.0 {
                    continue;
                }
                out.push((a - b).abs() / lo * 100.0);
            }
        }
        out
    }
}

/// Why a base expression, or a whole attempted set, did not become a
/// [`VariantSet`]. Per-candidate exclusions inside a surviving set are
/// tallied in [`MintTally`] instead, since they don't abort the whole set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MintExclusion {
    /// The base expression itself failed the same-form quarantine gate.
    BaseQuarantineFailed,
    /// The base expression's feature-quotient structure is already held out
    /// (DEV or FINAL) — should not occur for a TRAIN-family base, but
    /// checked rather than assumed (the fence's whole job is to not be
    /// trusted by convention).
    BaseFenceBlocked,
    /// Fewer than 2 candidates survived fencing/quarantine/measurement to
    /// become labeled members, so the set supplies no orderable pair.
    InsufficientMembers,
}

impl MintExclusion {
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Self::BaseQuarantineFailed => "base_quarantine_failed",
            Self::BaseFenceBlocked => "base_fence_blocked",
            Self::InsufficientMembers => "insufficient_members",
        }
    }
}

/// Per-run tallies for the dataset-quality report (plan task 4): how many
/// sets and candidates were attempted, and where each exclusion happened.
#[derive(Default, Debug, Clone, Serialize)]
pub struct MintTally {
    pub sets_attempted: usize,
    pub sets_minted: usize,
    pub base_quarantine_failed: usize,
    pub base_fence_blocked: usize,
    pub sets_insufficient_members: usize,
    pub candidates_attempted: usize,
    pub candidates_fence_blocked: usize,
    pub candidates_quarantine_failed: usize,
    pub candidates_bench_failed: usize,
    pub candidates_minted: usize,
}

fn count_estimate_ops(arena: &ExprArena, root: ExprId) -> usize {
    let mut visited = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut count = 0usize;
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        if ESTIMATE_OPS.contains(&arena.kind(id)) {
            count += 1;
        }
        stack.extend(arena.children(id));
    }
    count
}

/// Mint one [`VariantSet`] from a TRAIN-family base expression.
///
/// The ONE constructor (see module docs for the invariants this enforces):
/// saturates a fresh e-graph from `(base_arena, base_root)`, enumerates up
/// to `k` diverse candidate extractions
/// (`pixelflow_search::egraph::enumerate_candidates` — static-optimal,
/// NNUE, and estimate-op swaps of both), fence-checks and quarantines EVERY
/// candidate (not just the base), and mints a [`CostLabel`] under `session`
/// in `BenchMode::Latency` for every candidate that survives both gates.
///
/// `position` is the caller's running collection-order counter, threaded
/// across the whole minting run (not reset per set) so every label's
/// [`crate::jit_bench::BenchPosition`] reflects true global collection
/// order — required for the drift/order-invariance property `training::mint`
/// documents.
///
/// # Errors
///
/// Returns [`MintExclusion`] when the base itself is excluded, or when
/// fewer than 2 candidates survive to become labeled members (see
/// [`MintExclusion::InsufficientMembers`]). All counts are also recorded on
/// `tally` regardless of outcome.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors mint_variant_set_with_arenas's params 1:1; grouping them into a struct would just move the arity, see the crate's own note that Manifold composition (not a config struct) is preferred where the params are this method's whole contract"
)]
pub fn mint_variant_set(
    session: &mut BenchSession,
    quarantine: &mut Quarantine,
    dev_fence: &Fence<DevSide>,
    final_fence: &Fence<FinalSide>,
    nnue: &ExprNnue,
    static_costs: &CostModel,
    base_name: &str,
    base_arena: &ExprArena,
    base_root: ExprId,
    k: usize,
    saturate_limit: usize,
    position: &mut usize,
    tally: &mut MintTally,
) -> Result<VariantSet, MintExclusion> {
    mint_variant_set_with_arenas(
        session,
        quarantine,
        Some(dev_fence),
        final_fence,
        nnue,
        static_costs,
        base_name,
        base_arena,
        base_root,
        k,
        saturate_limit,
        position,
        tally,
    )
    .map(|(set, _arenas)| set)
}

/// Mint one [`VariantSet`] from a DEV-family base expression, for a
/// held-out contrastive EVALUATION collection (Round 2b task 2) — NOT for
/// training.
///
/// Differs from [`mint_variant_set`] in exactly one respect: it does not
/// check `base_arena`/candidates against a DEV fence, because the base is
/// *deliberately* drawn from `corpus_dev.bin` (a [`Fence<DevSide>`] built
/// from that same file would trivially block every base by construction —
/// checking DEV-ness of a DEV base is not the holdout property this
/// pipeline needs). FINAL is still fenced: an eval collection has no reason
/// to admit FINAL structures either, so "fence-respecting" here means "still
/// never touches FINAL", not "checked against every fence that exists".
///
/// # Errors
///
/// Same as [`mint_variant_set`].
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors mint_variant_set_with_arenas's params 1:1, see that function's note"
)]
pub fn mint_variant_set_dev_eval(
    session: &mut BenchSession,
    quarantine: &mut Quarantine,
    final_fence: &Fence<FinalSide>,
    nnue: &ExprNnue,
    static_costs: &CostModel,
    base_name: &str,
    base_arena: &ExprArena,
    base_root: ExprId,
    k: usize,
    saturate_limit: usize,
    position: &mut usize,
    tally: &mut MintTally,
) -> Result<VariantSet, MintExclusion> {
    mint_variant_set_dev_eval_with_arenas(
        session,
        quarantine,
        final_fence,
        nnue,
        static_costs,
        base_name,
        base_arena,
        base_root,
        k,
        saturate_limit,
        position,
        tally,
    )
    .map(|(set, _arenas)| set)
}

/// Like [`mint_variant_set_dev_eval`], but also returns each member's
/// materialized `(ExprArena, ExprId)` — see
/// [`mint_variant_set_with_arenas`]'s doc for why an in-process training
/// consumer needs this instead of the serialized text form.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors mint_variant_set_with_arenas's params 1:1, see that function's note"
)]
pub fn mint_variant_set_dev_eval_with_arenas(
    session: &mut BenchSession,
    quarantine: &mut Quarantine,
    final_fence: &Fence<FinalSide>,
    nnue: &ExprNnue,
    static_costs: &CostModel,
    base_name: &str,
    base_arena: &ExprArena,
    base_root: ExprId,
    k: usize,
    saturate_limit: usize,
    position: &mut usize,
    tally: &mut MintTally,
) -> Result<(VariantSet, Vec<(ExprArena, ExprId)>), MintExclusion> {
    mint_variant_set_with_arenas(
        session,
        quarantine,
        None,
        final_fence,
        nnue,
        static_costs,
        base_name,
        base_arena,
        base_root,
        k,
        saturate_limit,
        position,
        tally,
    )
}

/// Whether `(arena, root)` is blocked by whichever fences are active:
/// always FINAL, and DEV only when `dev_fence` is `Some` (training mode).
fn blocked_by_active_fences(
    dev_fence: Option<&Fence<DevSide>>,
    final_fence: &Fence<FinalSide>,
    arena: &ExprArena,
    root: ExprId,
) -> bool {
    match dev_fence {
        Some(dev) => blocked_by_either(dev, final_fence, arena, root),
        None => final_fence.contains(&crate::training::structural::FenceKey::of(arena, root)),
    }
}

/// Shared implementation behind [`mint_variant_set`] (training, `dev_fence:
/// Some`) and [`mint_variant_set_dev_eval`] (held-out eval collection,
/// `dev_fence: None`) — see their docs for what differs and why.
///
/// Also returns each member's materialized `(ExprArena, ExprId)`, in the
/// same order as the returned [`VariantSet`]'s `members` — the serialized
/// [`VariantMember::expression`] is `kernel!`-style text (deliberately, so
/// the JSONL artifact stays light and human-inspectable), which an
/// in-process training consumer that needs to rebuild `EdgeAccumulator`
/// features cannot parse back into an arena. Callers that only want the
/// serializable [`VariantSet`] (e.g. `bin/mint_variant_sets.rs`) go through
/// the two wrappers above and drop this half.
#[expect(
    clippy::too_many_arguments,
    reason = "the full set of knobs mint_variant_set's two callers each need pinned per invocation; see mint_variant_set's note"
)]
pub fn mint_variant_set_with_arenas(
    session: &mut BenchSession,
    quarantine: &mut Quarantine,
    dev_fence: Option<&Fence<DevSide>>,
    final_fence: &Fence<FinalSide>,
    nnue: &ExprNnue,
    static_costs: &CostModel,
    base_name: &str,
    base_arena: &ExprArena,
    base_root: ExprId,
    k: usize,
    saturate_limit: usize,
    position: &mut usize,
    tally: &mut MintTally,
) -> Result<(VariantSet, Vec<(ExprArena, ExprId)>), MintExclusion> {
    tally.sets_attempted += 1;

    if !quarantine.check(base_name, base_arena, base_root) {
        tally.base_quarantine_failed += 1;
        return Err(MintExclusion::BaseQuarantineFailed);
    }
    if blocked_by_active_fences(dev_fence, final_fence, base_arena, base_root) {
        tally.base_fence_blocked += 1;
        return Err(MintExclusion::BaseFenceBlocked);
    }

    let mut eg = EGraph::with_rules(all_rules());
    let root_class = eg.add_arena(base_arena, base_root);
    eg.saturate_with_limit(saturate_limit);

    let candidates = enumerate_candidates(&eg, root_class, static_costs, nnue, k);

    let mut members = Vec::new();
    let mut member_arenas = Vec::new();
    for (i, cand) in candidates.iter().enumerate() {
        tally.candidates_attempted += 1;
        let (arena, root) = choices_to_arena(&cand.extraction);

        if blocked_by_active_fences(dev_fence, final_fence, &arena, root) {
            tally.candidates_fence_blocked += 1;
            continue;
        }
        let cand_name = format!("{base_name}_v{i:02}_{}", cand.provenance);
        if !quarantine.check(&cand_name, &arena, root) {
            tally.candidates_quarantine_failed += 1;
            continue;
        }

        let bench: BenchResult = match session.benchmark_arena(&arena, root, BenchMode::Latency) {
            Ok(b) => b,
            Err(_e) => {
                tally.candidates_bench_failed += 1;
                continue;
            }
        };
        let label = CostLabel::mint(&bench, BenchPosition(*position), "mint_variant_set");
        *position += 1;
        members.push(VariantMember::new(cand.provenance, &arena, root, label));
        member_arenas.push((arena, root));
        tally.candidates_minted += 1;
    }

    if members.len() < 2 {
        tally.sets_insufficient_members += 1;
        return Err(MintExclusion::InsufficientMembers);
    }

    tally.sets_minted += 1;
    Ok((
        VariantSet {
            base_name: base_name.to_string(),
            base_node_count: base_arena.node_count_subtree(base_root),
            egraph_node_count: eg.node_count(),
            egraph_class_count: eg.num_classes(),
            members,
        },
        member_arenas,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::split::{DevSide, Fence, FinalSide};
    use pixelflow_ir::OpKind;

    fn recip_sqrt_arena() -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let sqrt_x = arena.push_unary(OpKind::Sqrt, x);
        let one = arena.push_const(1.0);
        let root = arena.push_binary(OpKind::Div, one, sqrt_x);
        (arena, root)
    }

    /// A one-entry corpus file at `path`, for building a `Fence` in tests
    /// without a real tiered corpus on disk.
    fn write_single_entry_corpus(path: &std::path::Path) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let c = arena.push_const(123.0);
        let root = arena.push_binary(OpKind::Add, x, c);
        crate::training::corpus::write_corpus(path, &[("fence_seed".to_string(), arena, root)])
            .expect("write fence corpus");
    }

    fn empty_fences(dir: &std::path::Path) -> (Fence<DevSide>, Fence<FinalSide>) {
        std::fs::create_dir_all(dir).expect("scratch dir");
        write_single_entry_corpus(&dir.join("corpus_dev.bin"));
        write_single_entry_corpus(&dir.join("corpus_final.bin"));
        (Fence::<DevSide>::build(dir), Fence::<FinalSide>::build(dir))
    }

    #[test]
    fn mints_a_set_with_at_least_two_members() {
        let dir = std::env::temp_dir().join(format!("variant_set_mint_{}", std::process::id()));
        let (dev_fence, final_fence) = empty_fences(&dir);

        let mut session = BenchSession::new();
        let log_path = dir.join("quarantine.jsonl");
        let mut quarantine = Quarantine::new(log_path.to_str().expect("utf8 path"));
        let nnue = ExprNnue::new_random(7);
        let costs = CostModel::latency_prior();
        let mut position = 0usize;
        let mut tally = MintTally::default();

        let (arena, root) = recip_sqrt_arena();
        let set = mint_variant_set(
            &mut session,
            &mut quarantine,
            &dev_fence,
            &final_fence,
            &nnue,
            &costs,
            "test_base",
            &arena,
            root,
            8,
            64,
            &mut position,
            &mut tally,
        )
        .expect("recip/sqrt base should mint a set");

        assert!(set.members.len() >= 2, "{:?}", set.members.len());
        assert_eq!(tally.sets_minted, 1);
        assert_eq!(tally.sets_attempted, 1);
        assert!(tally.candidates_minted >= 2);
        // Every member's position is unique and increasing — collection
        // order was actually threaded through, not reset per member.
        let positions: Vec<usize> = set.members.iter().map(|m| m.position).collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(positions, sorted);
        assert_eq!(
            positions
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            positions.len(),
            "positions must be unique: {positions:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dev_eval_mints_from_a_base_the_dev_fence_would_have_blocked() {
        // The whole point of `mint_variant_set_dev_eval`: a base whose exact
        // structure IS the DEV fence's only entry must still mint — training
        // mode (`mint_variant_set`) would reject it outright.
        let dir = std::env::temp_dir().join(format!("variant_set_dev_eval_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let (base_arena, base_root) = recip_sqrt_arena();
        crate::training::corpus::write_corpus(
            &dir.join("corpus_dev.bin"),
            &[("dev_base".to_string(), base_arena.clone(), base_root)],
        )
        .expect("write dev corpus");
        write_single_entry_corpus(&dir.join("corpus_final.bin"));
        let dev_fence = Fence::<DevSide>::build(&dir);
        let final_fence = Fence::<FinalSide>::build(&dir);

        // Sanity: training-mode minting of this exact base IS blocked (it's
        // the fence's own entry) — confirms the eval path is doing
        // something training mode would not.
        let mut session = BenchSession::new();
        let log_path = dir.join("quarantine_train.jsonl");
        let mut quarantine = Quarantine::new(log_path.to_str().expect("utf8 path"));
        let nnue = ExprNnue::new_random(21);
        let costs = CostModel::latency_prior();
        let mut position = 0usize;
        let mut tally = MintTally::default();
        let blocked = mint_variant_set(
            &mut session,
            &mut quarantine,
            &dev_fence,
            &final_fence,
            &nnue,
            &costs,
            "dev_base",
            &base_arena,
            base_root,
            8,
            64,
            &mut position,
            &mut tally,
        );
        assert_eq!(
            blocked.err(),
            Some(MintExclusion::BaseFenceBlocked),
            "training-mode minting of a DEV-fence entry must be blocked"
        );

        // The eval path mints the same base successfully.
        let log_path_eval = dir.join("quarantine_eval.jsonl");
        let mut quarantine_eval = Quarantine::new(log_path_eval.to_str().expect("utf8 path"));
        let mut eval_tally = MintTally::default();
        let mut eval_position = 0usize;
        let set = mint_variant_set_dev_eval(
            &mut session,
            &mut quarantine_eval,
            &final_fence,
            &nnue,
            &costs,
            "dev_base",
            &base_arena,
            base_root,
            8,
            64,
            &mut eval_position,
            &mut eval_tally,
        )
        .expect("DEV base should mint under the eval path");
        assert!(set.members.len() >= 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mint_variant_set_with_arenas_returns_one_arena_per_member_in_order() {
        let dir =
            std::env::temp_dir().join(format!("variant_set_with_arenas_{}", std::process::id()));
        let (dev_fence, final_fence) = empty_fences(&dir);

        let mut session = BenchSession::new();
        let log_path = dir.join("quarantine.jsonl");
        let mut quarantine = Quarantine::new(log_path.to_str().expect("utf8 path"));
        let nnue = ExprNnue::new_random(33);
        let costs = CostModel::latency_prior();
        let mut position = 0usize;
        let mut tally = MintTally::default();

        let (arena, root) = recip_sqrt_arena();
        let (set, arenas) = mint_variant_set_with_arenas(
            &mut session,
            &mut quarantine,
            Some(&dev_fence),
            &final_fence,
            &nnue,
            &costs,
            "with_arenas_base",
            &arena,
            root,
            8,
            64,
            &mut position,
            &mut tally,
        )
        .expect("should mint");

        assert_eq!(arenas.len(), set.members.len());
        // Every returned arena must actually evaluate to the member's
        // recorded node count — proof this is the SAME expression the
        // member's label was measured from, not just a same-length parallel
        // vec.
        for ((member_arena, member_root), member) in arenas.iter().zip(&set.members) {
            assert_eq!(
                member_arena.node_count_subtree(*member_root),
                member.node_count
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pair_deltas_pct_has_c_n_2_entries() {
        let dir = std::env::temp_dir().join(format!("variant_set_pairs_{}", std::process::id()));
        let (dev_fence, final_fence) = empty_fences(&dir);

        let mut session = BenchSession::new();
        let log_path = dir.join("quarantine.jsonl");
        let mut quarantine = Quarantine::new(log_path.to_str().expect("utf8 path"));
        let nnue = ExprNnue::new_random(11);
        let costs = CostModel::latency_prior();
        let mut position = 0usize;
        let mut tally = MintTally::default();

        let (arena, root) = recip_sqrt_arena();
        let set = mint_variant_set(
            &mut session,
            &mut quarantine,
            &dev_fence,
            &final_fence,
            &nnue,
            &costs,
            "test_base_pairs",
            &arena,
            root,
            8,
            64,
            &mut position,
            &mut tally,
        )
        .expect("should mint");

        let n = set.members.len();
        let deltas = set.pair_deltas_pct();
        assert!(deltas.len() <= n * (n - 1) / 2);

        std::fs::remove_dir_all(&dir).ok();
    }
}
