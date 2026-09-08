//! Which schedule entries a `Select`'s short-circuit branch may skip, and
//! the ordering that lets one branch span them.
//!
//! A property of the **schedule**, not of allocation and not of emission: it
//! asks only which values each arm of a `Select` computes for itself, and the
//! answer is the same whatever registers those values end up in. Both sides
//! read it — the emitter to place the branches, the allocator to keep a split
//! live range from naming a register a skipped arm never loaded — so it lives
//! beside them rather than inside either.
//!
//! Exclusivity is necessary and not sufficient: a branch skips a *range*, so
//! an arm is only guardable when the values it owns are one contiguous run.
//! An arm can own two hundred values and be refused because forty entries
//! belonging to some other expression happen to sit between its first and its
//! last. [`cluster_select_arms`] is the answer to exactly that case, and only
//! that case — it stable-partitions the region between the mask and the
//! select into shared, then true-exclusive, then false-exclusive values. That
//! is always a legal topological order, because a shared value can never
//! depend on an arm-exclusive one (if it did, the exclusivity filter would
//! have rejected the value: it has a consumer outside the arm).
//!
//! It runs only where it buys a branch. Moving every shared value ahead of
//! both arms stretches live ranges across the skipped arm, and that pressure
//! should be paid where a guard is bought and nowhere else.
//!
//! And a guard is only bought where it can pay: see
//! [`MISPREDICT_PENALTY_CYCLES`]. Whether a mask is *coherent* — uniform
//! across a batch often enough for the branch to fire and to predict — is a
//! property of the data, which no static analysis can know. Its worst case,
//! though, is known exactly, and bounding the downside by the upside is
//! enough to keep the analysis honest without a tuned number anywhere.

use alloc::vec::Vec;

use pixelflow_ir::kind::OpKind;

use super::ScheduledOp;
use super::demand::{self, Demand, Literal};
use super::regalloc::{Def, ValueId};

/// Describes a Select node's short-circuit structure in the schedule.
///
/// For `Select(mask, if_true, if_false)`, identifies contiguous ranges of
/// schedule entries that are exclusive to each arm (not shared with mask
/// or the other arm). These ranges can be guarded by conditional branches.
#[derive(Debug, Clone)]
pub(crate) struct SelectGuard {
    /// Schedule index of the Select node itself.
    pub(crate) select_idx: usize,
    /// ValueId of the mask operand (already computed before arms).
    pub(crate) mask_vid: ValueId,
    /// Range of schedule indices exclusive to the true arm: [true_start, true_end).
    /// Empty if true_start == true_end.
    pub(crate) true_range: (usize, usize),
    /// Range of schedule indices exclusive to the false arm: [false_start, false_end).
    pub(crate) false_range: (usize, usize),
}

/// Compute the transitive dependencies of a ValueId in the schedule.
///
/// `schedule_ops` is a dense Vec indexed by `ValueId.0`, pre-built by the
/// caller so each lookup is O(1) instead of O(n).
fn transitive_deps(
    vid: ValueId,
    schedule_ops: &[Option<ScheduledOp>],
) -> alloc::collections::BTreeSet<ValueId> {
    use alloc::collections::BTreeSet;

    let mut deps = BTreeSet::new();
    let mut worklist = alloc::vec![vid];
    while let Some(v) = worklist.pop() {
        if !deps.insert(v) {
            continue;
        }
        // O(1) lookup via dense Vec indexed by ValueId.0
        if let Some(Some(sop)) = schedule_ops.get(v.0 as usize) {
            match sop {
                ScheduledOp::Var(_) | ScheduledOp::Const(_) | ScheduledOp::Uniform(_) => {}
                ScheduledOp::Unary(_, c)
                | ScheduledOp::ShiftImm(_, c, _)
                | ScheduledOp::Gather(c, _) => {
                    worklist.push(*c);
                }
                ScheduledOp::Binary(_, l, r) => {
                    worklist.push(*l);
                    worklist.push(*r);
                }
                ScheduledOp::Ternary(_, a, b, c) => {
                    worklist.push(*a);
                    worklist.push(*b);
                    worklist.push(*c);
                }
            }
        }
    }
    deps
}
/// One `Select`'s arms as schedule positions: the entries each arm computes
/// for itself and nothing else.
///
/// Exclusivity only — whether a branch can actually span an arm is
/// [`SelectArms::range`], which is where the *order* gets its say.
struct SelectArms {
    select_idx: usize,
    select_vid: ValueId,
    mask_vid: ValueId,
    /// Where the mask lands, or `usize::MAX` when it is not in this scope's
    /// schedule (a live-in from an enclosing one).
    mask_idx: usize,
    true_indices: alloc::collections::BTreeSet<usize>,
    false_indices: alloc::collections::BTreeSet<usize>,
    /// Everything the select reads, transitively, as schedule positions —
    /// which is also, by complement, everything between the mask and the
    /// select that the select does NOT need.
    cone: alloc::collections::BTreeSet<usize>,
    /// What each arm's own entries cost, in latency-prior cycles — what a
    /// guard on that arm could save, against what the branch costs when it
    /// does not.
    true_cycles: usize,
    false_cycles: usize,
}

impl SelectArms {
    /// The half-open range a branch may skip for `indices`, or an empty range
    /// at the select when it may not.
    ///
    /// The branch skips the WHOLE range when the mask is uniform, so every
    /// index in it must belong to this arm; and the uniformity test reads the
    /// mask's register at the range's start, so the mask must be computed by
    /// then. (Schedules from the macro pipeline emit the mask before both
    /// arms, but arena-composed kernels may schedule an arm BEFORE it —
    /// guarding that would branch on an uninitialized register. The select
    /// still evaluates correctly through the unconditional blend.)
    fn range(
        &self,
        indices: &alloc::collections::BTreeSet<usize>,
        cycles: usize,
    ) -> (usize, usize) {
        if cycles <= MISPREDICT_PENALTY_CYCLES {
            return (self.select_idx, self.select_idx);
        }
        let (Some(&start), Some(&last)) = (indices.iter().next(), indices.iter().next_back())
        else {
            return (self.select_idx, self.select_idx);
        };
        let end = last + 1;
        let one_run = (start..end).all(|idx| indices.contains(&idx));
        if one_run && self.mask_idx < start {
            (start, end)
        } else {
            (self.select_idx, self.select_idx)
        }
    }

    fn true_range(&self) -> (usize, usize) {
        self.range(&self.true_indices, self.true_cycles)
    }

    fn false_range(&self) -> (usize, usize) {
        self.range(&self.false_indices, self.false_cycles)
    }

    /// An arm the ORDER refuses: it is worth guarding and no branch can span
    /// it. Distinct from an arm that owns nothing, and from one too cheap to
    /// guard — no reordering helps either of those.
    fn refused_for_order(&self) -> bool {
        let refused = |cycles: usize, range: (usize, usize)| {
            cycles > MISPREDICT_PENALTY_CYCLES && range.0 == range.1
        };
        refused(self.true_cycles, self.true_range())
            || refused(self.false_cycles, self.false_range())
    }
}

/// Analyze the schedule for Select nodes and compute short-circuit guard ranges.
///
/// For each Select, partitions schedule entries into:
/// - Shared: needed by mask, or by both arms (must always execute)
/// - True-exclusive: only needed by the true arm (skip if mask all-false)
/// - False-exclusive: only needed by the false arm (skip if mask all-true)
///
/// Returns guards sorted by select_idx (ascending).
pub(crate) fn analyze_select_guards(schedule: &[Def]) -> Vec<SelectGuard> {
    let arms = select_arms(schedule);
    let mut telemetry = Telemetry::new();
    let mut guards = Vec::new();

    // One backward pass over the whole DAG, and only when someone is
    // reading: demand is what an arm *may skip*, which is a weaker
    // condition than what the partition may *move*, so it is recorded
    // beside `exclusive` rather than replacing it. See `SelectStat`.
    let demand = telemetry
        .is_on()
        .then(|| {
            let root = schedule.last()?.value;
            Some(demand::demand_of(schedule, root))
        })
        .flatten();

    for select in &arms {
        let (true_range, false_range) = (select.true_range(), select.false_range());
        let demand_exclusive = demand.as_ref().map_or((0, 0), |d| {
            let observed = |v: ValueId| d.get(&v).cloned().unwrap_or_default();
            let arm = |lit: Literal| observed(select.select_vid).and_literal(lit);
            let count = |pred: &Demand| {
                schedule
                    .iter()
                    .filter(|def| {
                        let seen = observed(def.value);
                        !seen.is_never() && seen.implies(pred)
                    })
                    .count()
            };
            (
                count(&arm(Literal::set(select.mask_vid))),
                count(&arm(Literal::clear(select.mask_vid))),
            )
        });
        telemetry.select(|| SelectStat {
            demand_exclusive,
            select_idx: select.select_idx,
            mask_idx: select.mask_idx,
            exclusive: (select.true_indices.len(), select.false_indices.len()),
            guarded: (true_range.1 - true_range.0, false_range.1 - false_range.0),
            intruders: (
                intruders(&select.true_indices, schedule),
                intruders(&select.false_indices, schedule),
            ),
        });

        // Only create a guard if at least one arm has exclusive nodes
        if true_range.0 != true_range.1 || false_range.0 != false_range.1 {
            guards.push(SelectGuard {
                select_idx: select.select_idx,
                mask_vid: select.mask_vid,
                true_range,
                false_range,
            });
        }
    }

    telemetry.report(schedule.len());
    guards
}

/// Every `Select` in the schedule, with the entries exclusive to each arm.
fn select_arms(schedule: &[Def]) -> Vec<SelectArms> {
    use alloc::collections::BTreeSet;

    let mut arms = Vec::new();

    if schedule.is_empty() {
        return arms;
    }

    // The extraction cost model's table, which is the workspace's one answer
    // to "what does this op cost" — the guard's bound is denominated in the
    // same cycles the optimizer chose the expression with.
    let cycles = pixelflow_search::egraph::CostModel::latency_prior();

    // Build dense lookup: schedule_ops[vid.0] = Some(&ScheduledOp) for O(1) child traversal.
    // ValueIds are sequential starting from 0 (guaranteed by arena_to_schedule).
    let max_vid = schedule.iter().map(|def| def.value.0).max().unwrap_or(0) as usize;
    let mut schedule_ops: alloc::vec::Vec<Option<ScheduledOp>> = alloc::vec![None; max_vid + 1];
    for def in schedule {
        schedule_ops[def.value.0 as usize] = Some(def.op.clone());
    }

    // Build dense lookup: vid_to_sched_idx[vid.0] = schedule position (u32::MAX = absent).
    let mut vid_to_sched_idx: alloc::vec::Vec<usize> = alloc::vec![usize::MAX; max_vid + 1];
    for (i, def) in schedule.iter().enumerate() {
        vid_to_sched_idx[def.value.0 as usize] = i;
    }

    // Global consumer map: consumers[v.0] = every value that reads v as an
    // operand. A node may only be guarded (skipped when its arm's mask is
    // uniform) if EVERY consumer is inside that arm's subtree (or the select
    // itself) — otherwise an outer/sibling expression reads a register the
    // branch never computed. Subtree-local exclusivity (below) is necessary but
    // NOT sufficient; this is the global check that was missing.
    let mut consumers: alloc::vec::Vec<alloc::vec::Vec<ValueId>> =
        alloc::vec![alloc::vec::Vec::new(); max_vid + 1];
    for def in schedule {
        let vid = def.value;
        let mut add = |child: ValueId| {
            if (child.0 as usize) <= max_vid {
                consumers[child.0 as usize].push(vid);
            }
        };
        match &def.op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) | ScheduledOp::Uniform(_) => {}
            ScheduledOp::Unary(_, c)
            | ScheduledOp::ShiftImm(_, c, _)
            | ScheduledOp::Gather(c, _) => add(*c),
            ScheduledOp::Binary(_, a, b) => {
                add(*a);
                add(*b);
            }
            ScheduledOp::Ternary(_, a, b, c) => {
                add(*a);
                add(*b);
                add(*c);
            }
        }
    }

    for (i, def) in schedule.iter().enumerate() {
        let (sel_vid, sop) = (&def.value, &def.op);
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sop {
            // (the exclusivity analysis, unchanged)
            // Compute transitive deps for each subtree using the dense O(1) lookup
            let mask_deps = transitive_deps(*mask_vid, &schedule_ops);
            let true_deps = transitive_deps(*true_vid, &schedule_ops);
            let false_deps = transitive_deps(*false_vid, &schedule_ops);

            // A node is safe to skip under this arm only if every one of its
            // consumers is ALSO skipped under it — or is the select itself.
            // Reaching the arm is not enough: a value can be inside the arm's
            // cone and still be shared with the world outside it, and a
            // dependency of such a value would then be skipped while its
            // consumer runs, reading a register the branch never wrote.
            //
            // So exclusivity is a closure, not a filter. Seed it with the
            // values only this arm's cone reaches, then drop any whose
            // consumers are not themselves in the set, repeatedly, until the
            // set stops shrinking — the greatest set closed under "my
            // consumers are skipped with me".
            let closed_exclusive = |cone: &BTreeSet<ValueId>, other: &BTreeSet<ValueId>| {
                let mut set: BTreeSet<ValueId> = cone
                    .difference(&mask_deps)
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .difference(other)
                    .copied()
                    .collect();
                loop {
                    let doomed: alloc::vec::Vec<ValueId> = set
                        .iter()
                        .copied()
                        .filter(|v| {
                            consumers[v.0 as usize]
                                .iter()
                                .any(|c| *c != *sel_vid && !set.contains(c))
                        })
                        .collect();
                    if doomed.is_empty() {
                        return set;
                    }
                    for v in doomed {
                        set.remove(&v);
                    }
                }
            };

            let true_exclusive = closed_exclusive(&true_deps, &false_deps);
            let false_exclusive = closed_exclusive(&false_deps, &true_deps);

            // Map to schedule indices using dense O(1) lookup
            let true_indices: BTreeSet<usize> = true_exclusive
                .iter()
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();
            let false_indices: BTreeSet<usize> = false_exclusive
                .iter()
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();

            let mask_idx = vid_to_sched_idx
                .get(mask_vid.0 as usize)
                .copied()
                .unwrap_or(usize::MAX);

            let cone: BTreeSet<usize> = mask_deps
                .iter()
                .chain(true_deps.iter())
                .chain(false_deps.iter())
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();

            let arm_cycles = |indices: &BTreeSet<usize>| -> usize {
                indices
                    .iter()
                    .map(|&idx| match &schedule[idx].op {
                        ScheduledOp::Var(_) | ScheduledOp::Const(_) => 0,
                        // One broadcast load; priced as the leaf it is
                        // in the prologue, where it lands.
                        ScheduledOp::Uniform(_) => cycles.cost(OpKind::Uniform),
                        ScheduledOp::Unary(op, _) | ScheduledOp::Binary(op, _, _) => {
                            cycles.cost(*op)
                        }
                        ScheduledOp::ShiftImm(op, _, _) => cycles.cost(*op),
                        ScheduledOp::Ternary(op, _, _, _) => cycles.cost(*op),
                        ScheduledOp::Gather(_, _) => cycles.cost(OpKind::RawGather),
                    })
                    .sum()
            };
            let (true_cycles, false_cycles) =
                (arm_cycles(&true_indices), arm_cycles(&false_indices));

            arms.push(SelectArms {
                select_idx: i,
                select_vid: *sel_vid,
                mask_vid: *mask_vid,
                mask_idx,
                true_indices,
                false_indices,
                cone,
                true_cycles,
                false_cycles,
            });
        }
    }

    arms
}

/// What a guard costs when it never fires: the uniformity test, plus a
/// branch the hardware cannot predict because the mask is incoherent.
///
/// Taken as ~16 cycles, which is the mispredict penalty on the cores this
/// compiler targets — 15–20 on Intel since Skylake and on AMD since Zen
/// (Agner Fog, *The microarchitecture of Intel, AMD and VIA CPUs*, §"Branch
/// prediction"), 13–16 on ARM's recent out-of-order cores (Cortex-A76 and
/// Neoverse software optimization guides). It is an architectural figure, not
/// a knob: **do not sweep it**, and do not move it to make a kernel faster.
///
/// It is used as a *bound*, which is why one number for two architectures is
/// honest. A guard's upside depends on how often the mask is uniform, which
/// is data and unknowable here; its downside does not. An arm whose work
/// costs less than the penalty cannot pay for its own branch even if the
/// branch always fires, so guarding it is a loss in every world — while an
/// arm that costs far more is capped at this much loss and may save all of
/// it. Measured, that is the whole difference between a glyph's coverage
/// mask (a handful of ops per arm, varying per lane, 3.6x slower with a
/// guard) and a sphere's silhouette (214 entries, uniformly false in 97% of
/// batches, 3.2x faster with one).
const MISPREDICT_PENALTY_CYCLES: usize = 16;

/// How many partitions one scope may be given. Each accepted round strictly
/// increases the entries under a guard, so this only bounds the *compile*
/// cost of looking for more; it is not a correctness bound.
const MAX_CLUSTER_ROUNDS: usize = 8;

/// Reorder a scope's schedule so that a select's arm-exclusive entries form
/// one run — where that, and only that, is what stands between the arm and a
/// branch.
///
/// The transformation per select is a stable partition of the entries between
/// the mask and the select into shared, then true-exclusive, then
/// false-exclusive. It is always a legal topological order:
///
/// - No shared entry depends on an arm-exclusive one. If it did, that value
///   would have a consumer outside the arm and `only_used_within` would not
///   have called it exclusive.
/// - No true-exclusive entry depends on a false-exclusive one, or the reverse,
///   for the same reason.
/// - Relative order is preserved inside each group, and every group's
///   dependencies now precede it.
///
/// Selects are partitioned outermost first (descending schedule position: a
/// select is scheduled after everything in its arms, so an enclosing select
/// comes later), and the result is *accepted only if it is better* — strictly
/// more entries under a guard, and no select losing one it already had. That
/// is what keeps the pressure honest: shared values moved ahead of both arms
/// live across the skipped arm, which is a cost worth paying for a branch and
/// not otherwise.
pub(crate) fn cluster_select_arms(schedule: Vec<Def>) -> Vec<Def> {
    let mut current = schedule;
    let mut spans = guarded_spans(&current);

    for _ in 0..MAX_CLUSTER_ROUNDS {
        let arms = select_arms(&current);
        let mut improved = false;
        for candidate in arms.iter().rev() {
            if !candidate.refused_for_order() {
                continue;
            }
            let reordered = partition_around(&current, candidate);
            let reordered_spans = guarded_spans(&reordered);
            if !is_improvement(&spans, &reordered_spans) {
                continue;
            }
            current = reordered;
            spans = reordered_spans;
            improved = true;
            break;
        }
        if !improved {
            break;
        }
    }

    current
}

/// The entries each select has under a guard, keyed by the select's value —
/// the one identity that survives a reordering, unlike a schedule position.
fn guarded_spans(schedule: &[Def]) -> alloc::collections::BTreeMap<ValueId, (usize, usize)> {
    select_arms(schedule)
        .into_iter()
        .map(|select| {
            let (t, f) = (select.true_range(), select.false_range());
            (select.select_vid, (t.1 - t.0, f.1 - f.0))
        })
        .collect()
}

/// Strictly more schedule entries under a guard, with no select losing what it
/// already had — including the nested ones, whose arms lie inside the arm that
/// moved.
fn is_improvement(
    before: &alloc::collections::BTreeMap<ValueId, (usize, usize)>,
    after: &alloc::collections::BTreeMap<ValueId, (usize, usize)>,
) -> bool {
    let total = |spans: &alloc::collections::BTreeMap<ValueId, (usize, usize)>| -> usize {
        spans.values().map(|(t, f)| t + f).sum()
    };
    total(after) > total(before)
        && before.iter().all(|(vid, (t, f))| {
            let (at, af) = after.get(vid).copied().unwrap_or((0, 0));
            at >= *t && af >= *f
        })
}

/// The schedule with `select`'s region stable-partitioned into shared, then
/// true-exclusive, then false-exclusive entries.
fn partition_around(schedule: &[Def], select: &SelectArms) -> Vec<Def> {
    let first_arm = select
        .true_indices
        .iter()
        .chain(select.false_indices.iter())
        .copied()
        .min();
    let Some(first_arm) = first_arm else {
        return schedule.to_vec();
    };
    // The mask is shared, so it lands in the first group wherever it started;
    // the region begins at whichever of the two comes first.
    let start = first_arm.min(select.mask_idx);
    let region = start..select.select_idx;

    // A scope's result is its last entry, so nothing may be placed after it:
    // when the select IS the root, the strangers stay ahead of the arms.
    let sink_past_select = select.select_idx + 1 < schedule.len();
    let stays_before = |i: &usize| {
        (select.cone.contains(i) || !sink_past_select)
            && !select.true_indices.contains(i)
            && !select.false_indices.contains(i)
    };

    let mut out = Vec::with_capacity(schedule.len());
    out.extend_from_slice(&schedule[..start]);
    // What the select reads and neither arm owns: it must be computed before
    // the arms, because the arms read it.
    out.extend(
        region
            .clone()
            .filter(stays_before)
            .map(|i| schedule[i].clone()),
    );
    for arm in [&select.true_indices, &select.false_indices] {
        out.extend(
            arm.iter()
                .filter(|i| region.contains(i))
                .map(|i| schedule[*i].clone()),
        );
    }
    out.push(schedule[select.select_idx].clone());
    // What the select does NOT read sinks past it, keeping its order. Legal
    // for the same reason the partition is: nothing the select reads can read
    // one of these, or it would be in the cone. And it is the better place —
    // hoisting a stranger ahead of both arms would keep it live across the
    // arm a branch is there to skip, which is pressure bought for nothing.
    out.extend(
        region
            .clone()
            .filter(|i| {
                !stays_before(i)
                    && !select.true_indices.contains(i)
                    && !select.false_indices.contains(i)
            })
            .map(|i| schedule[i].clone()),
    );
    out.extend_from_slice(&schedule[select.select_idx + 1..]);

    debug_assert_eq!(
        out.len(),
        schedule.len(),
        "a partition moves entries, never adds or drops them"
    );
    debug_assert!(
        is_topological(&out),
        "a partition reordered a value ahead of an operand"
    );
    out
}

/// Every operand is defined before it is read — the property a partition must
/// preserve and the one a wrong exclusivity rule silently breaks.
///
/// This caught a real bug the day it was written: "exclusive" used to mean
/// every consumer *reaches* the arm, which admits a value whose consumer is
/// shared with the world outside it. Moving such a value behind its consumer
/// produced a kernel that read an undefined register, and the emitted code was
/// wrong in a way no unit test of the analysis would have shown.
fn is_topological(schedule: &[Def]) -> bool {
    let defined: alloc::collections::BTreeSet<ValueId> =
        schedule.iter().map(|def| def.value).collect();
    let mut seen = alloc::collections::BTreeSet::new();
    for def in schedule {
        let ready = |c: &ValueId| seen.contains(c) || !defined.contains(c);
        let ok = match &def.op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) | ScheduledOp::Uniform(_) => true,
            ScheduledOp::Unary(_, c)
            | ScheduledOp::ShiftImm(_, c, _)
            | ScheduledOp::Gather(c, _) => ready(c),
            ScheduledOp::Binary(_, a, b) => ready(a) && ready(b),
            ScheduledOp::Ternary(_, a, b, c) => ready(a) && ready(b) && ready(c),
        };
        if !ok {
            return false;
        }
        seen.insert(def.value);
    }
    true
}

/// What one `Select` in the schedule offered a guard, and what survived.
struct SelectStat {
    select_idx: usize,
    /// Where the mask lands in the schedule; a guard needs it before the arm.
    mask_idx: usize,
    /// Values exclusive to (true, false) — what a guard could skip if the
    /// exclusive set happened to be contiguous.
    exclusive: (usize, usize),
    /// What [`demand`](super::demand) calls exclusive to each arm — the
    /// values observed only where this arm's polarity holds.
    ///
    /// Always at least `exclusive`, and the gap is the point: demand
    /// answers *may this be skipped*, while `exclusive` answers the
    /// stronger *may this be skipped and also moved*, which is what
    /// [`cluster_select_arms`]'s three-way partition needs. Two selects
    /// sharing a mask separate them — the inner select's arms are
    /// demand-exclusive to the outer one, but the inner select itself is
    /// shared and reads them, so moving them past it is illegal. The gap
    /// is the headroom a partition that ordered within its groups would
    /// unlock.
    demand_exclusive: (usize, usize),
    /// Schedule entries a guard actually skips, (true, false).
    guarded: (usize, usize),
    /// Entries that are NOT this arm's but lie between its first and its last,
    /// as (total, of which leaves) — the values a single branch would have to
    /// jump over, which is why the arm is not guardable.
    intruders: ((usize, usize), (usize, usize)),
}

/// Entries inside `[min(arm), max(arm)]` that the arm does not own, and how
/// many of those are leaves (a `Const` or a coordinate). Diagnosis only.
fn intruders(arm: &alloc::collections::BTreeSet<usize>, schedule: &[Def]) -> (usize, usize) {
    let (Some(&start), Some(&end)) = (arm.iter().next(), arm.iter().next_back()) else {
        return (0, 0);
    };
    let mut total = 0;
    let mut leaves = 0;
    for (idx, def) in schedule.iter().enumerate().take(end + 1).skip(start) {
        if arm.contains(&idx) {
            continue;
        }
        total += 1;
        if matches!(def.op, ScheduledOp::Const(_) | ScheduledOp::Var(_)) {
            leaves += 1;
        }
    }
    (total, leaves)
}

/// The guard analysis, counted, on stderr when `PIXELFLOW_GUARD_TELEMETRY` is
/// set in the environment.
///
/// Diagnosis only, and off by default: nothing the emitter decides with, and
/// no emitted byte changes. It exists because "the guard did not fire" is a
/// claim about the *schedule*, and the only way to settle it is to count. The
/// two numbers per select are the two ways a guard is lost and they have
/// different fixes: `exclusive` short of the arm's size is the analysis
/// refusing (a value some other expression also reads), while `guarded` short
/// of `exclusive` is the *order* refusing (the arm's own values are not a
/// contiguous run, so one branch cannot span them).
struct Telemetry {
    stats: Option<Vec<SelectStat>>,
}

impl Telemetry {
    fn new() -> Self {
        Self {
            stats: std::env::var_os("PIXELFLOW_GUARD_TELEMETRY").map(|_| Vec::new()),
        }
    }

    /// Whether anything will read what is recorded. Callers ask before
    /// computing a statistic that costs more than reading a field.
    fn is_on(&self) -> bool {
        self.stats.is_some()
    }

    /// The stat is built lazily: computing it walks the schedule, and nothing
    /// should pay for that when the telemetry is off.
    fn select(&mut self, stat: impl FnOnce() -> SelectStat) {
        if let Some(stats) = self.stats.as_mut() {
            stats.push(stat());
        }
    }

    fn report(&self, sched_len: usize) {
        let Some(stats) = self.stats.as_ref() else {
            return;
        };
        let covered: usize = stats.iter().map(|s| s.guarded.0 + s.guarded.1).sum();
        let offered: usize = stats.iter().map(|s| s.exclusive.0 + s.exclusive.1).sum();
        let demanded: usize = stats
            .iter()
            .map(|s| s.demand_exclusive.0 + s.demand_exclusive.1)
            .sum();
        std::eprintln!(
            "guard-telemetry: schedule={sched_len} selects={} guarded={covered} \
             exclusive={offered} demand_exclusive={demanded} per_select={:?}",
            stats.len(),
            stats
                .iter()
                .map(|s| {
                    (
                        s.select_idx,
                        s.mask_idx,
                        s.exclusive,
                        s.guarded,
                        s.intruders,
                    )
                })
                .collect::<Vec<_>>()
        );
    }
}
