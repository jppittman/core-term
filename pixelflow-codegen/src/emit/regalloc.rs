//! Register allocation for scheduled DAG expressions.
//!
//! Allocation is one algorithm ([`LinearScan`]) parameterised by one
//! description of the target ([`RegisterFile`]). Everything that differs
//! between x86-64 and aarch64 — which registers hold the coordinate inputs,
//! where the allocatable window starts and how wide it is, which fixed
//! registers spilled operands reload into, how many bytes a spilled vector
//! occupies — is a field of that struct and appears nowhere else. Backends
//! declare one `const` and the allocator is architecture-independent.

use alloc::vec;
use alloc::vec::Vec;

use super::{Reg, ScheduledOp};

/// A value in the program (SSA-style).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

/// One step of a schedule: a value, and the operation that defines it.
///
/// Every step defines exactly one value — the DAG is in SSA form — so a
/// schedule is a sequence of these, and a value's program point is its index
/// in that sequence.
#[derive(Clone, Debug)]
pub struct Def {
    /// The value this step defines.
    pub value: ValueId,
    /// The operation that computes it.
    pub op: ScheduledOp,
}

/// The complete platform-dependent surface of register allocation.
///
/// Allocation policy is target-independent; only these numbers are not. A
/// backend declares one of these as a `const` next to its encodings, and
/// [`RegisterFile::checked`] turns a layout that contradicts itself — a reload
/// register inside the allocatable window, say — into a build error rather
/// than a miscompile that shows up as wrong pixels.
///
/// The register file is described once and consulted everywhere: the emitter
/// reads `reload`/`select_reload` for its spill choreography and
/// `vector_bytes` for its frame arithmetic, so there is no second copy of any
/// of it to drift.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RegisterFile {
    /// Registers holding the coordinate inputs, in order: X, Y, Z, W.
    ///
    /// A `Var(i)` in the schedule is pre-colored to `inputs[i]`; a `Var` past
    /// the end of this array is a value the target cannot supply.
    pub inputs: [Reg; 4],

    /// First allocatable scratch register.
    pub scratch_base: u8,

    /// How many allocatable scratch registers follow `scratch_base`.
    ///
    /// This is the whole pool: `scratch_base .. scratch_base + scratch_count`.
    /// Everything outside it — inputs, callee-saved registers, the reload
    /// registers, a backend's fixed builtin scratch — is off limits to the
    /// allocator by construction.
    pub scratch_count: u8,

    /// Fixed registers, outside the pool, that spilled operands reload into.
    ///
    /// Two suffice: every backend reads all of an instruction's sources before
    /// writing its destination, so a spilled destination register doubles as
    /// the temporary for one operand.
    pub reload: [Reg; 2],

    /// A third fixed reload register, for a `Select` whose result and *both*
    /// arms are spilled — `reload` is already fully committed there (the
    /// result holds `reload[0]`, the true arm `reload[1]`).
    ///
    /// Must be untouched by the backend's own `Select` emission.
    pub select_reload: Reg,

    /// Bytes one register occupies when spilled — the backend's vector width.
    ///
    /// 16 for SSE2 and NEON, 32 for AVX2, 64 for AVX-512. This is the stride
    /// [`FrameLayout`](super::FrameLayout) lays spill slots out at, so every
    /// offset the emitter sees is already a real byte displacement. It was
    /// once a universal 16 that each wide backend divided back out at its
    /// every use site; a slot offset that failed to be a multiple of 16 would
    /// then have truncated two live values onto the same stack slot.
    pub vector_bytes: u32,
}

impl RegisterFile {
    /// Reject a register file whose regions overlap.
    ///
    /// Call it on every backend's `const` declaration: const evaluation runs
    /// the checks at build time, so an allocatable window that swallows a
    /// reload register cannot reach a running kernel.
    #[must_use]
    pub const fn checked(self) -> Self {
        let lo = self.scratch_base;
        let hi = self.scratch_base + self.scratch_count;

        assert!(self.scratch_count > 0, "register file has no scratch pool");

        let mut i = 0;
        while i < self.inputs.len() {
            let r = self.inputs[i].0;
            assert!(
                r < lo || r >= hi,
                "an input register is inside the allocatable window: the \
                 allocator would hand a pre-colored register out twice"
            );
            i += 1;
        }

        let mut i = 0;
        while i < self.reload.len() {
            let r = self.reload[i].0;
            assert!(
                r < lo || r >= hi,
                "a reload register is inside the allocatable window: reloading \
                 a spilled operand would clobber another live value"
            );
            i += 1;
        }
        assert!(
            self.reload[0].0 != self.reload[1].0,
            "the two reload registers are the same register"
        );

        let sr = self.select_reload.0;
        assert!(
            sr < lo || sr >= hi,
            "select_reload is inside the allocatable window"
        );
        assert!(
            sr != self.reload[0].0 && sr != self.reload[1].0,
            "select_reload duplicates a reload register, so the all-spilled \
             Select has only two registers for three operands"
        );

        assert!(
            self.vector_bytes >= 16 && self.vector_bytes.is_power_of_two(),
            "vector_bytes must be a power of two of at least 16"
        );

        self
    }

    /// Cap the scratch pool at a smaller budget, leaving every other region
    /// where it is.
    ///
    /// This is how [`EmitCtx::max_regs`](super::EmitCtx) forces spilling for
    /// pressure testing. It only ever *shrinks* the pool: a budget above the
    /// target's own count would hand the allocator registers this file has
    /// reserved for reloads or builtins.
    #[must_use]
    pub const fn capped(self, max_scratch: u8) -> Self {
        Self {
            scratch_count: if max_scratch < self.scratch_count {
                max_scratch
            } else {
                self.scratch_count
            },
            ..self
        }
    }

    /// The allocatable scratch registers, low to high.
    fn scratch(&self) -> impl Iterator<Item = Reg> + use<> {
        let base = self.scratch_base;
        (0..self.scratch_count).map(move |i| Reg(base + i))
    }
}

/// Where the allocator decided a value lives.
///
/// Deliberately carries no stack address: choosing that a value spills and
/// choosing *where* it spills are different decisions, and the second belongs
/// to [`FrameLayout`](super::FrameLayout), which is what knows about frames.
/// The emitter reads the composition of the two as [`Loc`](super::Loc).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// In this register, for the whole of the value's live range.
    Reg(Reg),
    /// Evicted to a stack slot.
    Spilled,
    /// Evicted, but it is a constant (these are the `f32` bits): it lives
    /// nowhere and is re-emitted at each use, which beats a store plus a
    /// reload.
    Remat(u32),
}

/// A complete answer for one program: an evaluation order, and a placement for
/// every value in it.
///
/// The schedule is an *output* because choosing it is part of allocating.
/// [`LinearScan`] hands back the order it was given; an allocator whose
/// register assignment falls out of evaluation order — Sethi-Ullman, where the
/// heavier subtree is emitted first and the register is a function of tree
/// position — hands back the order it chose. Downstream, program point *is*
/// index into this schedule, so everything after allocation reads this one.
#[derive(Debug)]
pub struct Allocation {
    /// Evaluation order: the schedule the emitter walks.
    pub schedule: Vec<Def>,
    /// Dense by `ValueId.0`. Total over `schedule`; `None` only for values
    /// that are not in this schedule at all.
    placements: Vec<Option<Placement>>,
}

impl Allocation {
    /// Where `v` lives.
    ///
    /// Total over [`Allocation::schedule`] — every value the emitter walks has
    /// an answer here, which is the invariant that used to be spread across
    /// three parallel maps and defended by a runtime check that they jointly
    /// covered every value.
    ///
    /// # Panics
    /// If `v` is not in this schedule.
    #[must_use]
    pub fn placement(&self, v: ValueId) -> Placement {
        self.placements
            .get(v.0 as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("{v:?} is not in this allocation's schedule"))
    }

    /// Override where a value lives.
    ///
    /// The collapse-loop LICM pins a hoisted value to the slot its prologue
    /// parked it in, overriding whatever the allocator gave the placeholder
    /// def. One write, so the placement cannot desync from itself.
    pub fn place(&mut self, v: ValueId, placement: Placement) {
        let idx = v.0 as usize;
        if idx >= self.placements.len() {
            self.placements.resize(idx + 1, None);
        }
        self.placements[idx] = Some(placement);
    }

    /// The values that need a stack slot, in schedule order.
    ///
    /// [`FrameLayout`](super::FrameLayout) consumes exactly this.
    pub fn spilled(&self) -> impl Iterator<Item = ValueId> + use<'_> {
        self.schedule
            .iter()
            .map(|def| def.value)
            .filter(|v| self.placement(*v) == Placement::Spilled)
    }
}

/// Assign physical registers to an expression DAG.
///
/// A pure function from a program and a register file to a placement for every
/// value in it. Purity is load-bearing, not incidental: the collapse-loop
/// driver runs allocation once to size a stack frame and again to emit into
/// that frame, and a disagreement between the two runs misplaces every spill.
///
/// Allocation is not a local decision. Liveness needs the whole program, and
/// the eviction rule that makes the difference — Belady's, evict whatever is
/// used farthest in the future — is *defined* in terms of the future. So an
/// implementation sees the entire DAG, and owes an answer for every value in
/// the schedule it returns.
///
/// Running out of registers is not a failure; it is a spill. There is no error
/// case: a DAG this cannot allocate is a DAG the pipeline should never have
/// produced, and it panics at the point of failure rather than handing a
/// caller a string it can only propagate.
pub trait RegisterAllocator {
    /// Choose an evaluation order for `dag` and a placement for every value.
    ///
    /// Takes the DAG by value because choosing the order is part of the job:
    /// an implementation is free to permute what it is handed, and returns the
    /// order it settled on inside the [`Allocation`].
    fn allocate(&self, dag: Vec<Def>, file: &RegisterFile) -> Allocation;
}

/// Linear scan with Belady eviction and constant rematerialization.
///
/// Two passes. The first walks the program computing each value's last use;
/// the second walks it forward assigning registers, and at each definition:
/// 1. Frees registers whose values have expired.
/// 2. Takes a free scratch register if there is one.
/// 3. Otherwise evicts — preferring a constant (free to rematerialize, no
///    memory traffic) and otherwise the live value whose next use is farthest
///    in the future.
///
/// The evaluation order it returns is the one it was given: the arena's
/// append-only structure already guarantees a topological order, so there is
/// nothing to linearize.
///
/// Coordinate inputs are pinned to `file.inputs` and never enter the scratch
/// pool, which [`RegisterFile::checked`] keeps disjoint from them.
///
/// O(n × k) for n values and a k-register pool. For the pools here (4–10
/// registers) that is effectively O(n).
///
/// The DAGs reaching this are in SSA form, which makes their interference
/// graphs chordal — the shape on which greedy coloring is optimal. A
/// graph-coloring allocator therefore lived beside this one, unused, for as
/// long as this file existed: it never learned to rematerialize constants, and
/// on a pool this small that is the difference that decides the generated code.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearScan;

impl RegisterAllocator for LinearScan {
    fn allocate(&self, dag: Vec<Def>, file: &RegisterFile) -> Allocation {
        let vec_len = dag
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut placements: Vec<Option<Placement>> = vec![None; vec_len];

        if dag.is_empty() {
            return Allocation {
                schedule: dag,
                placements,
            };
        }

        // Pass one: last use of each value — the latest schedule index of any
        // operation that reads it. A value nothing reads (the root) is last
        // used at its own definition.
        let mut last_use: Vec<usize> = vec![usize::MAX; vec_len];
        for (i, def) in dag.iter().enumerate() {
            if last_use[def.value.0 as usize] == usize::MAX {
                last_use[def.value.0 as usize] = i;
            }
        }
        for (i, def) in dag.iter().enumerate() {
            for operand in operands(&def.op) {
                let lu = &mut last_use[operand.0 as usize];
                *lu = (*lu).max(i);
            }
        }

        // A constant is never worth a spill slot: bringing it back is one
        // instruction either way.
        let mut const_bits: Vec<Option<u32>> = vec![None; vec_len];
        for def in &dag {
            if let ScheduledOp::Const(val) = def.op {
                const_bits[def.value.0 as usize] = Some(val.to_bits());
            }
        }

        // Coordinate inputs are pinned to the registers the ABI delivers them
        // in. The scratch pool excludes those registers, so a pinned value
        // never competes for one.
        for def in &dag {
            if let ScheduledOp::Var(i) = def.op {
                placements[def.value.0 as usize] = Some(Placement::Reg(input_register(file, i)));
            }
        }

        // reg_owner[i] = the value currently held in the i'th scratch register.
        let mut reg_owner: Vec<Option<ValueId>> = vec![None; file.scratch_count as usize];
        let scratch: Vec<Reg> = file.scratch().collect();

        // Pass two: forward over the program in evaluation order.
        for (i, def) in dag.iter().enumerate() {
            let vid = &def.value;
            if placements[vid.0 as usize].is_some() {
                continue; // Pre-colored.
            }

            for slot in reg_owner.iter_mut() {
                if let Some(owner) = *slot
                    && last_use[owner.0 as usize] < i
                {
                    *slot = None;
                }
            }

            if let Some(free) = reg_owner.iter().position(Option::is_none) {
                placements[vid.0 as usize] = Some(Placement::Reg(scratch[free]));
                reg_owner[free] = Some(*vid);
                continue;
            }

            // Nothing free — evict. A constant goes first whatever its next
            // use, because rematerializing it costs no memory traffic; among
            // non-constants, Belady: whoever is used farthest out.
            let mut best_const: Option<(usize, usize)> = None; // (slot, last use)
            let mut best_any: (usize, usize) = (0, 0);

            for (slot_idx, slot) in reg_owner.iter().enumerate() {
                let Some(owner) = *slot else { continue };
                let lu = last_use[owner.0 as usize];
                if const_bits[owner.0 as usize].is_some()
                    && best_const.is_none_or(|(_, best)| lu > best)
                {
                    best_const = Some((slot_idx, lu));
                }
                if lu > best_any.1 {
                    best_any = (slot_idx, lu);
                }
            }

            let evict_slot = best_const.map_or(best_any.0, |(slot, _)| slot);
            let occupant = reg_owner[evict_slot].expect("all slots occupied but none found");

            // Which of the two goes to memory: a constant always loses, and
            // otherwise Belady decides.
            let evict_new = if const_bits[vid.0 as usize].is_some() {
                true
            } else if const_bits[occupant.0 as usize].is_some() {
                false
            } else {
                last_use[vid.0 as usize] >= last_use[occupant.0 as usize]
            };

            let loser = if evict_new { *vid } else { occupant };
            placements[loser.0 as usize] = Some(match const_bits[loser.0 as usize] {
                Some(bits) => Placement::Remat(bits),
                None => Placement::Spilled,
            });

            if !evict_new {
                placements[vid.0 as usize] = Some(Placement::Reg(scratch[evict_slot]));
                reg_owner[evict_slot] = Some(*vid);
            }
        }

        Allocation {
            schedule: dag,
            placements,
        }
    }
}

/// The register a `Var(i)` is pinned to.
///
/// A `Var` past the coordinate inputs is not a kernel the target cannot run;
/// it is a `Var` that should have been lowered away (a reduce binder, a
/// manifold-param slot) or an axis that does not exist. Both are bugs upstream
/// of here, and neither is something a caller could act on — so it fails at
/// the point of failure, with a stack trace, rather than as a string three
/// frames up.
#[cold]
#[inline(never)]
fn var_out_of_range(i: u8, inputs: usize) -> ! {
    panic!(
        "Var({i}) names coordinate {i} but the target supplies {inputs} \
         (X, Y, Z, W) — `passes::legalize` lowers every other Var, so this is \
         a bypassed pipeline or a missing lowering, not a bad kernel"
    )
}

fn input_register(file: &RegisterFile, i: u8) -> Reg {
    match file.inputs.get(i as usize) {
        Some(&reg) => reg,
        None => var_out_of_range(i, file.inputs.len()),
    }
}

/// The values an operation reads.
fn operands(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let (a, b, c) = match sop {
        ScheduledOp::Var(_) | ScheduledOp::Const(_) => (None, None, None),
        ScheduledOp::Unary(_, a) | ScheduledOp::ShiftImm(_, a, _) | ScheduledOp::Gather(a, _) => {
            (Some(*a), None, None)
        }
        ScheduledOp::Binary(_, a, b) => (Some(*a), Some(*b), None),
        ScheduledOp::Ternary(_, a, b, c) => (Some(*a), Some(*b), Some(*c)),
    };
    [a, b, c].into_iter().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::kind::OpKind;

    /// A four-register pool at base 4, mirroring the tightest real backend
    /// (AVX2's ymm4-7) so pressure tests need only a handful of values.
    const TEST_FILE: RegisterFile = RegisterFile {
        inputs: [Reg(0), Reg(1), Reg(2), Reg(3)],
        scratch_base: 4,
        scratch_count: 4,
        reload: [Reg(11), Reg(12)],
        select_reload: Reg(13),
        vector_bytes: 16,
    }
    .checked();

    fn def(value: u32, op: ScheduledOp) -> Def {
        Def {
            value: ValueId(value),
            op,
        }
    }

    fn alloc(schedule: Vec<Def>) -> Allocation {
        LinearScan.allocate(schedule, &TEST_FILE)
    }

    /// `v2 = X + Y`.
    fn add_xy() -> Vec<Def> {
        vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
        ]
    }

    #[test]
    fn an_empty_schedule_allocates_nothing() {
        let a = alloc(vec![]);
        assert!(a.schedule.is_empty());
        assert_eq!(a.spilled().count(), 0);
    }

    /// The schedule is an output, and for linear scan it is the input order:
    /// the arena's append-only structure already guarantees topological order.
    #[test]
    fn linear_scan_returns_the_order_it_was_given() {
        let a = alloc(add_xy());
        let order: Vec<ValueId> = a.schedule.iter().map(|d| d.value).collect();
        assert_eq!(order, vec![ValueId(0), ValueId(1), ValueId(2)]);
    }

    #[test]
    fn vars_are_pinned_to_the_input_registers() {
        let a = alloc(add_xy());
        assert_eq!(a.placement(ValueId(0)), Placement::Reg(Reg(0)), "X");
        assert_eq!(a.placement(ValueId(1)), Placement::Reg(Reg(1)), "Y");
        // The sum takes the pool, never an input register.
        assert_eq!(a.placement(ValueId(2)), Placement::Reg(Reg(4)));
        assert_eq!(a.spilled().count(), 0);
    }

    /// Every value the emitter walks has a placement — the invariant that used
    /// to be spread across three parallel maps and checked at runtime.
    #[test]
    fn placement_is_total_over_the_schedule() {
        let a = alloc(add_xy());
        let placed: Vec<Placement> = a.schedule.iter().map(|d| a.placement(d.value)).collect();
        assert_eq!(placed.len(), a.schedule.len());
    }

    #[test]
    #[should_panic(expected = "names coordinate 4")]
    fn a_var_past_the_input_registers_panics() {
        let _ = alloc(vec![def(0, ScheduledOp::Var(4))]);
    }

    /// Two values whose live ranges do not overlap may share one register —
    /// that is the whole point of tracking last use rather than defs.
    #[test]
    fn disjoint_live_ranges_share_a_register() {
        let a = alloc(vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(2))),
            def(4, ScheduledOp::Unary(OpKind::Abs, ValueId(3))),
        ]);
        assert_eq!(a.spilled().count(), 0);
        // v2's last use is v3, so v4 may reuse v2's register.
        assert_eq!(a.placement(ValueId(2)), a.placement(ValueId(4)));
    }

    /// Six values live at once cannot fit a four-register pool.
    #[test]
    fn pressure_beyond_the_pool_spills() {
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=6u32 {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let mut acc = ValueId(1);
        for i in 2..=6u32 {
            schedule.push(def(
                6 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(6 + i);
        }
        let a = alloc(schedule);
        assert!(a.spilled().count() > 0);
        // A placement is one choice, so nothing is both in a register and on
        // the stack — the miscompile the three parallel maps could express.
        for v in a.spilled() {
            assert_eq!(a.placement(v), Placement::Spilled);
        }
    }

    /// A constant under pressure is rematerialized, never spilled: re-emitting
    /// the load beats a store plus a reload.
    #[test]
    fn constants_are_rematerialized_rather_than_spilled() {
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=6u32 {
            schedule.push(def(i, ScheduledOp::Const(i as f32)));
        }
        let mut acc = ValueId(1);
        for i in 2..=6u32 {
            schedule.push(def(
                6 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(6 + i);
        }
        let a = alloc(schedule);

        let remat: Vec<(ValueId, u32)> = (1..=6u32)
            .filter_map(|i| match a.placement(ValueId(i)) {
                Placement::Remat(bits) => Some((ValueId(i), bits)),
                _ => None,
            })
            .collect();
        assert!(!remat.is_empty(), "constants under pressure should remat");
        assert_eq!(
            a.spilled().count(),
            0,
            "no constant belongs in a spill slot"
        );
        for (vid, bits) in remat {
            assert_eq!(
                bits,
                (vid.0 as f32).to_bits(),
                "{vid:?} rematerializes the wrong constant"
            );
        }
    }

    /// Belady: with no constants in play, the value used farthest in the
    /// future is the one that goes to memory.
    ///
    /// The scenario separates Belady from FIFO and LRU deliberately. Four
    /// values fill the pool in the order v1..v4, then v5 forces an eviction —
    /// but they are *consumed* in that same order, so v1 is simultaneously the
    /// oldest, the least recently used, and the one needed soonest. FIFO and
    /// LRU both evict v1. Only a rule that looks forward evicts v4.
    #[test]
    fn belady_evicts_the_value_used_farthest_out() {
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=5u32 {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        // Consume v1 first, then v2, v3, and v4 last.
        let mut acc = ValueId(5);
        for i in 1..=4u32 {
            schedule.push(def(
                100 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(100 + i);
        }
        let a = alloc(schedule);

        assert_eq!(
            a.placement(ValueId(4)),
            Placement::Spilled,
            "v4 is needed last, so it is the one to evict"
        );
        assert!(
            matches!(a.placement(ValueId(1)), Placement::Reg(_)),
            "v1 is needed next, so it must keep its register — evicting it is \
             what FIFO and LRU would have done"
        );
    }

    /// Purity is load-bearing: the collapse driver sizes a frame with one run
    /// and emits into it with another, and a disagreement misplaces every slot.
    #[test]
    fn allocation_is_deterministic() {
        let a = alloc(add_xy());
        let b = alloc(add_xy());
        for d in &a.schedule {
            assert_eq!(a.placement(d.value), b.placement(d.value));
        }
        assert_eq!(
            a.spilled().collect::<Vec<_>>(),
            b.spilled().collect::<Vec<_>>()
        );
    }

    /// A hoisted value is pinned to the slot its prologue parked it in,
    /// overriding whatever the allocator gave the placeholder def.
    #[test]
    fn a_placement_can_be_overridden() {
        let mut a = alloc(add_xy());
        assert!(matches!(a.placement(ValueId(2)), Placement::Reg(_)));
        a.place(ValueId(2), Placement::Spilled);
        assert_eq!(a.placement(ValueId(2)), Placement::Spilled);
        assert_eq!(a.spilled().collect::<Vec<_>>(), vec![ValueId(2)]);
    }

    /// All three `Ternary` operands count toward liveness. Missing one frees a
    /// register that is still in use.
    #[test]
    fn every_ternary_operand_extends_liveness() {
        let sel = ScheduledOp::Ternary(OpKind::Select, ValueId(0), ValueId(1), ValueId(2));
        assert_eq!(
            operands(&sel).collect::<Vec<_>>(),
            vec![ValueId(0), ValueId(1), ValueId(2)]
        );
        let a = alloc(vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Var(2)),
            def(3, sel),
        ]);
        assert_eq!(a.spilled().count(), 0);
    }

    /// `checked()` is the isolation's teeth: a file whose reload register sits
    /// inside the allocatable window would let a reload clobber a live value.
    #[test]
    #[should_panic(expected = "reload register is inside the allocatable window")]
    fn a_reload_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            scratch_count: 8, // swallows Reg(11)
            ..TEST_FILE
        }
        .checked();
    }

    #[test]
    #[should_panic(expected = "select_reload duplicates a reload register")]
    fn a_select_reload_aliasing_a_reload_register_is_refused() {
        let _refused = RegisterFile {
            select_reload: Reg(12),
            ..TEST_FILE
        }
        .checked();
    }

    #[test]
    #[should_panic(expected = "input register is inside the allocatable window")]
    fn an_input_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            inputs: [Reg(0), Reg(1), Reg(2), Reg(4)],
            ..TEST_FILE
        }
        .checked();
    }

    #[test]
    #[should_panic(expected = "vector_bytes")]
    fn a_non_power_of_two_vector_width_is_refused() {
        let _refused = RegisterFile {
            vector_bytes: 24,
            ..TEST_FILE
        }
        .checked();
    }
}
