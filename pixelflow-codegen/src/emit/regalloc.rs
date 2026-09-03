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

use pixelflow_ir::kind::OpKind;

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
/// reads `reload` for its spill choreography and `vector_bytes` for its frame
/// arithmetic, so there is no second copy of any of it to drift.
/// A set of registers from one file, as a bitmask over register numbers.
///
/// The allocatable pool used to be a base plus a count — a contiguous run. On
/// every real target the free registers are *not* contiguous: on SSE2 `xmm14`
/// and `xmm15` sit above the reload pair, and on AVX-512 fifteen free
/// registers sit above it with the gather's scratch in between. A range can only ever name whichever free registers happen to be
/// adjacent, so it silently rounded the pool down to a fraction of the machine.
///
/// A set says the true thing: these registers, whichever they are.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RegSet(u32);

impl RegSet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// The set containing exactly `regs`.
    #[must_use]
    pub const fn of(regs: &[Reg]) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < regs.len() {
            let r = regs[i].0;
            assert!(
                r < 32,
                "register number out of range for a 32-register file"
            );
            bits |= 1 << r;
            i += 1;
        }
        Self(bits)
    }

    /// The contiguous run `base .. base + count`.
    #[must_use]
    pub const fn range(base: u8, count: u8) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < count {
            let r = base + i;
            assert!(
                r < 32,
                "register number out of range for a 32-register file"
            );
            bits |= 1 << r;
            i += 1;
        }
        Self(bits)
    }

    /// This set plus every member of `other`.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn contains(self, r: Reg) -> bool {
        r.0 < 32 && self.0 & (1 << r.0) != 0
    }

    /// How many registers the set holds.
    #[must_use]
    pub const fn len(self) -> u8 {
        self.0.count_ones() as u8
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The lowest `n` members, or all of them if the set is smaller.
    ///
    /// This is how [`EmitCtx::max_regs`](super::EmitCtx) forces spilling for
    /// pressure testing: it only ever shrinks.
    #[must_use]
    pub const fn take(self, n: u8) -> Self {
        let mut kept = 0u32;
        let mut taken = 0u8;
        let mut r = 0u8;
        while r < 32 {
            if taken < n && self.0 & (1 << r) != 0 {
                kept |= 1 << r;
                taken += 1;
            }
            r += 1;
        }
        Self(kept)
    }

    /// Members low to high.
    pub fn iter(self) -> impl Iterator<Item = Reg> + use<> {
        (0u8..32).filter(move |r| self.0 & (1 << r) != 0).map(Reg)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RegisterFile {
    /// Registers holding the coordinate inputs, in order: X, Y, Z, W.
    ///
    /// A `Var(i)` in the schedule is pre-colored to `inputs[i]`; a `Var` past
    /// the end of this array is a value the target cannot supply.
    pub inputs: [Reg; 4],

    /// Every register the allocator may hand out.
    ///
    /// Everything outside it — inputs, callee-saved registers, the reload
    /// registers, the backend's own `fixed` scratch — is off limits by
    /// construction, and [`RegisterFile::checked`] proves the separation.
    pub scratch: RegSet,

    /// Fixed registers, outside the pool, that spilled operands reload into.
    ///
    /// Two suffice: every backend reads all of an instruction's sources before
    /// writing its destination, so a spilled destination register doubles as
    /// the temporary for one operand.
    pub reload: [Reg; 2],

    /// Registers the backend's own instruction emission clobbers, outside the
    /// allocator's knowledge: an ISA-level temp for a two-operand form, a
    /// gather's index register, and the like.
    ///
    /// The allocator never hands these out; declaring them is what lets
    /// [`RegisterFile::checked`] prove they miss the pool, the inputs and the
    /// reload pair — rather than a comment in an ISA file asserting it. Anything a backend takes for itself belongs here.
    pub fixed: &'static [Reg],

    /// How many registers this backend's encoding of `op` needs beyond the
    /// operands and destination — the instruction temps.
    ///
    /// A two-operand ISA needs one to break a destructive hazard; a sign-flip
    /// needs one to hold the mask. Those used to be `const`s outside the pool,
    /// reserved for the whole kernel because one instruction in it might want
    /// one. Declaring the demand here instead makes the temp an *allocated*
    /// value with a live range of exactly one instruction, so the register is
    /// the allocator's everywhere else.
    ///
    /// It lives on the register file because the file is already "the whole of
    /// what allocation needs to know about the target" — a backend that needs
    /// a temp is stating a fact about its register requirements, which is what
    /// this type is for.
    pub temps_for: fn(&ScheduledOp) -> u8,

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
        assert!(
            self.scratch.len() >= Self::MIN_SCRATCH,
            "the allocatable pool is too small for the widest instruction's \
             operands, scratch and destination at once"
        );

        let mut i = 0;
        while i < self.inputs.len() {
            assert!(
                !self.scratch.contains(self.inputs[i]),
                "an input register is inside the allocatable pool: the \
                 allocator would hand a pre-colored register out twice"
            );
            i += 1;
        }

        let mut i = 0;
        while i < self.reload.len() {
            assert!(
                !self.scratch.contains(self.reload[i]),
                "a reload register is inside the allocatable pool: reloading \
                 a spilled operand would clobber another live value"
            );
            i += 1;
        }
        assert!(
            self.reload[0].0 != self.reload[1].0,
            "the two reload registers are the same register"
        );

        assert!(
            self.vector_bytes >= 16 && self.vector_bytes.is_power_of_two(),
            "vector_bytes must be a power of two of at least 16"
        );

        // Everything a backend reserves for its own emission must miss every
        // register the allocator reasons about. Without this the disjointness
        // lives only in comments.
        let mut i = 0;
        while i < self.fixed.len() {
            assert!(
                !self.scratch.contains(self.fixed[i]),
                "a fixed backend scratch register is inside the allocatable \
                 pool: emitting an instruction would clobber a live value"
            );
            let mut k = 0;
            while k < self.reload.len() {
                assert!(
                    self.fixed[i].0 != self.reload[k].0,
                    "a fixed backend scratch register aliases a reload register"
                );
                k += 1;
            }
            let mut k = 0;
            while k < self.inputs.len() {
                assert!(
                    self.fixed[i].0 != self.inputs[k].0,
                    "a fixed backend scratch register aliases a coordinate input"
                );
                k += 1;
            }
            i += 1;
        }

        self
    }

    /// The smallest pool any schedule can be allocated against.
    ///
    /// Every *value* survives a small pool by spilling, so the pool has no
    /// lower bound from values alone — that is why one register used to be an
    /// acceptable budget. Instruction scratch has no such escape: it is
    /// registers the encoder destroys mid-instruction, each of which must be a
    /// register, and one that is neither the destination nor any operand.
    ///
    /// Six, set by the widest single demand rather than by the widest op. A
    /// ternary is three operands plus one temp; AVX2's gather is one operand
    /// plus [`Scratch::MAX_TEMPS`] — four, since it assembles two 128-bit
    /// halves — and both leave room for the destination.
    ///
    /// Below this, shrinking the pool stops producing more spilling and starts
    /// producing an instruction with nowhere to put its scratch.
    pub const MIN_SCRATCH: u8 = Scratch::MAX_TEMPS as u8 + 2;

    /// Cap the scratch pool at a smaller budget, leaving every other region
    /// where it is.
    ///
    /// This is how [`EmitCtx::max_regs`](super::EmitCtx) forces spilling for
    /// pressure testing. It only ever *shrinks* the pool: a budget above the
    /// target's own count would hand the allocator registers this file has
    /// reserved for reloads or builtins. It does not shrink past
    /// [`MIN_SCRATCH`](Self::MIN_SCRATCH), which is not a budget question but
    /// an encoding one.
    #[must_use]
    pub const fn capped(self, max_scratch: Option<u8>) -> Self {
        match max_scratch {
            Some(n) => Self {
                scratch: self.scratch.take(if n < Self::MIN_SCRATCH {
                    Self::MIN_SCRATCH
                } else {
                    n
                }),
                ..self
            },
            None => self,
        }
    }

    /// The allocatable scratch registers, low to high.
    fn scratch(&self) -> impl Iterator<Item = Reg> + use<> {
        self.scratch.iter()
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
    /// The scratch each position in `schedule` may destroy.
    ///
    /// Indexed by schedule position rather than by value because scratch is not
    /// a value: it holds nothing before the instruction and nothing after, so
    /// it has no `ValueId` and no place in `placements`.
    scratch: Vec<Scratch>,
}

/// The registers one instruction may destroy for its own duration.
///
/// Each field is a role, not a slot — the allocator picks a register for each
/// one it fills, disjoint from the instruction's operands, its destination, and
/// each other. Reading them positionally out of a shared array is exactly the
/// convention this replaced.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Scratch {
    /// The encoding's own scratch: a sign mask, a Newton-Raphson correction,
    /// the halves a gather assembles its result from — as many registers as
    /// [`RegisterFile::temps_for`] asked for, and no more.
    ///
    /// Private, and read through [`Scratch::temp`], because the *order* here is
    /// a contract between one backend's `temps_for` and that same backend's
    /// encoder. That is a local agreement inside one ISA file; letting anything
    /// else index it would make it a convention spanning the codebase, which is
    /// what the roles below exist to avoid.
    temps: [Option<Reg>; Scratch::MAX_TEMPS],

    /// A third register to reload an operand into.
    ///
    /// `Select` is the one instruction that can need three reload targets at
    /// once. Its mask has to end up in `dst`, so a spilled mask consumes the
    /// destination, and the two arms then need a target each; [`RegisterFile`]
    /// carries two. The third used to be `select_reload`, a register held out
    /// of *every* kernel's pool for the sake of the rare kernel that reaches
    /// this.
    ///
    /// Reserved for every `Select` rather than only the ones that turn out to
    /// need it, because a value resident when its reader is allocated can still
    /// be evicted by a later instruction — `Placement` is one answer per value
    /// for its whole life — so residency read at this point is not yet final.
    /// Over-reserving costs one register at a `Select`; the register it
    /// replaces cost one everywhere.
    pub arm_reload: Option<Reg>,
}

impl Scratch {
    /// The most scratch registers any one encoding asks for.
    ///
    /// Four: AVX2 assembles a 256-bit gather from two 128-bit halves, which
    /// costs the half-sequence's own index and value registers plus one of
    /// each to carry the high half while the low one is built.
    pub const MAX_TEMPS: usize = 4;

    /// A `Scratch` with the registers a test wants to hand an encoder.
    ///
    /// The allocator is what fills these in production; a test that exercises
    /// one encoding in isolation has no allocator, so it says outright which
    /// registers the encoder may destroy.
    #[cfg(test)]
    #[must_use]
    pub const fn for_test(temps: Option<[Reg; Self::MAX_TEMPS]>, arm_reload: Option<Reg>) -> Self {
        let temps = match temps {
            Some([a, b, c, d]) => [Some(a), Some(b), Some(c), Some(d)],
            None => [None; Self::MAX_TEMPS],
        };
        Self { temps, arm_reload }
    }

    /// The `i`'th register this instruction's encoding asked for.
    ///
    /// `i` is the backend's own numbering, matching the count its
    /// [`RegisterFile::temps_for`] returned.
    #[must_use]
    pub fn temp(&self, i: usize) -> Option<Reg> {
        self.temps.get(i).copied().flatten()
    }
}

impl Allocation {
    /// The scratch the instruction at schedule position `i` may destroy.
    #[must_use]
    pub fn scratch(&self, i: usize) -> Scratch {
        self.scratch.get(i).copied().unwrap_or_default()
    }

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

    /// Allocate a whole loop nest: every region, plus the slot order for the
    /// values that outlive the region computing them.
    ///
    /// The emitter used to call [`allocate`](Self::allocate) once per region
    /// and work out for itself which values crossed a back edge and where to
    /// park them. That made loop-carried liveness the *emitter's* concept,
    /// which is why it could only ever be answered one way — a fixed memory
    /// slot. Handing the allocator the nest puts the question where the
    /// answer lives: a value defined in one region and read in a region
    /// inside it has a live range spanning a back edge, which is an ordinary
    /// fact about liveness and not a special case.
    ///
    /// The policy here is still the old one — every carried value gets a
    /// slot — so this changes nothing about the code that comes out. It
    /// changes who is entitled to decide.
    ///
    /// Slots are named by index, not by offset: where slot `k` actually sits
    /// depends on the scaffold's own frame, which is the emitter's business.
    fn allocate_nest(&self, nest: ScopedSchedule, file: &RegisterFile) -> NestAllocation {
        let regions = nest
            .regions
            .into_iter()
            .map(|region| RegionAllocation {
                allocation: self.allocate(region.schedule, file),
                parked: region.roots,
            })
            .collect();
        NestAllocation {
            regions,
            body: self.allocate(nest.body, file),
        }
    }
}

/// A schedule split by scope: the innermost body, plus one region per binder
/// it can be lifted out of.
///
/// This is the loop nest as data. `regions` runs outermost first, so
/// `regions[0]` is what happens once per call and the last region is what
/// happens once per iteration of the next-to-innermost binder; `body` is what
/// happens at every sample. Each region's `roots` are the values it computes
/// for the regions inside it.
///
/// The body is a field rather than the last element of `regions` because it
/// is genuinely a different thing: it is inside every back edge, it produces
/// the result, and it parks nothing. A list that could hold it would let the
/// two be confused.
pub struct ScopedSchedule {
    /// One per binder, outermost first. A binder nothing can be lifted out of
    /// still gets a region; it is simply empty.
    pub regions: Vec<ScopeRegion>,
    /// The innermost region: evaluated at every sample.
    pub body: Vec<Def>,
}

/// One scope of a [`ScopedSchedule`].
pub struct ScopeRegion {
    /// Values this region computes for the ones inside it.
    pub roots: Vec<ValueId>,
    /// What it computes, in topological order.
    pub schedule: Vec<Def>,
}

/// What an allocator makes of a [`ScopedSchedule`].
pub struct NestAllocation {
    /// One per input region, in the same order — outermost first.
    pub regions: Vec<RegionAllocation>,
    /// The innermost body.
    pub body: Allocation,
}

/// One region's allocation, and what it leaves behind for the regions inside.
pub struct RegionAllocation {
    /// Placements and evaluation order for this region alone.
    pub allocation: Allocation,
    /// The values this region computes that outlive it, in slot order.
    pub parked: Vec<ValueId>,
}

impl NestAllocation {
    /// Every carried value, outermost region first — the order that defines
    /// slot indices.
    pub fn carried(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.regions.iter().flat_map(|r| r.parked.iter().copied())
    }

    /// Allocations for every region and then the body, in emission order.
    pub fn allocations(&self) -> impl Iterator<Item = &Allocation> + '_ {
        self.regions
            .iter()
            .map(|r| &r.allocation)
            .chain(core::iter::once(&self.body))
    }
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

        let mut scratch_for: Vec<Scratch> = vec![Scratch::default(); dag.len()];

        if dag.is_empty() {
            return Allocation {
                schedule: dag,
                placements,
                scratch: scratch_for,
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
        let mut reg_owner: Vec<Option<ValueId>> = vec![None; file.scratch.len() as usize];
        let scratch: Vec<Reg> = file.scratch().collect();

        // Pass two: forward over the program in evaluation order.
        for (i, def) in dag.iter().enumerate() {
            let vid = &def.value;

            for slot in reg_owner.iter_mut() {
                if let Some(owner) = *slot
                    && last_use[owner.0 as usize] < i
                {
                    *slot = None;
                }
            }

            // Scratch, reserved before the destination is placed: the encoder
            // writes it while every operand is still live and before the
            // destination is written, so it may share a register with neither.
            let mut taken: [Option<usize>; Scratch::MAX_TEMPS + 1] = [None; Scratch::MAX_TEMPS + 1];
            let mut reserve = |role: usize,
                               reg_owner: &mut Vec<Option<ValueId>>,
                               placements: &mut Vec<Option<Placement>>|
             -> Reg {
                let slot = reserve_scratch(def, reg_owner, &last_use, &const_bits, &taken);
                taken[role] = Some(slot);
                // Whatever was there loses its register for good: `placements`
                // is one answer per value, so an evicted value is spilled over
                // its whole life, exactly as destination eviction below does.
                if let Some(evicted) = reg_owner[slot].take() {
                    placements[evicted.0 as usize] = Some(match const_bits[evicted.0 as usize] {
                        Some(bits) => Placement::Remat(bits),
                        None => Placement::Spilled,
                    });
                }
                scratch[slot]
            };

            let wanted = (file.temps_for)(&def.op) as usize;
            assert!(
                wanted <= Scratch::MAX_TEMPS,
                "a backend asked for {wanted} scratch registers for one \
                 instruction; `Scratch::MAX_TEMPS` is {}",
                Scratch::MAX_TEMPS
            );
            for role in 0..wanted {
                scratch_for[i].temps[role] = Some(reserve(role, &mut reg_owner, &mut placements));
            }
            if matches!(def.op, ScheduledOp::Ternary(OpKind::Select, ..)) {
                scratch_for[i].arm_reload =
                    Some(reserve(Scratch::MAX_TEMPS, &mut reg_owner, &mut placements));
            }
            let reserved = taken;

            if placements[vid.0 as usize].is_some() {
                // Pre-colored. The temp's slot stays free for the next
                // instruction: nothing owns it after this one.
                continue;
            }

            if let Some(free) = reg_owner
                .iter()
                .enumerate()
                .position(|(idx, slot)| slot.is_none() && !reserved.contains(&Some(idx)))
            {
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
                if reserved.contains(&Some(slot_idx)) {
                    continue; // Reserved for this instruction's own scratch.
                }
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
            debug_assert!(
                !reserved.contains(&Some(evict_slot)),
                "the destination must not evict this instruction's own scratch"
            );
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
            scratch: scratch_for,
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
/// A backend whose encodings never need a register beyond their operands.
///
/// The default for [`RegisterFile::temps_for`]; naming it keeps the field
/// total, so a new backend states its answer rather than inheriting one.
#[must_use]
pub fn no_temps(_op: &ScheduledOp) -> u8 {
    0
}

/// Pick the scratch slot the instruction at `i` may destroy as its temp.
///
/// A free slot if there is one; otherwise the same eviction rule the
/// destination uses — a constant first, since rematerializing it costs no
/// memory traffic, and Belady among the rest.
///
/// Operands are excluded outright: the encoder writes the temp *before* it
/// reads the operands (that is what the temp is for), so sharing a register
/// with one would feed the instruction its own scratch. So are the roles this
/// instruction has already reserved, which is what `taken` carries.
///
/// A slot always survives both exclusions, and
/// [`RegisterFile::MIN_SCRATCH`] is what makes that true: the instructions
/// wanting several temps are the gathers, which read one operand, and the
/// ones reading three operands want at most one temp plus an arm reload — so
/// four exclusions is the worst case against a floor of `MAX_TEMPS + 2`.
fn reserve_scratch(
    def: &Def,
    reg_owner: &[Option<ValueId>],
    last_use: &[usize],
    const_bits: &[Option<u32>],
    taken: &[Option<usize>],
) -> usize {
    let is_operand = |v: ValueId| operands(&def.op).any(|o| o == v);

    let mut best_const: Option<(usize, usize)> = None;
    let mut best_any: Option<(usize, usize)> = None;
    for (slot_idx, slot) in reg_owner.iter().enumerate() {
        if taken.contains(&Some(slot_idx)) {
            continue; // Already this instruction's, in another role.
        }
        let Some(owner) = *slot else {
            return slot_idx; // Free: nothing to evict.
        };
        if is_operand(owner) {
            continue;
        }
        let lu = last_use[owner.0 as usize];
        if const_bits[owner.0 as usize].is_some() && best_const.is_none_or(|(_, best)| lu > best) {
            best_const = Some((slot_idx, lu));
        }
        if best_any.is_none_or(|(_, best)| lu > best) {
            best_any = Some((slot_idx, lu));
        }
    }

    best_const
        .or(best_any)
        .map(|(slot, _)| slot)
        .unwrap_or_else(|| {
            unreachable!(
                "every scratch register holds an operand of one instruction, \
                 so the pool is at most three registers — `RegisterFile::checked` \
                 and `capped` both hold it at {}",
                RegisterFile::MIN_SCRATCH
            )
        })
}

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

    /// The smallest pool a register file may declare, so pressure tests need
    /// only a handful of values to reach spilling.
    const TEST_FILE: RegisterFile = RegisterFile {
        fixed: &[],
        inputs: [Reg(0), Reg(1), Reg(2), Reg(3)],
        scratch: RegSet::range(4, RegisterFile::MIN_SCRATCH),
        reload: [Reg(11), Reg(12)],
        temps_for: no_temps,
        vector_bytes: 16,
    }
    .checked();

    /// `TEST_FILE`, but every `Neg` asks for a temp — the AVX2 sign-mask case.
    const TEMP_FILE: RegisterFile = RegisterFile {
        temps_for: neg_wants_a_temp,
        ..TEST_FILE
    }
    .checked();

    fn neg_wants_a_temp(op: &ScheduledOp) -> u8 {
        u8::from(matches!(op, ScheduledOp::Unary(OpKind::Neg, _)))
    }

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

    // -------------------------------------------------------------------------
    // Instruction temps
    // -------------------------------------------------------------------------

    /// The temp is a real register from the pool, and it is nobody's operand
    /// and not the destination — the encoder writes it while the operands are
    /// still live and before it writes `dst`.
    #[test]
    fn a_temp_collides_with_neither_the_operand_nor_the_destination() {
        let schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
        ];
        let a = LinearScan.allocate(schedule, &TEMP_FILE);

        let temp = a.scratch(1).temp(0).expect("`Neg` asked for a temp");
        assert!(
            TEMP_FILE.scratch.contains(temp),
            "{temp:?} is not the pool's"
        );
        assert_ne!(Placement::Reg(temp), a.placement(ValueId(0)));
        assert_ne!(Placement::Reg(temp), a.placement(ValueId(1)));
    }

    /// A `Select` gets its own third reload target, disjoint from its operands,
    /// its destination and its encoding temp — the four registers the
    /// instruction can be using at once.
    #[test]
    fn a_select_reserves_a_third_reload_target() {
        let schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
            def(4, ScheduledOp::Binary(OpKind::Sub, ValueId(0), ValueId(1))),
            def(
                5,
                ScheduledOp::Ternary(OpKind::Select, ValueId(2), ValueId(3), ValueId(4)),
            ),
        ];
        let a = LinearScan.allocate(schedule, &TEMP_FILE);
        let s = a.scratch(5);

        let arm = s.arm_reload.expect("a `Select` always reserves one");
        assert!(TEMP_FILE.scratch.contains(arm), "{arm:?} is not the pool's");
        for v in [ValueId(2), ValueId(3), ValueId(4), ValueId(5)] {
            assert_ne!(
                Placement::Reg(arm),
                a.placement(v),
                "{arm:?} is still holding {v:?} when the Select reloads into it"
            );
        }
        assert_ne!(Some(arm), s.temp(0), "the two roles must be two registers");
    }

    /// Only a `Select` asks: every other shape leaves the register alone.
    #[test]
    fn nothing_but_a_select_reserves_an_arm_reload() {
        let a = LinearScan.allocate(add_xy(), &TEMP_FILE);
        assert_eq!(a.scratch(2).arm_reload, None);
    }

    /// A pool shrunk below what an encoding needs is not a smaller budget, it
    /// is an instruction with nowhere to put its temp — so `capped` holds the
    /// floor rather than letting `max_regs` reach through it.
    #[test]
    fn capping_the_pool_stops_at_the_floor() {
        let tiny = TEMP_FILE.capped(Some(1));
        assert_eq!(tiny.scratch.len(), RegisterFile::MIN_SCRATCH);

        // The shape that used to fall through: a `Neg` whose operand is a
        // computed value, so the operand is pool-resident rather than
        // pre-colored into an input register.
        let a = LinearScan.allocate(
            vec![
                def(0, ScheduledOp::Var(0)),
                def(1, ScheduledOp::Var(1)),
                def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
                def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(2))),
            ],
            &tiny,
        );
        let temp = a.scratch(3).temp(0).expect("`Neg` asked for a temp");
        assert_ne!(Placement::Reg(temp), a.placement(ValueId(2)));
    }

    /// Only the ops that asked get one — the register is the allocator's
    /// everywhere else, which is the whole reason for asking per-op.
    #[test]
    fn an_op_that_asks_for_no_temp_gets_none() {
        let a = LinearScan.allocate(add_xy(), &TEMP_FILE);
        assert_eq!(a.scratch(2).temp(0), None, "`Add` asked for no temp");
    }

    /// The temp's live range is one instruction: the next one may take the
    /// same register for its result.
    #[test]
    fn a_temp_is_free_again_at_the_next_instruction() {
        let schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
            def(2, ScheduledOp::Unary(OpKind::Sqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Sqrt, ValueId(2))),
            def(4, ScheduledOp::Unary(OpKind::Sqrt, ValueId(3))),
        ];
        let a = LinearScan.allocate(schedule, &TEMP_FILE);
        assert_eq!(
            a.spilled().count(),
            0,
            "four values fit a four-register pool"
        );

        let temp = a.scratch(1).temp(0).expect("`Neg` asked for a temp");
        let reused = a
            .schedule
            .iter()
            .any(|d| a.placement(d.value) == Placement::Reg(temp));
        assert!(reused, "{temp:?} went back to the pool after the `Neg`");
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
        // One more independent value than the pool holds, so exactly one must
        // go to memory and the test is about *which*.
        let live = u32::from(RegisterFile::MIN_SCRATCH) + 1;
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=live {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        // Consume v1 first, then v2, v3, … and v`live-1` last.
        let mut acc = ValueId(live);
        for i in 1..live {
            schedule.push(def(
                100 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(100 + i);
        }
        let a = alloc(schedule);

        assert_eq!(
            a.placement(ValueId(live - 1)),
            Placement::Spilled,
            "v{live_minus_1} is needed last, so it is the one to evict",
            live_minus_1 = live - 1
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

    /// A destination never lands in a register one of its own operands is
    /// still living in — the invariant `resolve_operands` reads back off the
    /// allocation, and the reason SSE2 can write its two-operand form
    /// directly.
    ///
    /// `dst op= right` corrupts `right` when `dst == right` and `dst != left`.
    /// That assignment is unrepresentable here: a destination takes a slot
    /// with no live owner, or evicts one — and an evicted value's placement
    /// becomes `Spilled` for its whole life, so it is read back from memory
    /// rather than from the register the destination took. The backend needs
    /// no stashing temp to route around a case that cannot arise, which is
    /// what `emit_binary_safe` used to be and what held xmm10 out of every
    /// kernel's pool.
    #[test]
    fn a_destination_never_lands_on_a_resident_operand() {
        // Wide enough to evict: `width` values all live at once over a pool of
        // `MIN_SCRATCH`, then folded pairwise so every fold reads two of them.
        let width = u32::from(RegisterFile::MIN_SCRATCH) * 3;
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=width {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let mut acc = ValueId(1);
        for i in 2..=width {
            schedule.push(def(
                width + i,
                ScheduledOp::Binary(OpKind::Sub, acc, ValueId(i)),
            ));
            acc = ValueId(width + i);
        }

        // Both files: `TEMP_FILE` also reserves a temp before placing the
        // destination, which is the other way a destination could be pushed
        // onto an operand.
        for file in [&TEST_FILE, &TEMP_FILE] {
            let a = LinearScan.allocate(schedule.clone(), file);
            assert!(
                a.spilled().count() > 0,
                "the schedule has to reach eviction for this to test anything"
            );
            // Pairs where both ends are the pool's — the case the two-operand
            // form would corrupt. An input register can never collide with a
            // destination (the pool excludes them), so counting those would
            // let the test pass vacuously.
            let mut contested = 0;
            for d in &a.schedule {
                let Placement::Reg(dst) = a.placement(d.value) else {
                    continue; // Spilled: the destination is `reload[0]`, not the pool's.
                };
                for operand in operands(&d.op) {
                    let at = a.placement(operand);
                    assert_ne!(
                        at,
                        Placement::Reg(dst),
                        "{:?}: destination {dst:?} is where its operand {operand:?} lives",
                        d.value
                    );
                    if matches!(at, Placement::Reg(r) if file.scratch.contains(r)) {
                        contested += 1;
                    }
                }
            }
            assert!(
                contested > 0,
                "no instruction read a pool-resident operand, so nothing above \
                 could have collided"
            );
        }
    }

    /// `checked()` is the isolation's teeth: a file whose reload register sits
    /// inside the allocatable pool would let a reload clobber a live value.
    #[test]
    #[should_panic(expected = "reload register is inside the allocatable pool")]
    fn a_reload_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            scratch: RegSet::range(4, 8), // swallows Reg(11)
            ..TEST_FILE
        }
        .checked();
    }

    /// The same teeth for a backend's own scratch: `fixed` is declared so this
    /// is a const-eval failure rather than an argument in a comment.
    #[test]
    #[should_panic(expected = "fixed backend scratch register is inside the allocatable pool")]
    fn a_fixed_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            fixed: &[Reg(5)], // inside TEST_FILE's 4..8 pool
            ..TEST_FILE
        }
        .checked();
    }

    /// A set is not a range: the pool may hold registers on both sides of a
    /// reserved one, which is the whole reason `RegSet` replaced base+count.
    #[test]
    fn the_pool_may_straddle_a_reserved_register() {
        let straddling = RegisterFile {
            scratch: RegSet::of(&[Reg(4), Reg(5), Reg(6), Reg(7), Reg(14), Reg(15)]),
            ..TEST_FILE
        }
        .checked();
        assert_eq!(straddling.scratch.len(), 6);
        let regs: alloc::vec::Vec<Reg> = straddling.scratch.iter().collect();
        assert_eq!(
            regs,
            alloc::vec![Reg(4), Reg(5), Reg(6), Reg(7), Reg(14), Reg(15)]
        );
    }

    /// `capped` shrinks a non-contiguous pool to its lowest members.
    #[test]
    fn capping_takes_the_lowest_members() {
        let set = RegSet::of(&[Reg(4), Reg(9), Reg(14), Reg(15)]);
        let regs: alloc::vec::Vec<Reg> = set.take(2).iter().collect();
        assert_eq!(regs, alloc::vec![Reg(4), Reg(9)]);
        assert_eq!(set.take(99).len(), 4, "capping never grows the pool");
    }

    #[test]
    #[should_panic(expected = "input register is inside the allocatable pool")]
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
