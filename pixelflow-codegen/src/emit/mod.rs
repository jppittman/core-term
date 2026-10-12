//! JIT code emission for expression DAGs.
//!
//! ## Phases
//!
//! A schedule becomes machine code in three phases, run by `compile_native`
//! on the backend the host's tier selected:
//! - **Selection** (`select`) walks the scoped schedule and builds a
//!   `Function`: blocks of instructions over values, which no register has
//!   been chosen for.
//! - **Allocation** (`regalloc`) chooses every register and frame slot, and
//!   inserts the copies, spills and reloads its choices need.
//! - **Assembly** (`asm`) lays the blocks out and encodes them.
//!
//! The backends (`avx2`, `avx512`, `aarch64`) are `IsaBackend`s: what a
//! target contributes is its register file and its instructions, and nothing
//! else about a target reaches selection or allocation.
//!
//! ## The loop nest
//!
//! There is no collapse scaffold. A kernel reaches this module already
//! wrapped in the lattice's folds — rows, columns, lanes — by
//! [`pixelflow_ir::passes::legalize`], so the emitted function is one scope
//! (what runs once per call) with folds nested in it, every one a loop of
//! blocks: preheader, head, latch, exit. The lane fold is the one the `Write`
//! inside names as its lane, and it is executed *by lanes*: its binder is the
//! constant `[0, 1, …, L−1]`, it has no counter and no back edge, and its body
//! is inlined into the column fold's
//! (docs/plans/2026-09-16-collapse-is-a-fold.md §2.2).
//!
//! ## Spilling
//!
//! Values the register file cannot hold go to frame slots, which the allocator
//! leases from the function's `Frame`. A spilled
//! value is stored right after its definition, which every path that reads it
//! has run, and reloaded into a register the allocator chose for the reload.

/// The one way a backend refuses an op.
///
/// Reaching this is never "the target cannot do that". [`pixelflow_ir::passes::legalize`]
/// leaves only ops from the backend-legal set, and every backend owes an
/// encoding for all of them — so arriving here means the pipeline was bypassed
/// or this backend is incomplete. Both are bugs in the compiler rather than
/// facts about the kernel, and neither is something a caller could act on: the
/// only callers that ever saw the old `Err` immediately `.expect()`ed it.
///
/// So it panics, loudly and at the point of failure, naming the op and the
/// backend that owes it. Development gets a stack trace pointing at the missing
/// match arm instead of a `&'static str` surfacing three frames up.
#[cold]
#[inline(never)]
fn unimplemented_op(backend: &str, op: impl core::fmt::Debug) -> ! {
    panic!(
        "{backend} has no encoding for {op:?} — `passes::legalize` leaves only \
         backend-legal ops, so this is a missing implementation or a bypassed \
         pipeline, not a bad kernel"
    )
}

mod aarch64;
mod asm;
mod avx2;
mod avx512;
mod build;
#[cfg(test)]
mod coverage;
mod encoded;
mod executable;
mod regalloc;
mod register_file;
mod select;
mod traffic;
mod x86_64;

// The whole public surface: the driver (it lives in pipeline.rs, and keeps
// this path), what it hands back, and the counts inside that.
pub use crate::pipeline::compile;
pub use executable::CompiledKernel;
pub use traffic::{EmitTraffic, ScopeTraffic};

use asm::{Encoding, Label, Labels, Patch};
use encoded::EncodedInst;
use regalloc::resource::{FrameSlot, In, InOut, Lease, Lent, Out, SlotName};
use register_file::RegisterFile;

use pixelflow_ir::kind::OpKind;

use crate::program::IfArm;
use crate::program::Scope;
use crate::program::ScopedSchedule;

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::error::CompileError;
use crate::isa::Isa;

/// The constant pool's alignment: one NEON pool entry, so every `LDR Qt` from
/// it is an aligned vector load. x86's four-byte entries need no alignment and
/// take this one for the cache line. The padding that reaches it from the
/// last instruction follows the code's length, which is why a kernel's
/// trailing bytes can differ between two allocations of it by less than this.
const CONST_POOL_ALIGN: usize = 16;

/// What a value is at the machine level: the type an instruction's operand
/// field is declared with. `Pointer` and `Integer` live in the same file and
/// are different classes, so a base address cannot be handed a `row * pitch`
/// product.
///
/// Sealed: these five are all there are.
trait Class: sealed::Class + Copy + 'static {
    /// The register file a value of this class lives in.
    type File: File;
    /// This class as data: the allocator's view of a [`Value`].
    const ID: ClassId;
}

mod sealed {
    pub trait Class {}
    pub trait File {}
}

/// A physical register file of the machine. Sealed: these four are all any
/// target here has. A backend whose machine lacks one declares it empty, and
/// no instruction of that backend has a field in it.
trait File: sealed::File + Sized + 'static {
    const ID: FileId;
    /// The lease of this file that `lent` is, whose file was chosen at run
    /// time.
    ///
    /// # Panics
    /// If `lent` is a lease of another file.
    fn lease<'a, 'm, B: IsaBackend>(lent: &'a Lent<'m, B>) -> &'a Lease<'m, B, Self>;
}

/// `ymm`, `zmm`, `v`.
enum VectorFile {}
/// The 64-bit general-purpose registers.
enum GeneralFile {}
/// AVX-512's `k1`-`k7`. `k0` is not a member: an EVEX `aaa` of 0 means "no
/// mask".
enum OpmaskFile {}
/// x86 `EFLAGS`: one member. aarch64 has none, because nothing selected there
/// reads or writes `NZCV`.
enum FlagsFile {}

/// A [`File`] as data.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FileId {
    Vector,
    General,
    Opmask,
    Flags,
}

impl sealed::File for VectorFile {}
impl sealed::File for GeneralFile {}
impl sealed::File for OpmaskFile {}
impl sealed::File for FlagsFile {}

impl File for VectorFile {
    const ID: FileId = FileId::Vector;
    fn lease<'a, 'm, B: IsaBackend>(lent: &'a Lent<'m, B>) -> &'a Lease<'m, B, Self> {
        match lent {
            Lent::Vector(lease) => lease,
            other => panic!("{other:?} is not a lease of the vector file"),
        }
    }
}
impl File for GeneralFile {
    const ID: FileId = FileId::General;
    fn lease<'a, 'm, B: IsaBackend>(lent: &'a Lent<'m, B>) -> &'a Lease<'m, B, Self> {
        match lent {
            Lent::General(lease) => lease,
            other => panic!("{other:?} is not a lease of the general file"),
        }
    }
}
impl File for OpmaskFile {
    const ID: FileId = FileId::Opmask;
    fn lease<'a, 'm, B: IsaBackend>(lent: &'a Lent<'m, B>) -> &'a Lease<'m, B, Self> {
        match lent {
            Lent::Opmask(lease) => lease,
            other => panic!("{other:?} is not a lease of the opmask file"),
        }
    }
}
impl File for FlagsFile {
    const ID: FileId = FileId::Flags;
    fn lease<'a, 'm, B: IsaBackend>(lent: &'a Lent<'m, B>) -> &'a Lease<'m, B, Self> {
        match lent {
            Lent::Flags(lease) => lease,
            other => panic!("{other:?} is not a lease of the flags file"),
        }
    }
}

/// A [`Class`] as data.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ClassId {
    Vector,
    Pointer,
    Integer,
    Opmask,
    Flags,
}

impl ClassId {
    /// The register file a value of this class lives in.
    fn file(self) -> FileId {
        match self {
            ClassId::Vector => FileId::Vector,
            ClassId::Pointer | ClassId::Integer => FileId::General,
            ClassId::Opmask => FileId::Opmask,
            ClassId::Flags => FileId::Flags,
        }
    }
}

/// One batch of `f32` lanes, which on a machine with one lane file is also a
/// mask.
#[derive(Copy, Clone, Debug)]
enum Vector {}

/// An address.
#[derive(Copy, Clone, Debug)]
enum Pointer {}

/// An integer: an index, a product.
#[derive(Copy, Clone, Debug)]
enum Integer {}

/// The condition flags.
#[derive(Copy, Clone, Debug)]
enum Flags {}

/// One batch of predicate bits, in AVX-512's mask file: a lane per bit.
#[derive(Copy, Clone, Debug)]
enum Opmask {}

impl sealed::Class for Vector {}
impl sealed::Class for Pointer {}
impl sealed::Class for Integer {}
impl sealed::Class for Flags {}
impl sealed::Class for Opmask {}

impl Class for Vector {
    type File = VectorFile;
    const ID: ClassId = ClassId::Vector;
}

impl Class for Pointer {
    type File = GeneralFile;
    const ID: ClassId = ClassId::Pointer;
}

impl Class for Integer {
    type File = GeneralFile;
    const ID: ClassId = ClassId::Integer;
}

impl Class for Flags {
    type File = FlagsFile;
    const ID: ClassId = ClassId::Flags;
}

impl Class for Opmask {
    type File = OpmaskFile;
    const ID: ClassId = ClassId::Opmask;
}

/// The classes a frame slot can hold. `Flags` cannot be stored, so the
/// allocator's own verbs ([`IsaBackend::copy`], [`spill`](IsaBackend::spill),
/// [`reload`](IsaBackend::reload)) do not accept it: calling one with `Flags` is
/// a type error.
trait Spill: Class {}

impl Spill for Vector {}
impl Spill for Pointer {}
impl Spill for Integer {}
impl Spill for Opmask {}

/// What the operands of an instruction are. An instruction enum is generic
/// over a stage and declares each field by what it does to its register.
trait Stage {
    /// A register the instruction defines.
    type Write<C: Class>;
    /// A register the instruction defines before it is done reading, so no
    /// operand may share it.
    type Early<C: Class>;
    /// A register the instruction reads.
    type Read<C: Class>;
    /// A register the instruction reads and overwrites in place.
    type Tie<C: Class>;
    /// A frame slot, as the instructions the allocator builds name one.
    type Slot;
    /// Where a branch goes.
    type Target;
    /// The frame's size, as `Enter` and `Ret` take it.
    type FrameSize;
}

/// A stage whose operands are placed: registers an encoder reads the number
/// of, a frame slot it reads the offset of, and the position a branch goes
/// to.
trait Placed: Stage {
    fn read<C: Class>(operand: &Self::Read<C>) -> u8;
    fn write<C: Class>(operand: &Self::Write<C>) -> u8;
    fn early<C: Class>(operand: &Self::Early<C>) -> u8;
    fn tie<C: Class>(operand: &Self::Tie<C>) -> u8;
    /// Bytes from the stack pointer: a `disp32`.
    fn slot(slot: &Self::Slot) -> i32;
    fn target(target: &Self::Target) -> Label;
    fn frame_size(size: &Self::FrameSize) -> u32;
}

/// A name for something computed, of class `C`: one definition, any number
/// of reads. A name, so `Copy`: names copy, resources move.
///
/// Minted by [`build::Builder`], whose `Def` is the right to define one.
#[derive(Copy, Clone, Debug)]
struct Value<C: Class> {
    id: u64,
    _class: PhantomData<C>,
}

impl<C: Class> Value<C> {
    /// The same name with its class as data.
    fn name(self) -> ValueName {
        ValueName {
            id: self.id,
            class: C::ID,
        }
    }
}

/// The same name with its class as data: the allocator's view.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ValueName {
    id: u64,
    class: ClassId,
}

impl ValueName {
    /// This name as a `Value<C>`.
    ///
    /// # Panics
    /// If the name is not of class `C`.
    fn typed<C: Class>(self) -> Value<C> {
        assert_eq!(self.class, C::ID, "{self:?} is not of the class asked for");
        Value {
            id: self.id,
            _class: PhantomData,
        }
    }
}

/// The stage selection builds at: operands are names, and the rights to
/// define them.
enum Selected {}

impl Stage for Selected {
    type Write<C: Class> = build::Def<C>;
    type Early<C: Class> = build::Early<C>;
    type Read<C: Class> = Value<C>;
    type Tie<C: Class> = build::Tie<C>;
    // Only instructions the allocator builds carry one.
    type Slot = SlotName;
    type Target = Target;
    type FrameSize = FrameSize;
}

/// The frame's size, as an operand of `Enter` and `Ret`: not known until the
/// allocator has laid the frame out.
#[derive(Copy, Clone, Debug)]
struct FrameSize;

/// How far a function's entry may move the stack pointer before it touches
/// what it moved over: one guard page is the smallest a thread's stack can
/// end in, and every host's page is at least this.
const STACK_PAGE: u32 = 4096;

/// A stage whose fields hold nothing: what [`IsaBackend::walk`] builds when
/// only the traversal matters.
enum Observed {}

impl Stage for Observed {
    type Write<C: Class> = ();
    type Early<C: Class> = ();
    type Read<C: Class> = ();
    type Tie<C: Class> = ();
    type Slot = ();
    type Target = ();
    type FrameSize = ();
}

/// The stage allocation builds: borrowed tokens, borrowed slots, block
/// arguments already placed as moves, and the frame's size.
struct Bound<'m, B>(core::convert::Infallible, PhantomData<&'m fn() -> B>);

impl<'m, B: IsaBackend> Stage for Bound<'m, B> {
    type Write<C: Class> = Out<'m, B, C>;
    type Early<C: Class> = Out<'m, B, C>;
    type Read<C: Class> = In<'m, B, C>;
    type Tie<C: Class> = InOut<'m, B, C>;
    type Slot = &'m FrameSlot;
    type Target = Label;
    type FrameSize = u64;
}

impl<'m, B: IsaBackend> Placed for Bound<'m, B> {
    fn read<C: Class>(operand: &In<'m, B, C>) -> u8 {
        operand.number()
    }
    fn write<C: Class>(operand: &Out<'m, B, C>) -> u8 {
        operand.number()
    }
    fn early<C: Class>(operand: &Out<'m, B, C>) -> u8 {
        operand.number()
    }
    fn tie<C: Class>(operand: &InOut<'m, B, C>) -> u8 {
        operand.number()
    }
    fn slot(slot: &&'m FrameSlot) -> i32 {
        i32::try_from(slot.offset()).expect("MAX_FRAME bounds every slot offset")
    }
    fn target(target: &Label) -> Label {
        *target
    }
    fn frame_size(size: &u64) -> u32 {
        u32::try_from(*size).expect("MAX_FRAME bounds the frame")
    }
}

/// How an instruction is rebuilt at stage `T`, one operand at a time. Each
/// method is the only way to turn its kind of field into `T`'s, so `walk` can
/// neither misreport an access nor skip a register.
trait Rebind<T: Stage> {
    fn read<C: Class>(&mut self, v: Value<C>) -> T::Read<C>;
    fn write<C: Class>(&mut self, d: &build::Def<C>) -> T::Write<C>;
    fn early<C: Class>(&mut self, d: &build::Early<C>) -> T::Early<C>;
    fn tie<C: Class>(&mut self, t: &build::Tie<C>) -> T::Tie<C>;
    fn slot(&mut self, s: SlotName) -> T::Slot;
    fn target(&mut self, t: &Target) -> T::Target;
    fn frame_size(&mut self) -> T::FrameSize;
}

/// How an instruction touches a register operand. A `Tie` yields two
/// entries: `Read`, and a `Tied` write naming that read's entry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Access {
    Read,
    Write,
    /// Written before the instruction is done reading: no read shares its
    /// register.
    Early,
    /// Overwrites the read at this index of the operand list, in place.
    Tied {
        read: usize,
    },
}

impl Access {
    /// Whether the operand's value must already be defined.
    fn reads(self) -> bool {
        matches!(self, Access::Read)
    }
}

/// One operand, as the allocator and the CFG read it. Immediates are typed
/// fields no phase before encoding reads, and an address is a composite
/// whose registers are listed here as reads.
#[derive(Debug)]
enum Operand {
    Reg {
        value: ValueName,
        access: Access,
    },
    /// A branch target and the arguments it passes. A block's successors are
    /// the `Target` operands of its last instruction.
    Target(Target),
    /// A frame slot: only on instructions the allocator inserted.
    Frame(SlotName),
}

/// Every operand of `inst`, in walk order: `walk` at a stage whose fields are
/// `()`, recording as it goes. The operand list and the binding are one
/// traversal, so they cannot disagree.
fn operands<B: IsaBackend>(inst: &B::Inst<Selected>) -> Vec<Operand> {
    struct Recorder(Vec<Operand>);

    impl Recorder {
        fn reg<C: Class>(&mut self, value: Value<C>, access: Access) {
            self.0.push(Operand::Reg {
                value: value.name(),
                access,
            });
        }
    }

    impl Rebind<Observed> for Recorder {
        fn read<C: Class>(&mut self, v: Value<C>) {
            self.reg(v, Access::Read);
        }
        fn write<C: Class>(&mut self, d: &build::Def<C>) {
            self.reg(d.value(), Access::Write);
        }
        fn early<C: Class>(&mut self, d: &build::Early<C>) {
            self.reg(d.value(), Access::Early);
        }
        fn tie<C: Class>(&mut self, t: &build::Tie<C>) {
            let read = self.0.len();
            self.reg(t.read(), Access::Read);
            self.reg(t.write(), Access::Tied { read });
        }
        fn slot(&mut self, s: SlotName) {
            self.0.push(Operand::Frame(s));
        }
        fn target(&mut self, t: &Target) {
            self.0.push(Operand::Target(t.clone()));
        }
        fn frame_size(&mut self) {}
    }

    let mut recorder = Recorder(Vec::new());
    let _ = B::walk(inst, &mut recorder);
    recorder.0
}

/// A label operand before allocation: where to go, and the values for the
/// target's parameters, in order.
#[derive(Clone, Debug)]
struct Target {
    label: Label,
    args: Vec<ValueName>,
}

/// An instruction and its [`operands`], computed once when selection pushed
/// it. The list is the instruction's own, so no entry holds one without the
/// other.
struct Pushed<I> {
    inst: I,
    operands: Vec<Operand>,
}

/// A label, the values defined on entry to it, and its instructions.
struct Block<I> {
    label: Label,
    /// Values defined on entry: what a phi is.
    params: Vec<ValueName>,
    insts: Vec<I>,
    /// The nest scope this block's code belongs to, for traffic and trip
    /// weighting.
    scope: Scope,
}

/// The ABI's three arguments: the entry block's parameters, with their
/// classes.
#[derive(Copy, Clone, Debug)]
struct Entry {
    ctx: Value<Pointer>,
    out: Value<Pointer>,
    pitch: Value<Integer>,
}

/// One surviving fold's loop: a contiguous run of blocks from `head` to the
/// one block that branches back to it.
struct Loop {
    head: Label,
    /// The enclosing loop, as an index into [`Function::loops`].
    parent: Option<usize>,
    /// How many times its body runs per call: its own trip count times its
    /// parent's.
    trips: u64,
}

/// The pool section: its label, then each entry with its own label, in order.
struct Constants<K> {
    label: Label,
    entries: Vec<(Label, K)>,
}

/// One pool entry: its label (x86 reads it RIP-relative) and its byte offset
/// in the section (aarch64 reads `[pool base, #offset]`).
#[derive(Copy, Clone, Debug)]
struct Constant {
    label: Label,
    offset: u64,
}

/// A whole kernel, in layout order.
///
/// Invariants, established by selection and asserted by
/// [`build::Builder::finish`]:
/// - `blocks[0]` is the entry. Its parameters are `entry`'s three.
/// - Every minted label is bound by exactly one block or data item.
/// - Only a block's last instruction has `Target` operands.
/// - Exactly one block has none: the last.
/// - Every `Target`'s arguments match its block's parameters in arity and
///   class.
/// - A conditional branch has two targets, `taken` then `next`. `next` is the
///   following block and has no parameters.
/// - Every backward branch comes from the last block of a loop in `loops`
///   and targets that loop's head, and a loop has one.
/// - A forward branch that skips blocks stays in one scope and does not enter
///   or leave a loop.
///
/// **Dominance is relaxed in one way, and only one.**
/// - A value an `If` arm defines is read by that `If`'s blend.
/// - On the path where a uniform mask skipped the arm, the blend still runs,
///   and reads a register nothing wrote there. It takes every lane from the
///   arm that ran, so that register is never observed.
/// - So a definition dominates every read that *observes* it, not every
///   syntactic read.
/// - Interval liveness never carries the value above its definition, and join
///   invariant 2 takes its location from the path that defines it.
/// - No phase inserts a phi for it.
struct Function<B: IsaBackend> {
    blocks: Vec<Block<Pushed<B::Inst<Selected>>>>,
    entry: Entry,
    loops: Vec<Loop>,
    constants: Constants<B::Constant>,
    labels: Labels,
    /// The class of every value, by id: the allocator numbers the values it
    /// mints from where this ends.
    classes: Vec<ClassId>,
    /// Scheduled ops the driver selected, per scope (the body, then the folds):
    /// what [`EmitTraffic`] reports as a scope's `instructions`.
    scheduled: Vec<u64>,
}

/// Everything about a target that selection needs. The driver and the
/// allocator are generic over it, and neither names a register, an opcode or an
/// encoding. The assembler is not generic over it: it imports nothing, and
/// takes the program a backend's `encode` produces.
trait IsaBackend: Sized + 'static {
    type Inst<S: Stage>;
    /// One pool entry: x86 `u32` (each load broadcasts a scalar), aarch64
    /// `[u32; 4]`. It occupies `size_of` bytes of the pool section.
    type Constant: Copy + Ord;
    /// What selection keeps for the whole function: aarch64's pool base value;
    /// nothing on x86.
    type Anchors: Default;
    /// One IR lane value as this machine holds it: `Value<Vector>` on AVX2
    /// and NEON; a `Vector` or an `Opmask` value on AVX-512.
    type Lane: Copy;
    /// The allocatable registers, by number, and where the ABI puts the
    /// arguments.
    const FILE: RegisterFile;
    /// How many pool entries an instruction can reach. aarch64:
    /// `ldr q, [p, #imm12·16]`, so 4096. x86: `i32::MAX / 4`.
    const POOL_REACH: u64;

    // Selection. Operands are values; the driver resolved them.
    fn lane(b: &mut build::Builder<Self>, op: LaneOp<Self>) -> Result<Self::Lane, CompileError>;
    /// The name of `lane`, for a branch to pass it and a block to take its
    /// class.
    fn lane_name(lane: Self::Lane) -> ValueName;
    /// A block parameter, read back as a lane.
    fn param_lane(param: ValueName) -> Self::Lane;
    fn context(
        b: &mut build::Builder<Self>,
        ctx: Value<Pointer>,
        slot: u64,
    ) -> Result<Value<Pointer>, CompileError>;
    fn store(b: &mut build::Builder<Self>, store: Store<Self>);
    /// End the block: go to `edges.taken` when no lane of `test.cond` selects
    /// `test.dead`'s arm; otherwise fall through to `edges.next`.
    fn branch(b: &mut build::Builder<Self>, test: Test<Self>, edges: Edges);
    /// End the block: go to `to`. A `Fallthrough` when `to` is `next`.
    fn jump(b: &mut build::Builder<Self>, to: Target, next: Label);
    /// The entry block's `Enter`, and the backend's per-function values
    /// ([`build::Builder::anchors`]: aarch64's pool base).
    fn enter(b: &mut build::Builder<Self>);
    fn ret(b: &mut build::Builder<Self>);

    // The allocator's own instructions.
    fn copy<C: Spill>(b: &mut build::Spiller<'_, Self>, src: Value<C>) -> Value<C>;
    /// Store `src` to `slot`. When `slot.offset()` does not encode: the slot's
    /// address into a fresh `Pointer`, then the store through it.
    fn spill<C: Spill>(b: &mut build::Spiller<'_, Self>, src: Value<C>, slot: &FrameSlot);
    fn reload<C: Spill>(b: &mut build::Spiller<'_, Self>, slot: &FrameSlot) -> Value<C>;

    /// Whether the allocator may place this instruction again where a read
    /// needs its definition, instead of storing it. It must have no effect,
    /// write one register first, of any class but `Flags`, and otherwise only
    /// read. The allocator asserts that, and places what it reads first (a
    /// pool load through aarch64's base register reads the base, which is
    /// itself one): pool loads, `movi`/`fmov` immediates, the zero and
    /// all-ones idioms.
    fn rematerializable(inst: &Self::Inst<Selected>) -> bool;

    /// Whether `inst` reads a constant from the pool: the memory read
    /// [`EmitTraffic`] counts as a remat. The zero and all-ones idioms bring a
    /// constant into a register without reading memory.
    fn reads_pool<S: Stage>(inst: &Self::Inst<S>) -> bool;

    /// Rebuild `inst` at stage `T`, visiting each operand once, in field
    /// order. The only per-instruction traversal.
    fn walk<T: Stage>(inst: &Self::Inst<Selected>, f: &mut impl Rebind<T>) -> Self::Inst<T>;

    /// Encode one allocated instruction: its bytes, and the label fields in
    /// them.
    fn encode(inst: &Self::Inst<Bound<'_, Self>>, out: &mut Encoding<'_>);

    /// A pool entry as the bytes the data section holds.
    fn constant_bytes(constant: Self::Constant) -> Vec<u8>;
}

/// An operation producing one lane value, with its operands already lanes.
/// Comparisons and `BitAnd`/`BitOr` are `Binary`: which file their result
/// lives in is the backend's choice.
enum LaneOp<B: IsaBackend> {
    Const(f32),
    /// `[0, 1, …, L−1]`.
    Lanes,
    Unary(OpKind, B::Lane),
    Binary(OpKind, B::Lane, B::Lane),
    /// `a·b + c`: one rounding, every target.
    MulAdd(B::Lane, B::Lane, B::Lane),
    /// The `If`'s lane-varying path.
    Blend {
        cond: B::Lane,
        if_true: B::Lane,
        if_false: B::Lane,
    },
    Shift(OpKind, B::Lane, u8),
    /// `base[index]`, one element per lane.
    Gather {
        base: Value<Pointer>,
        index: B::Lane,
    },
    /// `base[index]` for lane 0's index, in every lane.
    Broadcast {
        base: Value<Pointer>,
        index: B::Lane,
    },
    Uniform {
        base: Value<Pointer>,
        element: u64,
    },
}

/// The lattice's effect: `value`'s first `lanes` lanes at
/// `out + 4·(row·pitch + col)`.
struct Store<B: IsaBackend> {
    out: Value<Pointer>,
    pitch: Value<Integer>,
    row: B::Lane,
    col: B::Lane,
    value: B::Lane,
    lanes: u32,
}

/// A branch condition: the lane to test, and the arm that is dead when no
/// lane of it is set.
struct Test<B: IsaBackend> {
    cond: B::Lane,
    dead: IfArm,
}

/// Where a conditional branch goes: `taken`, or `next`, the block laid out
/// right after.
struct Edges {
    taken: Target,
    next: Label,
}

/// What [`compile`] hands back: the code, and what was emitted, counted —
/// the static half of what a measurement harness prices a kernel by.
pub struct CompileResult {
    /// The executable code.
    pub code: CompiledKernel,
    /// Frame slots minted: every slot a value or block parameter lives in.
    pub spill_count: u64,
    /// The frame's size in bytes, capped at 2 MiB: the encoders address a slot
    /// through a disp32 / imm12.
    pub frame_bytes: u64,
    /// Values computed outside a loop and read inside it: the values that
    /// live to a loop's latch.
    pub hoisted_values: u64,
    /// What was emitted, per scope of the nest — the static half of a cost
    /// model's inputs. Counted, never optimized: see [`EmitTraffic`].
    pub traffic: EmitTraffic,
}

// =============================================================================
// The one place a target decides anything
// =============================================================================

/// Drive `program` to a kernel on the backend the host's CPU selected.
///
/// Every backend compiles on every host — emission is a pure function of the
/// program into a `Vec<u8>`, and an x86 machine is perfectly capable of
/// computing NEON instruction words. So the target does not decide which
/// backends *exist*; it decides which one is *instantiated*, here, from the
/// tier [`crate::isa::detect`] read off CPUID at startup. Each arm
/// monomorphizes [`compile_on`] against one concrete backend: static dispatch
/// inside the compile, no `dyn`, no vtable. The one `match` is this, and it
/// runs once per kernel, not once per instruction.
///
/// `detect` answers `Neon` only on aarch64 and `Avx2`/`Avx512` only on x86-64,
/// so no arm carries a `cfg`: the other architecture's backend is typechecked
/// and unit-tested on this host, and its arm is never taken. There is no tier
/// below AVX2+FMA on x86-64
/// (docs/plans/2026-09-22-the-isa-is-decided-at-startup.md).
///
/// Genuinely host-bound code lives in [`executable`] (the `KernelFn` ABI types
/// and the `mmap`/`mprotect` that makes bytes callable) and nowhere else.
pub(crate) fn compile_native(program: ScopedSchedule) -> Result<CompileResult, CompileError> {
    match crate::isa::detect() {
        Isa::Avx2 => compile_on::<avx2::Avx2>(&program),
        Isa::Avx512 => compile_on::<avx512::Avx512>(&program),
        Isa::Neon => compile_on::<aarch64::Neon>(&program),
    }
}

/// The selection pipeline: select, allocate, assemble.
fn compile_on<B: IsaBackend>(program: &ScopedSchedule) -> Result<CompileResult, CompileError> {
    let function = select::select::<B>(program)?;
    let pool = regalloc::resource::Pool::<B>::mint();
    let mut frame = regalloc::resource::Frame::empty(B::FILE.vector_bytes());
    let mut allocated = regalloc::local::allocate(function, &pool, &mut frame)?;
    let program = allocated.program();
    let assembled = asm::assemble(&program, |emitted, out| B::encode(&emitted.inst, out));
    let traffic = EmitTraffic::of(&allocated, &assembled);
    let code = unsafe { executable::CompiledKernel::from_code(&assembled.code)? };
    Ok(CompileResult {
        code,
        spill_count: allocated.slots,
        frame_bytes: allocated.frame_bytes,
        hoisted_values: allocated.hoisted,
        traffic,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tests::{BYTES_PER_LANE, schedule_for};
    use crate::program::ScheduledOp;
    use pixelflow_ir::LatticeShape;
    use pixelflow_ir::arena::{ExprArena, ExprId, UniformId};

    /// `a` compiled for backend `B` on whatever host runs the test: the
    /// schedule lowered at `B`'s own lane count, scoped the way a compile
    /// scopes it.
    fn compile_for<B: IsaBackend>(
        a: &ExprArena,
        root: ExprId,
        shape: LatticeShape,
    ) -> Result<CompileResult, CompileError> {
        let lanes = B::FILE.vector_bytes() as u32 / BYTES_PER_LANE;
        compile_on::<B>(&ScopedSchedule::from_schedule(schedule_for(
            a, root, shape, lanes,
        )))
    }

    /// Lanes in one SIMD batch at the tier this host selected.
    fn lanes() -> usize {
        crate::jit_vector_bytes() / core::mem::size_of::<f32>()
    }

    /// One sample, so the lattice's origin *is* the point the kernel is
    /// evaluated at: what a test about arithmetic rather than about the loop
    /// nest compiles for.
    const POINT: LatticeShape = LatticeShape::new([1, 1]);

    /// One full batch of one row: `x` runs `x0 .. x0 + lanes()`, which is
    /// what a test about per-lane behaviour needs.
    fn batch() -> LatticeShape {
        LatticeShape::new([lanes() as u32, 1])
    }

    /// Each backend's register file is the width its kernels are legalized
    /// and framed at, and the width is the ISA's: a `ymm` is 32 bytes, a
    /// `zmm` 64 and a NEON `q` register 16, by definition.
    #[test]
    fn every_backends_vector_width_is_its_tiers() {
        assert_eq!(avx2::Avx2::FILE.vector_bytes(), 32);
        assert_eq!(avx512::Avx512::FILE.vector_bytes(), 64);
        assert_eq!(aarch64::Neon::FILE.vector_bytes(), 16);
        // And the width a compile legalizes at (`jit_vector_bytes`) is the
        // width of the backend it instantiates.
        let native = match crate::isa::detect() {
            Isa::Avx2 => avx2::Avx2::FILE.vector_bytes(),
            Isa::Avx512 => avx512::Avx512::FILE.vector_bytes(),
            Isa::Neon => aarch64::Neon::FILE.vector_bytes(),
        };
        assert_eq!(native as usize, crate::jit_vector_bytes());
    }

    /// Run the collapse `code` is over a plane of exactly `shape`'s extent,
    /// and hand the plane back.
    ///
    /// `buffers` binds the arena's buffer slots, `uniforms` its uniform
    /// block, and `(x, y)` is where the lattice's sample `(0, 0)` lies. The
    /// pitch is the width, so a sample reads as `out[row * width + col]`.
    fn collapse_into(
        code: &executable::CompiledKernel,
        buffers: &[*const f32],
        uniforms: &[f32],
        (x, y): (f32, f32),
        shape: LatticeShape,
    ) -> Vec<f32> {
        let [width, height] = shape.extent().map(|n| n as usize);
        let mut out = alloc::vec![f32::NAN; width * height];
        let origin = [x, y];
        let mut ctx: Vec<*const f32> = buffers.to_vec();
        ctx.push(uniforms.as_ptr());
        ctx.push(origin.as_ptr());
        // SAFETY: `ctx` binds every buffer the arena declared, then the
        // uniform block and the origin block, each live for the call; `out`
        // is the whole plane the kernel was compiled at.
        unsafe {
            code.call(ctx.as_ptr(), out.as_mut_ptr(), width);
        }
        out
    }

    /// Evaluate a kernel compiled at [`POINT`] at one lattice point.
    fn eval_point(code: &executable::CompiledKernel, x: f32, y: f32) -> f32 {
        collapse_into(code, &[], &[], (x, y), POINT)[0]
    }

    /// Evaluate a kernel compiled at [`batch`]: one row of `lanes()` samples
    /// from `(x, y)`.
    fn eval_batch(
        code: &executable::CompiledKernel,
        buffers: &[*const f32],
        uniforms: &[f32],
        x: f32,
        y: f32,
    ) -> Vec<f32> {
        collapse_into(code, buffers, uniforms, (x, y), batch())
    }

    /// The route that *does* work: the compile entry expands the reference
    /// before it schedules, so a kernel composed by reference emits exactly
    /// what the spliced composition emits.
    #[test]
    fn a_reference_compiles_through_the_entry_point() {
        let body = pixelflow_ir::Kernel::x().mul(&pixelflow_ir::Kernel::constant(3.0));
        let named = body.by_ref().add(&pixelflow_ir::Kernel::y());
        let direct = body.add(&pixelflow_ir::Kernel::y());
        let (n_arena, n_root) = named.parts();
        let (d_arena, d_root) = direct.parts();
        let named_code = compile(n_arena, n_root, POINT).expect("a named kernel compiles");
        let direct_code = compile(d_arena, d_root, POINT).expect("and so does the spliced one");
        for (x, y) in [(0.0f32, 0.0f32), (1.5, -2.0), (-3.25, 7.5)] {
            let want = eval_point(&direct_code.code, x, y);
            let got = eval_point(&named_code.code, x, y);
            assert_eq!(got, want, "at ({x}, {y})");
        }
    }

    /// `passes::legalize` leaves every `Reduce` standing, so a fold reaches
    /// the emitter as a loop; this builds one by hand and compiles it.
    ///
    /// `SUM` over four terms, matched against the closed form
    /// `⊕_{i<4}(X+i) = 4X + 6`; `MIN` over the same range, matched against
    /// whichever term is smallest for a given `X` — two different monoids,
    /// so a bug specific to one algebra's identity or combine opcode has
    /// somewhere to show up.
    #[test]
    fn a_surviving_reduce_compiles_and_runs() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let binder = Binder::from_slot(0).expect("slot 0 exists");

        // sum_{i=0}^{3} (X + i) = 4*X + (0+1+2+3) = 4*X + 6
        let mut sum_arena = ExprArena::new();
        let x = sum_arena.push_var(0);
        let i = sum_arena.push_var(binder.var());
        let body = sum_arena.push_binary(OpKind::Add, x, i);
        let sum_fold = Fold::new(Monoid::SUM, binder, 0..4);
        let sum_root = sum_arena.push_reduce(sum_fold, body);
        let sum_code =
            compile(&sum_arena, sum_root, POINT).expect("a surviving SUM Reduce compiles");

        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&sum_code.code, x, 0.0);
            let want = 4.0 * x + 6.0;
            assert_eq!(got, want, "SUM at x={x}");
        }

        // min_{i=0}^{3} (X - i) = X - 3, always the last term.
        let mut min_arena = ExprArena::new();
        let x = min_arena.push_var(0);
        let i = min_arena.push_var(binder.var());
        let body = min_arena.push_binary(OpKind::Sub, x, i);
        let min_fold = Fold::new(Monoid::MIN, binder, 0..4);
        let min_root = min_arena.push_reduce(min_fold, body);
        let min_code =
            compile(&min_arena, min_root, POINT).expect("a surviving MIN Reduce compiles");

        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&min_code.code, x, 0.0);
            let want = x - 3.0;
            assert_eq!(got, want, "MIN at x={x}");
        }
    }

    /// A fold whose result is *not* the kernel's own root, and whose body
    /// shares an invariant leaf with code outside it.
    ///
    /// Both are the two ways `extract_folds` could get this wrong: reading
    /// the `Reduce` as an ordinary operand (most kernels' folds feed further
    /// arithmetic, unlike the previous test's bare fold) exercises the same
    /// slot override every other spilled value's reader already goes
    /// through; the shared `shared = X * X` node exercises that removing a
    /// fold's body must be *variance*-driven, not *reachability*-driven —
    /// reachability would drop `shared`'s def from the body's own schedule
    /// too and orphan its other reader.
    ///
    /// `reduce = sum_{i=0}^{2}(shared + i) = 3*shared + 3`;
    /// `root = reduce + shared + Y = 4*shared + 3 + Y`.
    #[test]
    fn a_surviving_reduce_shares_a_leaf_and_feeds_further_arithmetic() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let shared = a.push_binary(OpKind::Mul, x, x);
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(binder.var());
        let body = a.push_binary(OpKind::Add, shared, i);
        let fold = Fold::new(Monoid::SUM, binder, 0..3);
        let reduce = a.push_reduce(fold, body);
        let plus_shared = a.push_binary(OpKind::Add, reduce, shared);
        let root = a.push_binary(OpKind::Add, plus_shared, y);

        let code = compile(&a, root, POINT).expect("a fold that feeds further arithmetic compiles");

        for (x, y) in [(0.0f32, 0.0f32), (2.0, 1.0), (-1.5, 3.0), (10.0, -4.0)] {
            let got = eval_point(&code.code, x, y);
            let want = 4.0 * x * x + 3.0 + y;
            assert_eq!(got, want, "at (x={x}, y={y})");
        }
    }

    /// A surviving fold's own per-iteration recompute of a shared invariant
    /// leaf must not clobber the outer scope's copy of that same value.
    ///
    /// Same DAG as `a_surviving_reduce_shares_a_leaf_and_feeds_further_arithmetic`
    /// (`shared` read once inside the fold's body, once again after it),
    /// but pushed to the arena in the *interleaved* order unrolling a fold
    /// produces (const, add, const, add, const, add) rather than all three
    /// constants first. That reordering alone, with no change to the DAG's shape,
    /// used to compute `24` instead of `21`: the outer scope's own copy of
    /// `shared` shared a register with the fold's own per-iteration
    /// recompute of it, so its last iteration's write clobbered it.
    ///
    /// `shared = (X+0)+(X+1)+(X+2) = 3X+3`;
    /// `reduce = sum_{i=0}^{3}(shared + i) = 4*shared + 6`;
    /// `root = shared + reduce = 5*shared + 6`.
    #[test]
    fn a_surviving_reduces_own_recompute_of_a_shared_leaf_does_not_clobber_the_outer_copy() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let c0 = a.push_const(0.0);
        let t0 = a.push_binary(OpKind::Add, x, c0);
        let c1 = a.push_const(1.0);
        let t1 = a.push_binary(OpKind::Add, x, c1);
        let c2 = a.push_const(2.0);
        let t2 = a.push_binary(OpKind::Add, x, c2);
        let s01 = a.push_binary(OpKind::Add, t0, t1);
        let shared = a.push_binary(OpKind::Add, s01, t2);
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(binder.var());
        let body = a.push_binary(OpKind::Add, shared, i);
        let fold = Fold::new(Monoid::SUM, binder, 0..4);
        let reduce = a.push_reduce(fold, body);
        let root = a.push_binary(OpKind::Add, shared, reduce);

        let code = compile(&a, root, POINT).expect("interleaved shared-leaf order compiles");

        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let shared_val = 3.0 * x + 3.0;
            let want = shared_val + (4.0 * shared_val + 6.0);
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// A fold's spill slots do not alias its parent's.
    ///
    /// The frame used to be sized as a plain `max` over the nest's scopes,
    /// with every one of them handing out slots from offset 0. That is right
    /// for the two collapse prologues and the body — they run one after
    /// another, each parking what the next needs in a *hoist* slot above the
    /// frame, so an earlier scope's own slots are dead by the time a later one
    /// opens. A fold is the scope that is not like that: its loop runs in the
    /// middle of its parent's schedule, and the parent's spilled values are
    /// live across it. Sharing a base meant the fold's body wrote its own
    /// temporaries over them.
    ///
    /// It took real pressure on both sides to see: for a glyph, a 2,472-def
    /// fold body over a 2,130-def parent, which is every slot the parent had,
    /// and the whole atlas came out blank. So this builds that shape rather
    /// than a minimal fold — `K` values live across the loop and `K` more
    /// inside it, all defined before anything consumes them, which is what
    /// forces both scopes past the register file and into slots.
    ///
    /// `p_k = X + k`; `B(i) = Σ_j (i + j)·p_j`; `reduce = Σ_{i<R} B(i)`;
    /// `root = Σ_k (p_k + reduce)`.
    #[test]
    fn a_folds_spill_slots_do_not_alias_its_parents() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        const K: usize = 20;
        const R: u32 = 3;

        // A balanced sum, pushed level by level: every leaf is defined before
        // the first combine, so they are all live at once.
        fn tree_sum(a: &mut ExprArena, mut ids: alloc::vec::Vec<ExprId>) -> ExprId {
            while ids.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                for pair in ids.chunks(2) {
                    next.push(match pair {
                        [l, r] => a.push_binary(OpKind::Add, *l, *r),
                        [only] => *only,
                        _ => unreachable!("chunks(2) yields 1 or 2"),
                    });
                }
                ids = next;
            }
            ids[0]
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        // Live across the loop: defined here, consumed only past the `Reduce`.
        let p: alloc::vec::Vec<ExprId> = (0..K)
            .map(|k| {
                let c = a.push_const(k as f32);
                a.push_binary(OpKind::Add, x, c)
            })
            .collect();

        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(binder.var());
        // Live inside the loop, and sharing every `p_j` with the parent.
        let q: alloc::vec::Vec<ExprId> = (0..K)
            .map(|j| {
                let c = a.push_const(j as f32);
                let ij = a.push_binary(OpKind::Add, i, c);
                a.push_binary(OpKind::Mul, ij, p[j])
            })
            .collect();
        let body = tree_sum(&mut a, q);
        let reduce = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..R), body);

        let joined: alloc::vec::Vec<ExprId> = p
            .iter()
            .map(|&pk| a.push_binary(OpKind::Add, pk, reduce))
            .collect();
        let root = tree_sum(&mut a, joined);

        let code = compile(&a, root, POINT).expect("a fold under register pressure compiles");

        for xv in [0.0f32, 1.0, -2.5, 7.0] {
            let got = eval_point(&code.code, xv, 0.0);
            let pv = |k: usize| xv + k as f32;
            let reduce_v: f32 = (0..R)
                .map(|iv| (0..K).map(|j| (iv as f32 + j as f32) * pv(j)).sum::<f32>())
                .sum();
            let want: f32 = (0..K).map(|k| pv(k) + reduce_v).sum();
            // Summation order differs from the emitted tree's, so this is a
            // tolerance on rounding — the bug it guards was off by the whole
            // value, not the last bits.
            let tol = want.abs() * 1e-4 + 1e-3;
            assert!(
                (got - want).abs() <= tol,
                "at x={xv}: got {got}, want {want}"
            );
        }
    }

    /// A fold inside a fold's body is a loop inside a loop.
    ///
    /// `extract_folds` used to refuse this shape (one level, with
    /// `expand_nested_reduce` unrolling the inner fold ahead of it); it
    /// carves the inner fold out of the outer fold's closure now, the same
    /// way it carves the outer one out of the body. The inner body reads
    /// *both* binders — a contraction — which is what exercises the outer
    /// binder staying in its register across the inner loop, and the inner
    /// scope being handed the outer binder's park.
    ///
    /// `inner(j) = Σ_{i<3} (X + i·j) = 3X + 3j`;
    /// `root = Σ_{j<2} inner(j) = 6X + 3`.
    #[test]
    fn a_reduce_inside_a_reduce_compiles_and_runs() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let inner_binder = Binder::from_slot(0).expect("slot 0 exists");
        let outer_binder = Binder::from_slot(1).expect("slot 1 exists");
        let i = a.push_var(inner_binder.var());
        let j = a.push_var(outer_binder.var());
        let ij = a.push_binary(OpKind::Mul, i, j);
        let inner_body = a.push_binary(OpKind::Add, x, ij);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, inner_binder, 0..3), inner_body);
        let root = a.push_reduce(Fold::new(Monoid::SUM, outer_binder, 0..2), inner);

        let code = compile(&a, root, POINT).expect("a fold inside a fold compiles");
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let want = 6.0 * x + 3.0;
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// Three deep, every level's binder read at the bottom, and the inner
    /// fold's result feeding further arithmetic in its parent's body rather
    /// than being the parent's whole body.
    ///
    /// `Σ_{k<2} (k + Σ_{j<2} (j + Σ_{i<2} (X + i + j + k)))`: the innermost
    /// sums to `2X + 1 + 2j + 2k`, the middle to `Σ_j (2X + 1 + 2k + 3j)` =
    /// `4X + 5 + 4k`, the outer to `Σ_k (4X + 5 + 5k)` = `8X + 15`.
    #[test]
    fn a_reduce_three_deep_compiles_and_runs() {
        let (a, root) = three_deep_contraction();
        let code = compile(&a, root, POINT).expect("a fold three deep compiles");
        assert_three_deep(&code);
    }

    /// The three-deep contraction of the test above, as an arena.
    fn three_deep_contraction() -> (ExprArena, ExprId) {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let bj = Binder::from_slot(1).expect("slot 1 exists");
        let bk = Binder::from_slot(2).expect("slot 2 exists");
        let i = a.push_var(bi.var());
        let j = a.push_var(bj.var());
        let k = a.push_var(bk.var());
        let xi = a.push_binary(OpKind::Add, x, i);
        let xij = a.push_binary(OpKind::Add, xi, j);
        let xijk = a.push_binary(OpKind::Add, xij, k);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..2), xijk);
        let j_inner = a.push_binary(OpKind::Add, j, inner);
        let middle = a.push_reduce(Fold::new(Monoid::SUM, bj, 0..2), j_inner);
        let k_middle = a.push_binary(OpKind::Add, k, middle);
        let root = a.push_reduce(Fold::new(Monoid::SUM, bk, 0..2), k_middle);
        (a, root)
    }

    /// `8X + 15`, at four points.
    fn assert_three_deep(code: &CompileResult) {
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let want = 8.0 * x + 15.0;
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// Two sibling folds binding the same slot read one `Var` node, and
    /// each reads its own counter through it.
    ///
    /// A glyph's distance and winding folds are this shape, and the e-graph
    /// hash-conses their binder leaf into one node. A binder slot keyed by
    /// that node would be one slot for two loops: the fold allocated last
    /// would own it, and the other would step its own counter but test and
    /// read the sibling's.
    ///
    /// `Σ_{i<3} (X + i) + Σ_{i<2} 2i = (3X + 3) + 2 = 3X + 5`.
    #[test]
    fn sibling_folds_sharing_a_binder_node_read_their_own_counters() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(bi.var());
        let xi = a.push_binary(OpKind::Add, x, i);
        let left = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..3), xi);
        let two = a.push_const(2.0);
        let two_i = a.push_binary(OpKind::Mul, two, i);
        let right = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..2), two_i);
        let root = a.push_binary(OpKind::Add, left, right);

        let code = compile(&a, root, POINT).expect("sibling folds compile");
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let want = 3.0 * x + 5.0;
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// A fold invariant across the lattice is the **body's** own fold — it
    /// opens once per call, outside the lattice's row, column and lane folds
    /// — and its result reaches the scopes inside from wherever its loop left
    /// it.
    ///
    /// The `Reduce` arm ends its def early, past the hand-off, so a carried
    /// fold result used to reach the scopes inside in a register nothing had
    /// loaded. A fold result is never a root now (`stays_put`), so nothing
    /// carries one: the scopes inside read it from its accumulator slot.
    /// `X + Σ_{i<3} 2i = X + 6`.
    #[test]
    fn a_lattice_invariant_fold_is_the_bodys_own() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(bi.var());
        let two = a.push_const(2.0);
        let two_i = a.push_binary(OpKind::Mul, two, i);
        let sum = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..3), two_i);
        let root = a.push_binary(OpKind::Add, x, sum);

        let code = compile(&a, root, POINT).expect("a hoisted fold compiles");
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            assert_eq!(got, x + 6.0, "at x={x}");
        }
    }

    /// The values a fold's body reads and its enclosing scopes computed are
    /// its roots. The allocator carries them in registers across the loop
    /// while a budget lasts, so the loop reloads nothing; past it, every trip
    /// reloads the roots that did not fit, and the reloads grow with the roots
    /// the fold reads.
    ///
    /// `Σ_{i<5} Σ_k (x·c_k + y)·i`, with `k` over the roots. Both counts
    /// compile and run.
    #[test]
    fn a_folds_roots_are_placed_by_the_budget() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};
        /// The fold's trips, which no lattice loop shares at a point.
        const TRIPS: u32 = 5;
        /// Roots within every tier's carry budget, and roots past the widest
        /// vector file, which AVX-512 and NEON share.
        const FEW: usize = 2;
        const WIDEST_FILE: usize = 32;
        const MANY: usize = WIDEST_FILE + 16;
        /// AVX2's 16 vector registers less the reserve of 7 carry 9: the roots
        /// below, the binder, the accumulator and one more the fold reads.
        const AVX2_ROOTS_THAT_FIT: usize = 6;
        const AVX2_VECTOR_BYTES: usize = 32;

        let reloads_per_trip = |roots: usize| -> u64 {
            let mut a = ExprArena::new();
            let (x, y) = (a.push_var(0), a.push_var(1));
            let b = Binder::from_slot(0).expect("slot 0 exists");
            let i = a.push_var(b.var());
            let terms: Vec<ExprId> = (0..roots)
                .map(|k| {
                    let c = a.push_const(1.0 + 0.5 * k as f32);
                    let xc = a.push_binary(OpKind::Mul, x, c);
                    let root = a.push_binary(OpKind::Add, xc, y);
                    a.push_binary(OpKind::Mul, root, i)
                })
                .collect();
            let body = terms[1..]
                .iter()
                .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t));
            let sum = a.push_reduce(Fold::new(Monoid::SUM, b, 0..TRIPS), body);
            let code = compile(&a, sum, POINT).expect("a fold over roots compiles");

            let (px, py) = (0.5f32, 1.5f32);
            let want: f32 = (0..TRIPS)
                .map(|i| {
                    (0..roots)
                        .map(|k| (px * (1.0 + 0.5 * k as f32) + py) * i as f32)
                        .sum::<f32>()
                })
                .sum();
            let got = eval_point(&code.code, px, py);
            assert!(
                (got - want).abs() <= want.abs() * 1e-5,
                "{roots} roots: {got} against {want}"
            );

            let traffic = &code.traffic;
            traffic
                .scopes
                .iter()
                .zip(&traffic.trips)
                .filter(|(_, trips)| **trips == u64::from(TRIPS))
                .map(|(scope, _)| scope.loads)
                .sum()
        };

        assert_eq!(
            reloads_per_trip(FEW),
            0,
            "{FEW} roots are within the budget, so the loop reads them from registers"
        );
        let past = reloads_per_trip(MANY);
        assert!(
            past >= (MANY - WIDEST_FILE) as u64,
            "{MANY} roots are more than any vector file holds, and the loop reloads \
             only {past} of them a trip"
        );

        // The budget's own edge, where the tier's file is known: the last root
        // that fits reloads nothing, and each one past it reloads once a trip.
        if crate::jit_vector_bytes() == AVX2_VECTOR_BYTES {
            const PAST: usize = 3;
            assert_eq!(
                reloads_per_trip(AVX2_ROOTS_THAT_FIT),
                0,
                "{AVX2_ROOTS_THAT_FIT} roots are the most the budget carries"
            );
            let edge = reloads_per_trip(AVX2_ROOTS_THAT_FIT + PAST);
            assert!(
                edge >= PAST as u64,
                "{PAST} roots past the budget reload only {edge} a trip"
            );
        }
    }

    /// A fold inside a fold's body that does not depend on the outer binder
    /// is the outer scope's fold, run once, and read inside from its slot —
    /// not a loop per iteration. Both readings of it here: the outer body
    /// reads it under its own binder, and the root reads it again outside.
    ///
    /// `inner = Σ_{i<3} (X + i) = 3X + 3`; `outer = Σ_{j<2} (inner + j) =
    /// 2·inner + 1`; `root = outer + inner = 3·inner + 1 = 9X + 10`.
    #[test]
    fn an_invariant_reduce_inside_a_reduce_is_hoisted_and_shared() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let bj = Binder::from_slot(1).expect("slot 1 exists");
        let i = a.push_var(bi.var());
        let xi = a.push_binary(OpKind::Add, x, i);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..3), xi);
        let j = a.push_var(bj.var());
        let inner_j = a.push_binary(OpKind::Add, inner, j);
        let outer = a.push_reduce(Fold::new(Monoid::SUM, bj, 0..2), inner_j);
        let root = a.push_binary(OpKind::Add, outer, inner);

        let code = compile(&a, root, POINT).expect("a hoisted fold compiles");
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let want = 9.0 * x + 10.0;
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// The same hoist when the invariant fold *is* the outer body: the
    /// outer's schedule is one placeholder, and its result comes from the
    /// slot. `Σ_{j<2} Σ_{i<3} (X + i) = 2·(3X + 3) = 6X + 6`.
    #[test]
    fn an_invariant_reduce_as_a_folds_whole_body_is_hoisted() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let bj = Binder::from_slot(1).expect("slot 1 exists");
        let i = a.push_var(bi.var());
        let xi = a.push_binary(OpKind::Add, x, i);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..3), xi);
        let root = a.push_reduce(Fold::new(Monoid::SUM, bj, 0..2), inner);

        let code = compile(&a, root, POINT).expect("a fold whose body is a hoisted fold compiles");
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let want = 6.0 * x + 6.0;
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// `a_folds_spill_slots_do_not_alias_its_parents`, one level deeper: the
    /// outer fold's body holds `K` values live across the inner loop, and the
    /// inner loop holds `K` more, so both scopes go to slots and the inner
    /// fold's frame is based at the outer fold's top rather than at its
    /// parent's. The outer binder is read on both sides of the inner loop.
    ///
    /// `p_k = X + j + k` (in the outer body); `inner(j) = Σ_{i<R} Σ_k (i +
    /// p_k)`; `outer body = Σ_k (p_k + inner(j))`; `root = Σ_{j<J} outer`.
    #[test]
    fn a_nested_folds_spill_slots_do_not_alias_its_parents() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        const K: usize = 20;
        const R: u32 = 3;
        const J: u32 = 2;

        fn tree_sum(a: &mut ExprArena, mut ids: alloc::vec::Vec<ExprId>) -> ExprId {
            while ids.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                for pair in ids.chunks(2) {
                    next.push(match pair {
                        [l, r] => a.push_binary(OpKind::Add, *l, *r),
                        [only] => *only,
                        _ => unreachable!("chunks(2) yields 1 or 2"),
                    });
                }
                ids = next;
            }
            ids[0]
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let bj = Binder::from_slot(1).expect("slot 1 exists");
        let i = a.push_var(bi.var());
        let j = a.push_var(bj.var());
        let xj = a.push_binary(OpKind::Add, x, j);
        // Live across the inner loop: defined in the outer body, consumed
        // only past the inner `Reduce`.
        let p: alloc::vec::Vec<ExprId> = (0..K)
            .map(|k| {
                let c = a.push_const(k as f32);
                a.push_binary(OpKind::Add, xj, c)
            })
            .collect();
        let q: alloc::vec::Vec<ExprId> = (0..K)
            .map(|k| a.push_binary(OpKind::Add, i, p[k]))
            .collect();
        let inner_body = tree_sum(&mut a, q);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..R), inner_body);
        let joined: alloc::vec::Vec<ExprId> = p
            .iter()
            .map(|&pk| a.push_binary(OpKind::Add, pk, inner))
            .collect();
        let outer_body = tree_sum(&mut a, joined);
        let root = a.push_reduce(Fold::new(Monoid::SUM, bj, 0..J), outer_body);

        let code = compile(&a, root, POINT).expect("nested folds under register pressure compile");

        for xv in [0.0f32, 1.0, -2.5, 7.0] {
            let got = eval_point(&code.code, xv, 0.0);
            let want: f32 = (0..J)
                .map(|jv| {
                    let pv = |k: usize| xv + jv as f32 + k as f32;
                    let inner_v: f32 = (0..R)
                        .map(|iv| (0..K).map(|k| iv as f32 + pv(k)).sum::<f32>())
                        .sum();
                    (0..K).map(|k| pv(k) + inner_v).sum::<f32>()
                })
                .sum();
            let tol = want.abs() * 1e-4 + 1e-3;
            assert!(
                (got - want).abs() <= tol,
                "at x={xv}: got {got}, want {want}"
            );
        }
    }

    // =========================================================================
    // Cross-host emission: every backend, from whatever host runs the tests
    // =========================================================================

    /// Every backend emits from every host.
    ///
    /// This is the property the ISA files buy. Emission is a pure function of
    /// `(schedule, RegisterFile)` into a `Vec<u8>`, so an x86 box computes NEON
    /// instruction words and an arm box computes AVX-512 ones. Only running
    /// them needs the matching CPU.
    ///
    /// Before the backends stopped being `#[cfg]`-gated into existence, three
    /// of these four could not even be *named* here.
    ///
    /// The arena reads a uniform, so each backend's `Uniform` selection — not
    /// only the encoder behind it — is what emits here.
    #[test]
    fn every_backend_emits_from_this_host() {
        use pixelflow_ir::arena::{UniformDecl, UniformIdentity};
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let u = a.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: 1.0,
        });
        let u = a.push_uniform(u);
        let scaled = a.push_binary(OpKind::Mul, y, u);
        let root = a.push_binary(OpKind::Add, a.clone().push_var(0).max(x).min(x), scaled);

        // Each backend is handed a schedule legalized at *its own* lane
        // count: the lattice's lane fold is the one part of a collapse whose
        // shape belongs to the target.
        assert!(
            schedule_for(&a, root, POINT, 4)
                .iter()
                .any(|d| matches!(d.op, ScheduledOp::Uniform(..))),
            "the schedule must carry the uniform load for the backends to dispatch on"
        );

        let neon_len = compile_for::<aarch64::Neon>(&a, root, POINT)
            .expect("NEON emit")
            .code
            .as_bytes()
            .len();
        assert!(
            neon_len > 0 && neon_len.is_multiple_of(4),
            "aarch64 is fixed-width"
        );
        for (name, len) in [
            (
                "AVX2",
                compile_for::<avx2::Avx2>(&a, root, POINT)
                    .expect("AVX2")
                    .code
                    .as_bytes()
                    .len(),
            ),
            (
                "AVX-512",
                compile_for::<avx512::Avx512>(&a, root, POINT)
                    .expect("AVX-512")
                    .code
                    .as_bytes()
                    .len(),
            ),
        ] {
            assert!(len > 0, "{name} emitted nothing");
        }
    }

    /// Every x86 program leaves through `vzeroupper; ret`, on both tiers.
    ///
    /// The return is found from the program's structure, not by scanning for
    /// `C3`, which a ModRM byte, an immediate or a pool entry holds just as
    /// well. A program has one return, and every byte before its end is
    /// counted by a scope, so the return ends where those counts do.
    ///
    /// The pool trails the return, and its position is a label: it must sit
    /// at the first aligned byte at or after the return when it holds
    /// anything, at the return's end when it does not, across padding that
    /// is all zeros.
    #[test]
    fn every_x86_return_clears_the_upper_halves_first() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        /// `VZEROUPPER` (`VEX.128.0F.WIG 77`) then `RET` (`C3`), as the SDM
        /// spells them rather than as the encoder under test does.
        const CLEAN_RETURN: [u8; 4] = [0xC5, 0xF8, 0x77, 0xC3];
        /// Bytes in an x86 pool entry: one `f32`'s bits.
        const POOL_ENTRY: usize = 4;
        /// Three rows, and a width that leaves a remainder on either tier's
        /// batch of 8 or 16 lanes, so every fold of the lattice emits.
        const PLANE: LatticeShape = LatticeShape::new([37, 3]);

        /// A kernel to compile, by name.
        type Case<'a> = (&'static str, &'a ExprArena, ExprId);

        fn check<B: IsaBackend>(tier: &str, kernels: &[Case<'_>]) {
            for &(name, arena, root) in kernels {
                let result = compile_for::<B>(arena, root, PLANE)
                    .unwrap_or_else(|e| panic!("{tier}/{name}: {e:?}"));
                let code = result.code.as_bytes();
                // Every byte up to the return is a scope's, so what they
                // leave uncounted is what trails the return.
                let ret_end = result.traffic.scopes.iter().map(|s| s.bytes).sum::<u64>() as usize;
                assert_eq!(
                    code[ret_end - CLEAN_RETURN.len()..ret_end],
                    CLEAN_RETURN,
                    "{tier}/{name}: the return is not `vzeroupper; ret`"
                );

                // Padding exists only in front of entries, so a pool that
                // trails anything is aligned, and one that trails nothing is
                // bound where the return ends.
                let pool = match code.len() - ret_end {
                    0 => ret_end,
                    _ => ret_end.next_multiple_of(CONST_POOL_ALIGN),
                };
                assert!(
                    code[ret_end..pool].iter().all(|&b| b == 0),
                    "{tier}/{name}: the pool's padding is not zeros"
                );
                assert_eq!(
                    (code.len() - pool) % POOL_ENTRY,
                    0,
                    "{tier}/{name}: the pool is not whole entries"
                );
            }
        }

        // Nothing but coordinates: the least a program is.
        let mut plain = ExprArena::new();
        let (x, y) = (plain.push_var(0), plain.push_var(1));
        let plain_root = plain.push_binary(OpKind::Add, x, y);

        // Constants, and an `If` on a comparison: a pool to pad.
        let mut if_arena = ExprArena::new();
        let (x, y) = (if_arena.push_var(0), if_arena.push_var(1));
        let edge = if_arena.push_const(2.5);
        let scale = if_arena.push_const(3.7);
        let bias = if_arena.push_const(0.25);
        let cond = if_arena.push_binary(OpKind::Lt, x, edge);
        let scaled = if_arena.push_binary(OpKind::Mul, x, scale);
        let biased = if_arena.push_binary(OpKind::Add, y, bias);
        let if_root = if_arena.push_ternary(OpKind::If, cond, scaled, biased);

        // A surviving fold: a loop of the kernel's own inside the lattice's.
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let mut fold = ExprArena::new();
        let x = fold.push_var(0);
        let i = fold.push_var(binder.var());
        let body = fold.push_binary(OpKind::Add, x, i);
        let fold_root = fold.push_reduce(Fold::new(Monoid::SUM, binder, 0..4), body);

        let kernels = [
            ("plain", &plain, plain_root),
            ("if", &if_arena, if_root),
            ("fold", &fold, fold_root),
        ];
        check::<avx2::Avx2>("AVX2", &kernels);
        check::<avx512::Avx512>("AVX-512", &kernels);
    }

    // =========================================================================
    // Arena compilation tests
    // =========================================================================

    #[test]
    fn arena_compile_simple() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let result = compile(&arena, sum, POINT).expect("arena DAG compile failed");
        // Two leaves and one add force nothing to memory: no scope of the
        // nest stores or reloads a value. Not `spill_count`, which counts the
        // frame's slots — the lattice's own folds reserve one the emitted
        // code never touches, on every backend.
        assert_eq!(result.traffic.dynamic_memory_ops(), 0);

        assert_eq!(eval_point(&result.code, 3.0, 4.0), 7.0);
    }

    #[test]
    fn arena_compile_with_constant() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let two = arena.push_const(2.0);
        let y = arena.push_var(1);
        let prod = arena.push_binary(OpKind::Mul, x, two);
        let sum = arena.push_binary(OpKind::Add, prod, y);

        let result = compile(&arena, sum, POINT).expect("arena DAG compile failed");

        // 3*2 + 4 = 10
        assert_eq!(eval_point(&result.code, 3.0, 4.0), 10.0);
    }

    // =========================================================================
    // The shared driver's If short-circuit guard, on every backend that
    // has a JIT.
    //
    // `sched_if_guards` below covers this path on whichever tier the
    // host runs, and `avx512_if_guards` covers AVX-512 by name. aarch64
    // had no guard test at all, which mattered because
    // that is the one backend whose guard needs a scratch register: reducing a
    // mask with `UMAXV`/`UMINV` writes a scalar into a vector register, where
    // the x86 tiers use `movmskps`/`kortest` and the flags. So the register
    // that reduction destroys was, on aarch64 alone, an untested choice.
    //
    // These run wherever a backend exists, against the same expected values,
    // so no backend's guard can drift from another's.
    // =========================================================================
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    mod if_guard_driver {
        use super::*;

        /// `(X > 0) ? B³ : 3B` over a shared
        /// `B = X·Y` — arms that are exclusive *and* contiguous in the
        /// schedule.
        ///
        /// Both properties are needed and the second is easy to lose: a guard
        /// skips a whole index range, so every index in it must belong to that
        /// arm. Giving each arm its own `Var` looks exclusive but is not
        /// contiguous — the `Var` is scheduled with the other leaves, far
        /// below the arm's body, and the range from there to the arm swallows
        /// the mask. Deriving both arms from one shared value keeps every leaf
        /// out of both arms, which is what leaves the arms' own nodes adjacent.
        fn guarded_if(a: &mut ExprArena) -> ExprId {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let base = a.push_binary(OpKind::Mul, x, y);
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let bb = a.push_binary(OpKind::Mul, base, base);
            let bbb = a.push_binary(OpKind::Mul, bb, base);
            let b2 = a.push_binary(OpKind::Add, base, base);
            let b3 = a.push_binary(OpKind::Add, b2, base);
            let sel = a.push_ternary(OpKind::If, cond, bbb, b3);
            // Live ACROSS the `If` and read after it. Without something in
            // this role the `If` is the root, nothing downstream reads a
            // register, and a guard that clobbered a live one would still
            // produce the right answer — the test would be blind to exactly
            // the mistake it exists to catch.
            let carried = a.push_binary(OpKind::Sub, x, y);
            a.push_binary(OpKind::Add, sel, carried)
        }

        /// How many terms the filler below has.
        const FILLER: usize = 40;

        /// Coprime with [`FILLER`], so the pairing has no short cycle.
        fn pair(i: usize) -> usize {
            (i * 7 + 3) % FILLER
        }

        /// Filler that is live all at once whatever the evaluation order.
        ///
        /// [`FILLER`] terms off `seed`, multiplied in pairs by a permutation,
        /// so each is read twice with the others in between: no order keeps
        /// them all in registers, which is what makes the tests below about a
        /// *spilled* value rather than about an arithmetic identity. Defining
        /// them all before consuming any is not enough on its own —
        /// `passes::lattice::collapse` rebuilds the arena from the root, and
        /// the order it hands the scheduler is its own.
        fn filler(a: &mut ExprArena, seed: ExprId) -> ExprId {
            let terms: alloc::vec::Vec<ExprId> = (0..FILLER)
                .map(|i| {
                    let c = a.push_const(i as f32 + 1.0);
                    a.push_binary(OpKind::Add, seed, c)
                })
                .collect();
            let mut sum = a.push_const(0.0);
            for i in 0..FILLER {
                let product = a.push_binary(OpKind::Mul, terms[i], terms[pair(i)]);
                sum = a.push_binary(OpKind::Add, sum, product);
            }
            sum
        }

        /// [`filler`], in scalar `f32`.
        fn filler_value(seed: f32) -> f32 {
            let term = |i: usize| seed + i as f32 + 1.0;
            (0..FILLER).map(|i| term(i) * term(pair(i))).sum()
        }

        /// An `If` whose true arm contains an `If`, with entries belonging
        /// to the root sitting inside both arms — so NEITHER level is
        /// guardable as scheduled, and both become guardable once the layout
        /// gathers each arm into one run.
        ///
        /// Nesting is the case that can go wrong quietly: an inner `If`'s
        /// arms lie inside an outer arm, so partitioning the outside moves the
        /// inside with it. If that broke an inner guard the kernel would still
        /// be correct and merely slower, which no value test would catch —
        /// hence the assertion on the analysis as well as on the arithmetic.
        ///
        /// The two "intruders" are read by the root, so they are shared with
        /// the world outside the arms and can never be skipped; they are what
        /// makes the arms non-contiguous to begin with.
        fn nested_guarded_ifs(a: &mut ExprArena) -> (ExprId, ExprId, ExprId) {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let base = a.push_binary(OpKind::Mul, x, y);
            let outer_cond = a.push_binary(OpKind::Gt, x, zero);
            let inner_cond = a.push_binary(OpKind::Gt, y, zero);

            // Inner true arm, split around an entry the root reads.
            let t1 = a.push_binary(OpKind::Mul, base, base);
            let across_inner = a.push_binary(OpKind::Add, x, y);
            let t2 = a.push_binary(OpKind::Mul, t1, base);
            let one = a.push_const(1.0);
            let t3 = a.push_binary(OpKind::Add, t2, one);

            // Inner false arm.
            let f1 = a.push_binary(OpKind::Add, base, base);
            let two = a.push_const(2.0);
            let f2 = a.push_binary(OpKind::Add, f1, two);

            let inner = a.push_ternary(OpKind::If, inner_cond, t3, f2);

            // The rest of the outer true arm, split around a second one.
            let three = a.push_const(3.0);
            let o1 = a.push_binary(OpKind::Add, inner, three);
            let four = a.push_const(4.0);
            let across_outer = a.push_binary(OpKind::Mul, x, four);
            let five = a.push_const(5.0);
            let o2 = a.push_binary(OpKind::Mul, o1, five);

            // Outer false arm.
            let six = a.push_const(6.0);
            let p1 = a.push_binary(OpKind::Add, base, six);
            let seven = a.push_const(7.0);
            let p2 = a.push_binary(OpKind::Mul, p1, seven);

            let outer = a.push_ternary(OpKind::If, outer_cond, o2, p2);
            let carried = a.push_binary(OpKind::Add, across_inner, across_outer);
            let root = a.push_binary(OpKind::Add, outer, carried);
            (root, outer, inner)
        }

        /// What `nested_guarded_ifs` computes, in scalar `f32` and with no
        /// guard anywhere — every operation exact at the points below.
        fn nested_expected(x: f32, y: f32) -> f32 {
            let base = x * y;
            let inner = if y > 0.0 {
                base * base * base + 1.0
            } else {
                base + base + 2.0
            };
            let outer = if x > 0.0 {
                (inner + 3.0) * 5.0
            } else {
                (base + 6.0) * 7.0
            };
            outer + (x + y) + x * 4.0
        }

        /// Trip count of [`a_fold_owned_by_an_arm_is_guarded`]'s fold.
        const ARM_FOLD_TRIPS: u32 = 64;

        /// `(X > 0) ? Σ_{j<64} |X − j| : 0`, plus a value carried across: an
        /// arm that is a loop and nothing else. The arm owns the loop and is
        /// guarded — and the answer on the batch that skips the loop is the
        /// false arm's.
        #[test]
        fn a_fold_owned_by_an_arm_is_guarded() {
            use pixelflow_ir::fold::{Binder, Fold, Monoid};

            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let binder = Binder::from_slot(0).expect("slot 0 exists");
            let j = a.push_var(binder.var());
            let diff = a.push_binary(OpKind::Sub, x, j);
            let term = a.push_unary(OpKind::Abs, diff);
            let fold = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..ARM_FOLD_TRIPS), term);
            let sel = a.push_ternary(OpKind::If, cond, fold, zero);
            let carried = a.push_binary(OpKind::Sub, x, y);
            let root = a.push_binary(OpKind::Add, sel, carried);

            let point = compile(&a, root, POINT).expect("a guarded fold compiles");
            for &(px, py) in &[(3.0f32, 4.0f32), (-3.0, 4.0), (40.5, -2.0), (-0.5, 0.0)] {
                let arm = if px > 0.0 {
                    (0..ARM_FOLD_TRIPS).map(|j| (px - j as f32).abs()).sum()
                } else {
                    0.0
                };
                let want = arm + (px - py);
                let got = eval_point(&point.code, px, py);
                assert_eq!(got, want, "at ({px}, {py})");
            }
        }

        /// The laid-out kernel's answer, against the same expression
        /// evaluated in scalar `f32` with no guards: uniform masks (which take
        /// the branches) and mixed lanes (which fall through to the blend),
        /// exactly equal — every operation here is exact at these points, so
        /// there is no tolerance to hide a wrong branch in.
        #[test]
        fn a_nested_guarded_if_agrees_lane_for_lane() {
            let mut a = ExprArena::new();
            let (root, _outer, _inner) = nested_guarded_ifs(&mut a);
            let point = compile(&a, root, POINT).expect("nested guarded If compile");

            // One point at a time: all four combinations of the two masks,
            // each of which takes a pair of branches.
            for &(x, y) in &[(3.0f32, 4.0f32), (3.0, -4.0), (-3.0, 4.0), (-3.0, -4.0)] {
                let got = eval_point(&point.code, x, y);
                assert_eq!(
                    got,
                    nested_expected(x, y),
                    "nested guarded If at ({x}, {y})"
                );
            }

            // Mixed lanes: the batch straddles `x = 0`, so the outer mask
            // varies by lane, its guard cannot fire and the blend has to
            // produce every lane. `y` is the row, so the inner mask is
            // uniform over a batch and takes its branch — both paths, in one
            // call.
            let batch = compile(&a, root, batch()).expect("nested guarded If compile");
            let x0 = -(lanes() as f32) / 2.0;
            for y in [4.0f32, -4.0] {
                let got = eval_batch(&batch.code, &[], &[], x0, y);
                for (lane, got) in got.iter().enumerate() {
                    assert_eq!(
                        *got,
                        nested_expected(x0 + lane as f32, y),
                        "lane {lane} of a mixed-mask batch at y={y}"
                    );
                }
            }
        }

        /// Uniform masks take the all-true and all-false branches; a mixed
        /// mask falls through to the blend. All three must agree with the
        /// arithmetic.
        #[test]
        fn a_guarded_if_takes_every_branch() {
            let mut a = ExprArena::new();
            let root = guarded_if(&mut a);

            let result = compile(&a, root, POINT).expect("guarded If compile");
            for &(x, y) in &[
                (3.0f32, 4.0f32), // all-true  -> B³
                (-2.0, 0.5),      // all-false -> 3B
                (0.5, -1.0),
                (-0.25, 2.0),
            ] {
                let b = x * y;
                let want = if x > 0.0 { b * b * b } else { 3.0 * b } + (x - y);
                let got = eval_point(&result.code, x, y);
                assert!(
                    (got - want).abs() <= 1e-3,
                    "guarded If at ({x}, {y}): got {got}, want {want}"
                );
            }
        }

        /// The same, with the mask itself spilled.
        ///
        /// Getting the mask to be the value that spills takes care: eviction
        /// is Belady, so the victim is whatever is used farthest out. The mask
        /// is read only at the `If`, and the [`filler`] between fills the
        /// pool — so the mask is the farthest-out live value there, and it is
        /// the one to go.
        #[test]
        fn a_guarded_if_survives_a_spilled_mask() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);

            // Read only at the very end: the farthest-out live value.
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let mid = filler(&mut a, x);

            // Shared-base arms, as in `guarded_if`.
            let base = a.push_binary(OpKind::Mul, mid, y);
            let bb = a.push_binary(OpKind::Mul, base, base);
            let bbb = a.push_binary(OpKind::Mul, bb, base);
            let b2 = a.push_binary(OpKind::Add, base, base);
            let b3 = a.push_binary(OpKind::Add, b2, base);
            let sel = a.push_ternary(OpKind::If, cond, bbb, b3);
            let carried = a.push_binary(OpKind::Sub, x, y);
            let root = a.push_binary(OpKind::Add, sel, carried);

            let result = compile(&a, root, POINT).expect("a spilled mask compiles");
            assert!(
                result.traffic.scopes.iter().map(|s| s.stores).sum::<u64>() > 0,
                "nothing spilled, so the spilled-mask path this test exists for is never reached"
            );

            for &(px, py) in &[(3.0f32, 2.0f32), (-2.0, 0.5), (0.5, -1.0)] {
                let b = filler_value(px) * py;
                let want = if px > 0.0 { b * b * b } else { 3.0 * b } + (px - py);
                let got = eval_point(&result.code, px, py);
                assert!(
                    (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                    "spilled guarded If at ({px}, {py}): got {got}, want {want}"
                );
            }
        }

        /// A value spilled before a guarded arm, read again inside it, and read
        /// once more after it.
        ///
        /// The arm is code a uniform mask skips, so a register a read inside it
        /// reloads is one the skipped path never loaded: the value read after
        /// the arm must still be the value. `split` seeds the [`filler`], so it
        /// is live across all of it and is the farthest-out live value where
        /// the pool fills.
        #[test]
        fn a_split_range_inside_a_guarded_arm_is_correct_when_the_arm_is_skipped() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);

            let cond = a.push_binary(OpKind::Gt, x, zero);
            let xy = a.push_binary(OpKind::Mul, x, y);
            let split = a.push_unary(OpKind::Abs, xy);
            let mid = filler(&mut a, split);

            // Shared-base arms, so neither arm's leaves land outside it and
            // the arms' own nodes stay adjacent (see `guarded_if`).
            let base = a.push_binary(OpKind::Mul, mid, y);
            // The true arm reads `split` twice: one read would be reloaded
            // into a scratch and kept nowhere, which is not the case under
            // test.
            let t1 = a.push_binary(OpKind::Mul, base, split);
            let t2 = a.push_binary(OpKind::Add, t1, split);
            let t3 = a.push_binary(OpKind::Mul, t2, base);
            let f1 = a.push_binary(OpKind::Add, base, base);
            let f2 = a.push_binary(OpKind::Add, f1, base);
            let sel = a.push_ternary(OpKind::If, cond, t3, f2);
            // Read after the arm: on the skipped path this must not name the
            // register the arm would have loaded.
            let after = a.push_binary(OpKind::Add, sel, split);
            let carried = a.push_binary(OpKind::Sub, x, y);
            let root = a.push_binary(OpKind::Add, after, carried);

            let result = compile(&a, root, POINT).expect("a split range compiles");
            assert!(
                result.traffic.scopes.iter().map(|s| s.stores).sum::<u64>() > 0,
                "the value under test stayed in a register, so nothing is split"
            );
            // x < 0 is the all-false mask: the true arm — and the reload
            // inside it — never runs, and the read after it must still be the
            // value.
            for &(px, py) in &[(-2.0f32, 3.0f32), (-0.5, -4.0), (3.0, 2.0), (0.25, 1.5)] {
                let v = (px * py).abs();
                let b = filler_value(v) * py;
                let arm_value = if px > 0.0 {
                    (b * v + v) * b
                } else {
                    (b + b) + b
                };
                let want = arm_value + v + (px - py);
                let got = eval_point(&result.code, px, py);
                assert!(
                    (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                    "split across a guarded arm at ({px}, {py}): got {got}, want {want}"
                );
            }
        }
    }

    /// Run an arena kernel at `(x, 0)`, on whichever tier this host selected.
    /// The builtin-parity tests below use it.
    fn run1(arena: &ExprArena, root: ExprId, x: f32) -> f32 {
        run_xy(arena, root, x, 0.0)
    }

    /// Eval at `(x, y)`.
    fn run_xy(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
        let r = compile(arena, root, POINT).expect("compile failed");
        eval_point(&r.code, x, y)
    }

    /// A `Dwrt`-carrying arena must JIT-compile end-to-end: the compile entry
    /// runs `lower_dwrt`, so `D(√(x²+y²), x)` compiles to `x / √(x²+y²)`
    /// without the caller ever seeing the derivative machinery.
    #[test]
    fn dwrt_compiles_to_analytic_derivative() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let sum = a.push_binary(OpKind::Add, x2, y2);
        let dist = a.push_unary(OpKind::Sqrt, sum);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, dist, v0);

        for (px, py) in [(3.0f32, 4.0f32), (1.0, 1.0), (-2.0, 5.0)] {
            let got = run_xy(&a, root, px, py);
            let want = px / (px * px + py * py).sqrt();
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "d/dx dist at ({px},{py}): got {got}, want {want}"
            );
        }
    }

    /// A `Dwrt` over an op with no derivative rule must surface as a compile
    /// error (loud refusal), not a miscompile or a scheduler panic.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dwrt_of_gather_refuses_to_compile() {
        use pixelflow_ir::arena::BufferDecl;
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let bufleaf = a.push_buffer(buf);
        let x = a.push_var(0);
        let y = a.push_var(1);
        let g = a.push_ternary(OpKind::Gather, bufleaf, x, y);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, g, v0);
        assert!(compile(&a, root, POINT).is_err());
    }

    /// A deep spill frame must compile and produce correct results — the
    /// glyph-scale-kernel case that used to refuse with "exceeds 128-byte red
    /// zone". There is no red zone any more, so what is under test is only
    /// that a frame dozens of slots deep is emitted and addressed correctly.
    ///
    /// Forty terms, **paired by a permutation** so that each is read twice,
    /// far apart: no evaluation order keeps them all in registers. Pushing
    /// them all before consuming any is no longer enough on its own, because
    /// `passes::lattice::collapse` rebuilds the arena from the root and the
    /// order it hands the scheduler is its own.
    #[test]
    fn a_deep_spill_frame_compiles_correctly() {
        const TERMS: usize = 40;
        /// Coprime with `TERMS`, so `i -> PAIR(i)` is a permutation with no
        /// short cycle: a term's two readers are far apart in every order.
        fn pair(i: usize) -> usize {
            (i * 7 + 3) % TERMS
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let terms: alloc::vec::Vec<ExprId> = (0..TERMS)
            .map(|i| {
                let c = a.push_const(i as f32 + 1.0);
                let xc = a.push_binary(OpKind::Add, x, c);
                a.push_binary(OpKind::Mul, xc, y)
            })
            .collect();
        let mut root = a.push_const(0.0);
        for i in 0..TERMS {
            let product = a.push_binary(OpKind::Mul, terms[i], terms[pair(i)]);
            root = a.push_binary(OpKind::Add, root, product);
        }

        let result = compile(&a, root, POINT).expect("large spill frame must compile");
        assert!(
            result.frame_bytes > 128,
            "test did not force a deep frame (frame_bytes = {})",
            result.frame_bytes
        );

        for (px, py) in [(1.5f32, -2.0f32), (0.0, 0.0), (3.0, 4.0)] {
            let got = run_xy(&a, root, px, py);
            let term = |i: usize| (px + i as f32 + 1.0) * py;
            let want: f32 = (0..TERMS).map(|i| term(i) * term(pair(i))).sum();
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "at ({px},{py}): jit {got}, scalar {want}"
            );
        }
    }

    /// Every x86-64 unary transcendental/round op must match its scalar
    /// reference across a range of inputs — these exercise `emit_arena` →
    /// `emit_unary` directly (not the compiler's lowering).
    #[test]
    fn x86_unary_builtins_match_scalar() {
        // Tolerances reflect the shared (with aarch64) minimax-polynomial
        // accuracy over a sensible input range; exact ops use tight bounds.
        // `rel_err = |jit - scalar| / (1 + |scalar|)`.
        type UnaryCase<'a> = (OpKind, fn(f32) -> f32, &'a [f32], f32);
        let unary: &[UnaryCase] = &[
            (
                OpKind::Sqrt,
                |x| x.sqrt(),
                &[0.25, 1.0, 2.0, 9.0, 100.0],
                1e-5,
            ),
            (OpKind::Abs, |x| x.abs(), &[-3.0, -0.5, 0.0, 2.5], 1e-6),
            (OpKind::Neg, |x| -x, &[-3.0, 0.5, 2.5], 1e-6),
            (
                OpKind::Floor,
                |x| x.floor(),
                &[-2.3, -0.1, 0.9, 1.5, 3.99],
                1e-6,
            ),
            (
                OpKind::Ceil,
                |x| x.ceil(),
                &[-2.3, -0.1, 0.9, 1.5, 3.01],
                1e-6,
            ),
            (
                OpKind::Round,
                |x| x.round_ties_even(),
                &[-2.4, -0.4, 0.4, 1.5, 2.6],
                1e-6,
            ),
            // sin/cos: 4-term Chebyshev — accurate well inside [-π, π].
            (
                OpKind::Sin,
                |x| x.sin(),
                &[-2.0, -1.0, -0.3, 0.0, 0.5, 1.5, 2.0],
                6e-3,
            ),
            (
                OpKind::Cos,
                |x| x.cos(),
                &[-1.0, -0.3, 0.0, 0.5, 1.0],
                1.5e-2,
            ),
            (
                OpKind::Tan,
                |x| x.tan(),
                &[-1.0, -0.3, 0.0, 0.3, 1.0],
                2.5e-2,
            ),
            (
                OpKind::Exp,
                |x| x.exp(),
                &[-2.0, -0.5, 0.0, 1.0, 2.0, 3.0],
                5e-3,
            ),
            (
                OpKind::Exp2,
                |x| x.exp2(),
                &[-3.0, -0.5, 0.0, 1.0, 4.0],
                5e-3,
            ),
            (OpKind::Ln, |x| x.ln(), &[0.25, 0.5, 1.0, 2.0, 10.0], 5e-3),
            (
                OpKind::Log2,
                |x| x.log2(),
                &[0.25, 0.5, 1.0, 2.0, 8.0],
                5e-3,
            ),
            (
                OpKind::Log10,
                |x| x.log10(),
                &[0.1, 0.5, 1.0, 10.0, 100.0],
                5e-3,
            ),
            (
                OpKind::Atan,
                |x| x.atan(),
                &[-5.0, -0.5, -0.2, 0.0, 0.2, 0.5, 5.0],
                8e-3,
            ),
            (
                OpKind::Asin,
                |x| x.asin(),
                &[-0.8, -0.5, 0.0, 0.5, 0.8],
                1e-2,
            ),
            (
                OpKind::Acos,
                |x| x.acos(),
                &[-0.8, -0.5, 0.0, 0.5, 0.8],
                1e-2,
            ),
        ];
        for &(op, scalar, inputs, tol) in unary {
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            let root = arena.push_unary(op, x);
            for &xv in inputs {
                let got = run1(&arena, root, xv);
                let want = scalar(xv);
                let err = (got - want).abs() / (1.0 + want.abs());
                assert!(
                    err <= tol,
                    "{op:?}({xv}): jit={got} scalar={want} rel_err={err} > {tol}"
                );
            }
        }
    }

    /// Binary transcendentals + comparisons + ternaries, JIT vs scalar.
    #[test]
    fn x86_binary_ternary_builtins_match_scalar() {
        // Helper: compile f(X, Y) and eval at (x, y).
        fn run2(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
            run_xy(arena, root, x, y)
        }

        // atan2(y, x): arena Binary(Atan2, Y, X)  (op order: src1=y, src2=x)
        let pts = [
            (0.5, 2.0),
            (2.0, 0.5),
            (-0.5, 2.0),
            (0.5, -2.0),
            (-2.0, -0.5),
            (3.0, -0.5),
        ];
        {
            let mut a = ExprArena::new();
            let y = a.push_var(1);
            let x = a.push_var(0);
            let root = a.push_binary(OpKind::Atan2, y, x);
            for &(yv, xv) in &pts {
                let got = run2(&a, root, xv, yv);
                let want = yv.atan2(xv);
                assert!(
                    (got - want).abs() <= 1.5e-2,
                    "atan2({yv},{xv}): {got} vs {want}"
                );
            }
        }
        // pow(X, Y)
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(OpKind::Pow, x, y);
            for &(xv, yv) in &[(2.0f32, 3.0f32), (9.0, 0.5), (4.0, -1.0), (1.5, 2.0)] {
                let got = run2(&a, root, xv, yv);
                let want = xv.powf(yv);
                let err = (got - want).abs() / (1.0 + want.abs());
                assert!(err <= 5e-3, "pow({xv},{yv}): {got} vs {want} err={err}");
            }
        }
        // hypot(X, Y) — the sqrt(x² + y²) composition it denotes.
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let root = a.push_unary(OpKind::Sqrt, sum);
            for &(xv, yv) in &[(3.0f32, 4.0f32), (1.0, 1.0), (0.0, 2.0)] {
                let got = run2(&a, root, xv, yv);
                let want = xv.hypot(yv);
                assert!(
                    (got - want).abs() <= 1e-4,
                    "hypot({xv},{yv}): {got} vs {want}"
                );
            }
        }
        // Min / Max
        for (op, f) in [
            (OpKind::Min, f32::min as fn(f32, f32) -> f32),
            (OpKind::Max, f32::max as fn(f32, f32) -> f32),
        ] {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(op, x, y);
            for &(xv, yv) in &[(1.0f32, 2.0f32), (3.0, -1.0), (-2.0, -5.0)] {
                let got = run2(&a, root, xv, yv);
                assert!((got - f(xv, yv)).abs() <= 1e-6, "{op:?}({xv},{yv})");
            }
        }
        // clamp(X, 0.0, 1.0) — the min/max composition it denotes.
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let lo = a.push_const(0.0);
            let hi = a.push_const(1.0);
            let floored = a.push_binary(OpKind::Max, x, lo);
            let root = a.push_binary(OpKind::Min, floored, hi);
            for &xv in &[-0.5f32, 0.25, 0.9, 1.7] {
                let got = run1(&a, root, xv);
                assert!(
                    (got - xv.clamp(0.0, 1.0)).abs() <= 1e-6,
                    "clamp({xv})={got}"
                );
            }
        }
        // If(X >= 0, 1.0, -1.0) == signum-ish
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Ge, x, zero);
            let pos = a.push_const(1.0);
            let neg = a.push_const(-1.0);
            let root = a.push_ternary(OpKind::If, cond, pos, neg);
            for &xv in &[-2.0f32, -0.1, 0.1, 3.0] {
                let got = run1(&a, root, xv);
                let want = if xv >= 0.0 { 1.0 } else { -1.0 };
                assert!((got - want).abs() <= 1e-6, "select({xv})={got} want={want}");
            }
        }
    }

    // =========================================================================
    // Forward-mode dual (jet) lowering — validated against analytic derivatives.
    // Uses hardware sqrtps/divps (no polynomial approximations), so tolerances
    // are tight.
    /// Transcendental lowering: sin/cos/tan JIT through the shared driver with
    /// no backend ever emitting a transcendental (they expand to arithmetic in
    /// `lowering`). Validated against `f32` on whichever tier this host
    /// selected.
    mod lowering_tests {
        use super::*;
        use pixelflow_ir::arena::ExprArena;

        // The degree-11 Chebyshev in `passes` measures 6e-7 across the whole
        // reduced interval, so this bound sits an order of magnitude above the
        // measured worst case: tight enough to test the polynomial, loose
        // enough not to test the last bit of the build's rounding. A bound in
        // the 1e-2 range would only be able to catch gross logic errors.
        const TRIG_TOL: f32 = 1e-5;

        #[test]
        fn sin_cos_tan_match_scalar() {
            // Range beyond [-π,π] to exercise the floor-based range reduction.
            let pts = [0.0f32, 0.3, 1.0, 2.0, 3.5, -1.7, 6.0, -4.2];
            for &xv in &pts {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let s = a.push_unary(OpKind::Sin, x);
                assert!((run1(&a, s, xv) - xv.sin()).abs() <= TRIG_TOL, "sin({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let c = a.push_unary(OpKind::Cos, x);
                assert!((run1(&a, c, xv) - xv.cos()).abs() <= TRIG_TOL, "cos({xv})");
            }
            // tan away from its poles (ratio of two ~3e-3 approximations).
            for &xv in &[0.0f32, 0.3, 0.7, -0.5, 1.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let t = a.push_unary(OpKind::Tan, x);
                // tan = sin/cos amplifies both errors by 1/cos²(x); from
                // 6e-7 apiece that is ~3e-6 at x=1.
                assert!((run1(&a, t, xv) - xv.tan()).abs() <= 1e-4, "tan({xv})");
            }
        }

        /// exp/exp2/ln/log2/log10 lower to arithmetic via the bit-manip
        /// primitives (TruncToInt/IntToFloat/IAdd/Shl/Shr/BitAnd/BitOr) — the
        /// float↔int twiddling no backend can avoid. Validated vs `f32`.
        #[test]
        fn exp_log_match_scalar() {
            // exp / exp2 over a moderate range.
            for &xv in &[-2.0f32, -0.5, 0.0, 0.7, 1.5, 3.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let e = a.push_unary(OpKind::Exp, x);
                let rel = (run1(&a, e, xv) - xv.exp()).abs() / xv.exp().max(1.0);
                assert!(rel <= 1e-2, "exp({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let e2 = a.push_unary(OpKind::Exp2, x);
                let rel = (run1(&a, e2, xv) - xv.exp2()).abs() / xv.exp2().max(1.0);
                assert!(rel <= 1e-2, "exp2({xv})");
            }
            // ln / log2 / log10 over positive inputs.
            for &xv in &[0.25f32, 0.5, 1.0, 2.0, 5.0, 100.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let l = a.push_unary(OpKind::Ln, x);
                assert!((run1(&a, l, xv) - xv.ln()).abs() <= 3e-2, "ln({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let l2 = a.push_unary(OpKind::Log2, x);
                assert!((run1(&a, l2, xv) - xv.log2()).abs() <= 3e-2, "log2({xv})");
            }
        }

        /// atan/atan2/asin/acos lower to arithmetic + If (atan2 is the core;
        /// the others derive from it). Value path only — atan2 uses If, which
        /// the jet path can't differentiate. Validated vs `f32`.
        #[test]
        fn inverse_trig_match_scalar() {
            // The atan polynomial is minimax: 8.7e-5 across the interval.
            // What sets this bound is therefore not the polynomial but
            // `Recip`, which is a hardware
            // *estimate* — ~12 bits from `rcpps`, ~14 from `vrcp14ps` — so it
            // injects ~1.2e-4 into the ratio and differs by ISA level. That is
            // also why this is looser than the same check in
            // `pixelflow-ir/tests/trig_range.rs`: the scalar oracle's `Recip`
            // is an exact `1.0/x`, so that test bounds the polynomial and this
            // one bounds the polynomial plus the estimate.
            const ATAN_TOL: f32 = 1e-3;

            // atan over a wide range (exercises the |ratio|>1 swap branch).
            for &xv in &[0.0f32, 0.3, 1.0, 2.5, -0.7, -4.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let at = a.push_unary(OpKind::Atan, x);
                assert!(
                    (run1(&a, at, xv) - xv.atan()).abs() <= ATAN_TOL,
                    "atan({xv})"
                );
            }
            // asin/acos on [-1, 1].
            for &xv in &[-0.9f32, -0.4, 0.0, 0.4, 0.9] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let s = a.push_unary(OpKind::Asin, x);
                assert!(
                    (run1(&a, s, xv) - xv.asin()).abs() <= ATAN_TOL,
                    "asin({xv})"
                );

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let c = a.push_unary(OpKind::Acos, x);
                assert!(
                    (run1(&a, c, xv) - xv.acos()).abs() <= ATAN_TOL,
                    "acos({xv})"
                );
            }
            // atan2 across quadrants (y in var0, x in var1). The (1,1)/(-1,-1)…
            // cases sit at |ratio|=1, the polynomial's worst point.
            let pts = [
                (1.0f32, 1.0f32),
                (1.0, -1.0),
                (-1.0, -1.0),
                (-1.0, 1.0),
                (0.5, -2.0),
            ];
            for &(yv, xv) in &pts {
                let mut a = ExprArena::new();
                let y = a.push_var(0);
                let x = a.push_var(1);
                let r = a.push_binary(OpKind::Atan2, y, x);
                let got = run_xy(&a, r, yv, xv);
                assert!(
                    (got - yv.atan2(xv)).abs() <= ATAN_TOL,
                    "atan2({yv},{xv}) = {got}"
                );
            }
        }

        /// A transcendental composed inside arithmetic still works: sin(x)·x + 1.
        #[test]
        fn transcendental_in_expression() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let s = a.push_unary(OpKind::Sin, x);
            let sx = a.push_binary(OpKind::Mul, s, x);
            let one = a.push_const(1.0);
            let root = a.push_binary(OpKind::Add, sx, one);
            for &xv in &[0.2f32, 0.9, 2.1, -1.3] {
                let want = xv.sin() * xv + 1.0;
                // sin's ~3e-3 error is scaled by |x|, so allow for that.
                let tol = 3e-3 * (1.0 + xv.abs());
                assert!(
                    (run1(&a, root, xv) - want).abs() <= tol,
                    "sin(x)·x+1 @ {xv}"
                );
            }
        }
    }

    // =========================================================================
    // Shared-pipeline path (schedule → regalloc → spill), on the host's tier.
    // =========================================================================
    mod sched {
        use super::*;

        /// One point of a kernel whose single argument is bound to `u` — the
        /// lattice-invariant third input a test used to spell `Var(2)`.
        fn eval_point_with_arg(code: &executable::CompiledKernel, x: f32, y: f32, u: f32) -> f32 {
            collapse_into(code, &[], &[u], (x, y), POINT)[0]
        }

        /// Declare one argument in `a` and return its leaf.
        fn arg_leaf(a: &mut ExprArena, default: f32) -> ExprId {
            let slot = a.declare_uniform(pixelflow_ir::Uniform::new(default).decl());
            a.push_uniform(slot)
        }

        fn run(res: &CompileResult, x: f32, y: f32) -> f32 {
            eval_point(&res.code, x, y)
        }

        const PTS: &[(f32, f32, f32)] = &[
            (3.0, 4.0, 0.0),
            (1.0, 2.0, 3.0),
            (-2.0, 0.5, 1.5),
            (0.7, -1.3, 2.1),
        ];

        /// An expression that fits in registers compiles without spilling and
        /// computes the right answer.
        ///
        /// This used to compare a "Sethi-Ullman path" against a "scheduled
        /// path". Both arms had long since become the same `compile` call,
        /// so it compiled one function twice and asserted it equalled
        /// itself; only the ground-truth comparison was load-bearing.
        #[test]
        fn sched_no_spill_is_correct() {
            // f = sqrt(X*X + Y*Y) - Y*U, a non-commutative shape whose third
            // input is the kernel's argument rather than a third coordinate.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = arg_leaf(&mut a, 0.0);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let dist = a.push_unary(OpKind::Sqrt, sum);
            let yz = a.push_binary(OpKind::Mul, y, z);
            let sub = a.push_binary(OpKind::Sub, dist, yz); // dist - Y*Z
            let root = sub;

            let sched = compile(&a, root, POINT).expect("compile");

            for &(px, py, pz) in PTS {
                let want = (px * px + py * py).sqrt() - py * pz;
                let got = eval_point_with_arg(&sched.code, px, py, pz);
                assert!((got - want).abs() <= 1e-4, "got {got} want {want}");
            }
        }

        /// Exercises the shared driver's If short-circuit guard path on x86
        /// (MOVMSKPS all-true/all-false branches): `(X > 0) ? Y*Y*Y : X+X+X`,
        /// with arm-exclusive subexpressions so a guard region forms. Uniform
        /// inputs take the all-true / all-false branches.
        ///
        /// Both arms are per-*lane*, which is what makes them arms: an arm of
        /// the kernel's arguments alone would be lattice-invariant and hoist
        /// out of the body entirely, leaving nothing for a guard to skip.
        #[test]
        fn sched_if_guards() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero); // X > 0 -> mask
            let yy = a.push_binary(OpKind::Mul, y, y);
            let yyy = a.push_binary(OpKind::Mul, yy, y); // true arm: Y^3
            let zz = a.push_binary(OpKind::Add, x, x);
            let zzz = a.push_binary(OpKind::Add, zz, x); // false arm: 3X
            let root = a.push_ternary(OpKind::If, cond, yyy, zzz);

            let sched = compile(&a, root, POINT).expect("scheduled compile");

            // x>0 -> all-true -> Y^3 ; x<=0 -> all-false -> 3X.
            for &(px, py, _pz) in PTS {
                let want = if px > 0.0 { py * py * py } else { 3.0 * px };
                let got = run(&sched, px, py);
                assert!(
                    (got - want).abs() <= 1e-3,
                    "select: ({px},{py}) got {got} want {want}"
                );
            }
        }
    }

    // =========================================================================
    // Backend op-coverage completeness (docs/designs/2026-07-25-two-level-ir-
    // and-backend-completeness.md)
    // =========================================================================
    //
    // Turns "backend X silently doesn't support op Y" into a named, itemized
    // test failure instead of a gap nobody notices until something happens to
    // exercise it (this is exactly how AVX-512's binary-op dispatch sat at
    // 6-of-15 required ops — nothing enumerated "the ops every backend must
    // support" anywhere, so the hole was invisible until 36 tests failed by
    // accident the first time someone compiled with `-C target-feature=
    // +avx512f`).
    //
    // Every backend compiles on every host — emission is a pure function
    // into bytes — so the sweeps below run for all three from whichever
    // machine runs the tests, and a gap in any of them fails every CI job by
    // name rather than only the leg that happens to select that backend.
    mod uniforms {
        use super::*;
        use pixelflow_ir::arena::{UniformDecl, UniformIdentity};

        fn decl(default: f32) -> UniformDecl {
            UniformDecl {
                id: UniformIdentity::mint(),
                default,
            }
        }

        /// `x + u₀ + 2·u₁`, compiled once and run under two blocks: the
        /// values come from the block at the call, and the uniform-only
        /// product was hoisted.
        #[test]
        fn a_block_is_read_at_the_call_not_at_compile() {
            let mut a = ExprArena::new();
            let u0 = a.declare_uniform(decl(0.0));
            let u1 = a.declare_uniform(decl(0.0));
            let x = a.push_var(0);
            let r0 = a.push_uniform(u0);
            let r1 = a.push_uniform(u1);
            let two = a.push_const(2.0);
            let scaled = a.push_binary(OpKind::Mul, r1, two);
            let sum = a.push_binary(OpKind::Add, x, r0);
            let root = a.push_binary(OpKind::Add, sum, scaled);
            let res = compile(&a, root, batch()).expect("compile");
            assert!(res.hoisted_values >= 1, "2·u₁ is per call");

            for block in [[1.0f32, 10.0], [-2.5, 0.25]] {
                let out = eval_batch(&res.code, &[], &block, 0.0, 0.0);
                for (i, got) in out.iter().enumerate() {
                    assert_eq!(
                        *got,
                        i as f32 + block[0] + 2.0 * block[1],
                        "lane {i} under {block:?}"
                    );
                }
            }
        }

        /// The block's pointer is the context entry after the buffer slots:
        /// a kernel over one buffer reads its block from `ctx[1]`.
        #[test]
        fn the_block_pointer_follows_the_buffer_slots() {
            use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
            let data = [10.0f32, 20.0, 30.0, 40.0];
            let mut a = ExprArena::new();
            let buf = a.declare_buffer(BufferDecl {
                id: BufferIdentity::mint(),
                width: 4,
                height: 1,
            });
            let u = a.declare_uniform(decl(0.0));
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let g = a.push_gather(buf, x, zero);
            let r = a.push_uniform(u);
            let root = a.push_binary(OpKind::Add, g, r);
            let res = compile(&a, root, batch()).expect("compile");

            let out = eval_batch(&res.code, &[data.as_ptr()], &[0.5f32], 0.0, 0.0);
            for (i, got) in out.iter().enumerate() {
                assert_eq!(*got, data[i.min(3)] + 0.5, "lane {i}");
            }
        }

        /// `select(u > 0, p(t[u]), 0) + (u + 1) + x`, `p` a polynomial long
        /// enough to be worth a branch: an `If` over per-call values, so the
        /// body computes it and clusters its arms, and `u + 1` — read by the
        /// root, not the `If` — is what makes the true arm non-contiguous
        /// until it does. The arm reads the table through its `Context`
        /// pointer, which only the arm's broadcast reads. A pointer operand is
        /// a read to the guard analysis, so clustering keeps that pointer
        /// ahead of the broadcast rather than sinking it past the `If` as a
        /// stranger.
        #[test]
        fn a_per_call_if_reads_its_table_through_a_defined_pointer() {
            use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
            let data = [4.0f32, 1.5, -2.0, 0.5];
            let poly = |t: f32| ((t * t + t) * t + 3.0) * t * t + 1.0;
            let mut a = ExprArena::new();
            let buf = a.declare_buffer(BufferDecl {
                id: BufferIdentity::mint(),
                width: data.len() as u32,
                height: 1,
            });
            let u = a.declare_uniform(decl(0.0));
            let x = a.push_var(0);
            let uu = a.push_uniform(u);
            let zero = a.push_const(0.0);
            let one = a.push_const(1.0);
            let three = a.push_const(3.0);
            let mask = a.push_binary(OpKind::Gt, uu, zero);
            let leaf = a.push_buffer(buf);
            let t = a.push_binary(OpKind::RawGather, leaf, uu);
            let tt = a.push_binary(OpKind::Mul, t, t);
            let p = a.push_binary(OpKind::Add, tt, t);
            let p = a.push_binary(OpKind::Mul, p, t);
            let p = a.push_binary(OpKind::Add, p, three);
            let p = a.push_binary(OpKind::Mul, p, t);
            let p = a.push_binary(OpKind::Mul, p, t);
            let p = a.push_binary(OpKind::Add, p, one);
            let sel = a.push_ternary(OpKind::If, mask, p, zero);
            let intruder = a.push_binary(OpKind::Add, uu, one);
            let lhs = a.push_binary(OpKind::Add, sel, intruder);
            let root = a.push_binary(OpKind::Add, lhs, x);

            let res = compile(&a, root, batch()).expect("compile");
            for block in [1.0f32, 2.0, 3.0, 0.0, -1.0] {
                let arm = if block > 0.0 {
                    poly(data[block as usize])
                } else {
                    0.0
                };
                let out = eval_batch(&res.code, &[data.as_ptr()], &[block], 0.0, 0.0);
                for (i, got) in out.iter().enumerate() {
                    assert_eq!(
                        *got,
                        arm + (block + 1.0) + i as f32,
                        "lane {i}, u = {block}"
                    );
                }
            }
        }

        /// The slot `UniformId` used to stop at, and one past it, in
        /// bytes: offset 3 shifted up by a full 16-bit range, so that the
        /// byte offset `262_156` (`0x0004_000C`) is `0x0C` wrapped to 16 bits
        /// — the load a narrower offset would have emitted for it, reading
        /// argument 3.
        const PAST_U16: u64 = 3 + (u16::MAX as u64 + 1);
        const PAST_U16_BYTES: u32 = 262_156;

        /// The whole path at that width: an arena declaring more arguments
        /// than 16 bits index, reading the last, scheduled and emitted by
        /// each backend from this host. The slot survives selection unnarrowed,
        /// and the bytes carry the displacement of the argument actually read.
        ///
        /// The schedule is read *by block*, the way
        /// `a_uniform_and_what_depends_on_it_alone_land_in_the_body` tells
        /// the kernel's uniform from the origin's: the link's block (context
        /// slot 0, there being no buffers) is read exactly once, at
        /// `PAST_U16`, and the origin's block (slot 1) exactly twice, at 0
        /// and 1 — `x0` and `y0` at the slots `origin_slots` found for them,
        /// which lie past every one of the kernel's own. That is what pins
        /// `origin_slots` at the widened width: were it to narrow its
        /// answer to 16 bits, slots `PAST_U16 + 1` and `+ 2` would come back
        /// as 4 and 5, match no read, and the origin would schedule as two
        /// *link* reads past the end of the block — while the `PAST_U16`
        /// read and its displacement in the bytes stayed exactly as they are.
        #[test]
        fn a_uniform_past_the_old_u16_width_loads_on_every_backend() {
            use pixelflow_ir::arena::{UniformDecl, UniformIdentity};
            const ARGUMENTS: u64 = PAST_U16 + 1;
            let mut a = ExprArena::new();
            let mut last = None;
            for i in 0..ARGUMENTS {
                last = Some(a.declare_uniform(UniformDecl {
                    id: UniformIdentity::mint(),
                    default: i as f32,
                }));
            }
            let last = last.expect("declared");
            assert_eq!(last, UniformId(PAST_U16));
            let x = a.push_var(0);
            let y = a.push_var(1);
            let xy = a.push_binary(OpKind::Add, x, y);
            let u = a.push_uniform(last);
            let root = a.push_binary(OpKind::Add, xy, u);

            // Every uniform read, as (the context slot of the block it reads,
            // its offset in that block), sorted.
            let block_reads = |schedule: &[crate::program::Def]| -> Vec<(u16, u64)> {
                let block_of = |base: crate::program::ValueId| {
                    schedule
                        .iter()
                        .find_map(|d| match d.op {
                            ScheduledOp::Context(slot) if d.value == base => Some(slot),
                            _ => None,
                        })
                        .expect("a uniform read's base is a block's Context def")
                };
                let mut reads: Vec<(u16, u64)> = schedule
                    .iter()
                    .filter_map(|d| match d.op {
                        ScheduledOp::Uniform(base, offset) => Some((block_of(base), offset)),
                        _ => None,
                    })
                    .collect();
                reads.sort_unstable();
                reads
            };
            assert_eq!(
                block_reads(&schedule_for(&a, root, POINT, 8)),
                [(0, PAST_U16), (1, 0), (1, 1)],
                "the last argument from the link's block, at its full width; \
                 x0 and y0 from the origin's block, at theirs"
            );

            let disp = PAST_U16_BYTES.to_le_bytes();
            for (tier, code) in [
                (
                    "AVX2",
                    compile_for::<avx2::Avx2>(&a, root, POINT)
                        .expect("AVX2")
                        .code,
                ),
                (
                    "AVX-512",
                    compile_for::<avx512::Avx512>(&a, root, POINT)
                        .expect("AVX-512")
                        .code,
                ),
            ] {
                assert!(
                    code.as_bytes().windows(disp.len()).any(|w| w == disp),
                    "{tier}: no vbroadcastss with disp32 {PAST_U16_BYTES:#x}"
                );
            }

            // NEON adds the high part of the offset to the block's address,
            // into a register the allocator chose, and loads from that
            // register with the low part: `N` is read from the bytes.
            let selected = compile_for::<aarch64::Neon>(&a, root, POINT)
                .expect("NEON")
                .code;
            let words: Vec<u32> = selected
                .as_bytes()
                .chunks(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let (hi, lo) = (PAST_U16_BYTES >> 12, PAST_U16_BYTES & 0xFFF);
            // `add xN, xM, #hi, lsl 12`, then `ldr sT, [xN, #lo]`.
            let add_hi = |w: u32| (w & !0x3FF) == 0x9140_0000 | (hi << 10);
            let ldr_through =
                |w: u32, n: u32| (w & !0x1F) == 0xBD40_0000 | ((lo / 4) << 10) | (n << 5);
            assert!(
                words.iter().any(
                    |&add| add_hi(add) && words.iter().any(|&ldr| ldr_through(ldr, add & 0x1F))
                ),
                "NEON selection: no `add xN, base, #{hi}, lsl 12` feeding an `ldr s, [xN, #{lo}]`"
            );
        }
    }

    /// A one-row buffer of `width` samples, declared in `a`.
    fn table(a: &mut ExprArena, width: u32) -> pixelflow_ir::arena::BufferId {
        a.declare_buffer(pixelflow_ir::arena::BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width,
            height: 1,
        })
    }

    /// What an `If` arm may own when the values it reads are shared with the
    /// world outside it — through a block pointer the lowering mints.
    mod arm_ownership {
        use super::*;
        use pixelflow_ir::arena::{UniformDecl, UniformIdentity};

        fn decl(default: f32) -> UniformDecl {
            UniformDecl {
                id: UniformIdentity::mint(),
                default,
            }
        }

        /// `x > 0 ? rsqrt(x·u1 + 2) : -x`, plus `x·u2` outside the `If`: two
        /// arguments of one link block, the first read only inside the true
        /// arm and the second outside it.
        ///
        /// The block's base is a `Context` def made at the first `Uniform`
        /// read, which sits inside the arm's span — and is read again by the
        /// second uniform's load, outside it. An arm that owned the base
        /// would skip it on a batch with no true lane, and the second
        /// uniform's load would read a register nothing wrote. So the base
        /// belongs to the scope, the arm keeps its branch, and the kernel is
        /// right on a batch of all-true, all-false and mixed lanes.
        #[test]
        fn a_block_pointer_read_inside_an_arm_and_outside_it_is_the_scopes() {
            let mut a = ExprArena::new();
            let (u1, u2) = (a.declare_uniform(decl(3.0)), a.declare_uniform(decl(5.0)));
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let two = a.push_const(2.0);
            let mask = a.push_binary(OpKind::Gt, x, zero);
            let first = a.push_uniform(u1);
            let scaled = a.push_binary(OpKind::Mul, x, first);
            let shifted = a.push_binary(OpKind::Add, scaled, two);
            let heavy = a.push_unary(OpKind::Rsqrt, shifted);
            let light = a.push_unary(OpKind::Neg, x);
            let sel = a.push_ternary(OpKind::If, mask, heavy, light);
            let second = a.push_uniform(u2);
            let tail = a.push_binary(OpKind::Mul, x, second);
            let root = a.push_binary(OpKind::Add, sel, tail);

            let shape = LatticeShape::new([lanes() as u32, 1]);
            let result = compile(&a, root, shape).expect("compiles");
            let expect = |x: f32| {
                let sel = if x > 0.0 {
                    1.0 / (x * 3.0 + 2.0).sqrt()
                } else {
                    -x
                };
                sel + x * 5.0
            };
            let width = lanes();
            for (label, start) in [
                ("all lanes true", 1.0),
                ("all lanes false", -(width as f32) - 1.0),
                ("mixed lanes", -(width as f32 / 2.0) + 0.5),
            ] {
                let out = collapse_into(&result.code, &[], &[3.0, 5.0], (start, 0.0), shape);
                for (lane, got) in out.iter().enumerate() {
                    let want = expect(start + lane as f32);
                    assert!(
                        (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "{label}, lane {lane}: {got} != {want}"
                    );
                }
            }
        }
    }

    /// A gather whose address the lane binder does not reach is one scalar
    /// load broadcast — `ScheduledOp::Broadcast`, split from `Gather` in
    /// `arena_to_schedule` by the index's variance.
    mod broadcast {
        use super::*;

        /// Every lane holds the one element the row names, and another row
        /// another element: the broadcast reads through the same context
        /// slot a gather does, at the index lane 0 holds.
        #[test]
        fn every_lane_holds_the_rows_element() {
            let data: Vec<f32> = (0..8).map(|i| 10.0 * i as f32 + 1.0).collect();
            let mut a = ExprArena::new();
            let buf = table(&mut a, data.len() as u32);
            let y = a.push_var(1);
            let leaf = a.push_buffer(buf);
            let root = a.push_binary(OpKind::RawGather, leaf, y);
            let res = compile(&a, root, batch()).expect("compile");
            for row in [0.0f32, 3.0, 7.0] {
                let out = eval_batch(&res.code, &[data.as_ptr()], &[], 0.0, row);
                assert!(
                    out.iter().all(|&v| v == data[row as usize]),
                    "row {row}: {out:?}"
                );
            }
        }
    }

    mod backend_op_coverage {
        use super::super::coverage::*;
        use super::*;

        /// Whether backend `B` compiles `arena`'s kernel.
        ///
        /// Every backend signals an op it cannot select the same way — by
        /// panicking through [`unimplemented_op`], because after `legalize`
        /// that is a missing implementation rather than a property of the
        /// kernel. `catch_unwind` is what lets this test report *which* ops a
        /// backend owes, by name and all of them, instead of dying on the
        /// first one.
        fn compiles<B: IsaBackend>(arena: &ExprArena, root: ExprId) -> bool {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compile_for::<B>(arena, root, POINT).is_ok()
            }))
            .unwrap_or(false)
        }

        /// A kernel of one op over the coordinates, and a constant for a
        /// shift's immediate amount.
        fn kernel(
            build: impl FnOnce(&mut ExprArena, ExprId, ExprId, ExprId) -> ExprId,
        ) -> (ExprArena, ExprId) {
            let mut a = ExprArena::new();
            let (x, y) = (a.push_var(0), a.push_var(1));
            let one = a.push_const(1.0);
            let root = build(&mut a, x, y, one);
            (a, root)
        }

        /// Sweep the required unary/binary/shift op lists plus the two
        /// ternary ops (`MulAdd`, `If`) against `B`, collecting every failure
        /// instead of stopping at the first one — a completeness gap is much
        /// cheaper to fix as an itemized list than rediscovered one `cargo
        /// test` run per missing op.
        fn assert_covers_required_ops<B: IsaBackend>(backend_name: &str) {
            debug_assert_eq!(REQUIRED_TERNARY_OPS, &[OpKind::MulAdd, OpKind::If]);
            let mut missing = alloc::vec::Vec::new();
            let mut check = |what: alloc::string::String, (a, root): (ExprArena, ExprId)| {
                if !compiles::<B>(&a, root) {
                    missing.push(what);
                }
            };

            for &op in REQUIRED_UNARY_OPS {
                check(
                    alloc::format!("unary {op:?}"),
                    kernel(|a, x, _, _| a.push_unary(op, x)),
                );
            }
            for &op in REQUIRED_BINARY_OPS {
                check(
                    alloc::format!("binary {op:?}"),
                    kernel(|a, x, y, _| a.push_binary(op, x, y)),
                );
            }
            for &op in REQUIRED_SHIFT_OPS {
                check(
                    alloc::format!("shift {op:?}"),
                    kernel(|a, x, _, one| a.push_binary(op, x, one)),
                );
            }
            check(
                alloc::string::String::from("ternary MulAdd"),
                kernel(|a, x, y, one| a.push_ternary(OpKind::MulAdd, x, y, one)),
            );
            check(
                alloc::string::String::from("ternary If"),
                kernel(|a, x, y, one| {
                    let mask = a.push_binary(OpKind::Lt, x, one);
                    a.push_ternary(OpKind::If, mask, x, y)
                }),
            );

            assert!(
                missing.is_empty(),
                "{backend_name} is missing required ops: {missing:?} (see \
                 pixelflow-codegen/src/emit/coverage.rs for the full \
                 completeness contract)"
            );
        }

        // Ungated, like every sweep below it: these only *encode* — bytes
        // into a Vec, never executed — and every backend now compiles on
        // every host, so a coverage gap in any of the three fails every CI
        // job rather than only the one leg that happens to select it.
        #[test]
        fn avx2_backend_covers_required_ops() {
            assert_covers_required_ops::<avx2::Avx2>("Avx2");
        }

        #[test]
        fn avx512_backend_covers_required_ops() {
            assert_covers_required_ops::<avx512::Avx512>("Avx512");
        }

        #[test]
        fn aarch64_backend_covers_required_ops() {
            assert_covers_required_ops::<aarch64::Neon>("Neon");
        }
    }

    // =========================================================================
    // MulAdd: the instruction behind `LaneOp::MulAdd`.
    //
    // `MulAdd` rounds once on every target: each has an FMA. The instruction is
    // checked as *present*, not just as "it emitted something": a backend that
    // quietly encoded a multiply and an add would still emit, still pass every
    // ULP-tolerant equivalence test, and change the last bit of the answer.
    //
    // Ungated: emission is a pure function into a `Vec<u8>`, so all three
    // backends are checked from whichever host runs the tests.
    // =========================================================================
    mod muladd_encoding {
        use super::*;

        /// A `MulAdd` DAG reaches each backend's fused multiply-add: no
        /// multiply-then-add pair stands in for it.
        #[test]
        fn fused_encodes_to_the_targets_fma() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_binary(OpKind::Add, y, x);
            let root = a.push_ternary(OpKind::MulAdd, x, y, z);

            // VEX.256.66.0F38.W0 B8 /r — vfmadd231ps: the 3-byte prefix `C4`,
            // whose second byte carries the map (`0F38` is `00010`) under
            // three register-extension bits the allocator's choice of
            // registers decides, then `W = 0` with `pp = 01`, then the opcode.
            let avx2 = compile_for::<avx2::Avx2>(&a, root, POINT).expect("AVX2");
            assert!(
                avx2.code.as_bytes().windows(4).any(|w| w[0] == 0xc4
                    && w[1] & 0x1f == 0x02
                    && w[2] & 0x83 == 0x01
                    && w[3] == 0xb8),
                "AVX2: a MulAdd DAG did not reach vfmadd231ps"
            );
            // EVEX.512.66.0F38.W0 B8 /r — the 4-byte prefix `62`, then the opcode.
            let avx512 = compile_for::<avx512::Avx512>(&a, root, POINT).expect("AVX-512");
            assert!(
                avx512.code.as_bytes().windows(5).any(|w| w[0] == 0x62
                    && w[1] & 0x07 == 0x02
                    && w[2] & 0x83 == 0x01
                    && w[4] == 0xb8),
                "AVX-512: a MulAdd DAG did not reach vfmadd231ps"
            );
            // FMLA Vd.4S, Vn.4S, Vm.4S: `0 1 0 01110 0 0 1 Rm 11001 1 Rn Rd`.
            let neon = compile_for::<aarch64::Neon>(&a, root, POINT).expect("NEON");
            assert!(
                neon.code
                    .as_bytes()
                    .chunks(4)
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .any(|word| word & 0xFFE0_FC00 == 0x4E20_CC00),
                "NEON: a MulAdd DAG did not reach fmla"
            );
        }
    }
}
