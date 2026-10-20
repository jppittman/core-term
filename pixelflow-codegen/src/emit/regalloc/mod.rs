//! Register allocation for selected functions.
//!
//! [`RegisterAllocator`] is a function from a selected [`Function`] to an
//! [`Allocated`] one; [`LinearScan`] (`local`) is its one implementation.
//! [`policy`] is what an allocator prefers, and [`resource`] is what it
//! allocates from: the pool, the frame, and the tokens they lend.

mod local;
mod policy;
pub(super) mod resource;

pub(super) use local::LinearScan;

use crate::emit::asm::{AsmProgram, Item, Label, Labels};
use crate::emit::{Block, Bound, CONST_POOL_ALIGN, Constants, Function, IsaBackend, Loop};
use crate::error::CompileError;
use alloc::vec::Vec;

/// Why an instruction is there. `EmitTraffic` counts them per scope.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) enum Origin {
    Selected,
    /// A definition the backend can recompute, placed again where a read needs
    /// its value in a register ([`IsaBackend::rematerializable`]):
    /// `EmitTraffic`'s remats when it reads the constant pool
    /// ([`IsaBackend::reads_pool`]).
    Remat,
    Spill,
    Reload,
    /// A register copy ([`IsaBackend::copy`]): an operand the instruction
    /// overwrites in place, which is read again later.
    Copy,
}

/// An allocated instruction, and why it is there.
pub(in crate::emit) struct Emitted<'m, B: IsaBackend> {
    pub(in crate::emit) inst: B::Inst<Bound<'m, B>>,
    pub(in crate::emit) origin: Origin,
}

/// An allocated kernel: every register a borrowed token, every slot a
/// borrowed slot, every block argument a placed move. A block has no
/// parameters: they are slots.
pub(in crate::emit) struct Allocated<'m, B: IsaBackend> {
    pub(in crate::emit) blocks: Vec<Block<Emitted<'m, B>>>,
    pub(in crate::emit) loops: Vec<Loop>,
    constants: Constants<B::Constant>,
    labels: Labels,
    /// The position after the last instruction, where the data section's
    /// padding begins.
    pub(in crate::emit) text_end: Label,
    pub(in crate::emit) frame_bytes: u64,
    /// Frame slots minted.
    pub(in crate::emit) slots: u64,
    /// Values read inside a loop that does not contain their definition.
    pub(in crate::emit) hoisted: u64,
    pub(in crate::emit) scheduled: Vec<u64>,
}

impl<'m, B: IsaBackend> Allocated<'m, B> {
    /// The kernel as a program for the assembler: its code, then its pool.
    /// The program owns the label mint from here on.
    pub(in crate::emit) fn program(&mut self) -> AsmProgram<&Emitted<'m, B>> {
        let mut text = Vec::new();
        for block in &self.blocks {
            text.push(Item::Bind(block.label));
            text.extend(block.insts.iter().map(Item::Inst));
        }
        text.push(Item::Bind(self.text_end));
        let mut data = Vec::new();
        if !self.constants.entries.is_empty() {
            data.push(Item::Align(CONST_POOL_ALIGN as u64));
        }
        data.push(Item::Bind(self.constants.label));
        for &(label, constant) in &self.constants.entries {
            let bytes = B::constant_bytes(constant);
            assert_eq!(
                bytes.len(),
                size_of::<B::Constant>(),
                "pool entries sit back to back, so `Constant::offset` is the entry's size times its index"
            );
            data.push(Item::Bind(label));
            data.push(Item::Bytes(bytes));
        }
        AsmProgram {
            text,
            data,
            labels: core::mem::take(&mut self.labels),
        }
    }
}

/// Binds every value of a selected function to a register or a frame slot.
pub(in crate::emit) trait RegisterAllocator<'m, B: IsaBackend> {
    /// # Errors
    /// [`CompileError::BudgetExceeded`] when the frame cannot hold the spills.
    fn allocate(self, function: Function<B>) -> Result<Allocated<'m, B>, CompileError>;
}
