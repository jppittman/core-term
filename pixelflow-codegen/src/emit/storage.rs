//! Physical storage locations, stack slots, and frame allocation.
//!
//! A vector value on the physical machine resides in one of two physical
//! storage classes:
//! - In a hardware vector [`Reg`]
//! - In an aligned stack frame [`Slot`]
//!
//! This module models these entities as first-class domain objects, providing
//! capability traits for writing ([`StoreTarget`]) and reading ([`SourceOperand`])
//! across registers and memory, along with a recycling [`StackFrame`] slot allocator.

use super::Reg;
use alloc::vec::Vec;

/// An aligned slot in the stack frame.
///
/// A `Slot` represents a concrete stack address: it knows its byte displacement
/// relative to the stack/frame pointer and the vector width (in bytes).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot {
    offset: u32,
    bytes: u32,
}

impl Slot {
    /// Create a new stack slot with the given byte displacement and size.
    #[inline]
    #[must_use]
    pub const fn new(offset: u32, bytes: u32) -> Self {
        Self { offset, bytes }
    }

    /// Displacement in bytes from the stack/frame pointer (e.g. `[rsp + offset]`).
    #[inline]
    #[must_use]
    pub const fn offset(self) -> u32 {
        self.offset
    }

    /// Size of the vector slot in bytes (16 for SSE/NEON, 32 for AVX2, 64 for AVX-512).
    #[inline]
    #[must_use]
    pub const fn bytes(self) -> u32 {
        self.bytes
    }
}

/// A physical location where a vector value can reside: in a register or on the stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    /// A hardware vector register.
    Reg(Reg),
    /// A stack frame slot.
    Slot(Slot),
}

impl From<Reg> for Storage {
    #[inline]
    fn from(r: Reg) -> Self {
        Storage::Reg(r)
    }
}

impl From<Slot> for Storage {
    #[inline]
    fn from(s: Slot) -> Self {
        Storage::Slot(s)
    }
}

/// Capability: a physical storage location that can be written to by an instruction or transfer.
pub trait StoreTarget: Copy + core::fmt::Debug {
    /// Convert to the canonical [`Storage`] enum.
    fn as_storage(self) -> Storage;

    /// Extract the register if this target is in-register.
    fn as_reg(self) -> Option<Reg>;

    /// Extract the stack slot if this target is on-stack.
    fn as_slot(self) -> Option<Slot>;
}

impl StoreTarget for Reg {
    #[inline]
    fn as_storage(self) -> Storage {
        Storage::Reg(self)
    }
    #[inline]
    fn as_reg(self) -> Option<Reg> {
        Some(self)
    }
    #[inline]
    fn as_slot(self) -> Option<Slot> {
        None
    }
}

impl StoreTarget for Slot {
    #[inline]
    fn as_storage(self) -> Storage {
        Storage::Slot(self)
    }
    #[inline]
    fn as_reg(self) -> Option<Reg> {
        None
    }
    #[inline]
    fn as_slot(self) -> Option<Slot> {
        Some(self)
    }
}

impl StoreTarget for Storage {
    #[inline]
    fn as_storage(self) -> Storage {
        self
    }
    #[inline]
    fn as_reg(self) -> Option<Reg> {
        match self {
            Storage::Reg(r) => Some(r),
            Storage::Slot(_) => None,
        }
    }
    #[inline]
    fn as_slot(self) -> Option<Slot> {
        match self {
            Storage::Reg(_) => None,
            Storage::Slot(s) => Some(s),
        }
    }
}

/// Capability: a location or constant that can be read from as an operand.
pub trait SourceOperand: Copy + core::fmt::Debug {
    /// Extract the physical storage location if this operand resides in memory or register.
    fn as_storage(self) -> Option<Storage>;

    /// Extract the register if this operand resides in a register.
    fn as_reg(self) -> Option<Reg>;

    /// Extract the stack slot if this operand resides on the stack.
    fn as_slot(self) -> Option<Slot>;

    /// Extract constant bit pattern if this operand is a rematerialized immediate.
    fn as_const(self) -> Option<u32>;
}

impl SourceOperand for Reg {
    #[inline]
    fn as_storage(self) -> Option<Storage> {
        Some(Storage::Reg(self))
    }
    #[inline]
    fn as_reg(self) -> Option<Reg> {
        Some(self)
    }
    #[inline]
    fn as_slot(self) -> Option<Slot> {
        None
    }
    #[inline]
    fn as_const(self) -> Option<u32> {
        None
    }
}

impl SourceOperand for Slot {
    #[inline]
    fn as_storage(self) -> Option<Storage> {
        Some(Storage::Slot(self))
    }
    #[inline]
    fn as_reg(self) -> Option<Reg> {
        None
    }
    #[inline]
    fn as_slot(self) -> Option<Slot> {
        Some(self)
    }
    #[inline]
    fn as_const(self) -> Option<u32> {
        None
    }
}

impl SourceOperand for Storage {
    #[inline]
    fn as_storage(self) -> Option<Storage> {
        Some(self)
    }
    #[inline]
    fn as_reg(self) -> Option<Reg> {
        match self {
            Storage::Reg(r) => Some(r),
            Storage::Slot(_) => None,
        }
    }
    #[inline]
    fn as_slot(self) -> Option<Slot> {
        match self {
            Storage::Reg(_) => None,
            Storage::Slot(s) => Some(s),
        }
    }
    #[inline]
    fn as_const(self) -> Option<u32> {
        None
    }
}

/// A recycling stack frame slot allocator.
///
/// Manages allocation and deallocation of vector stack slots at a fixed byte stride,
/// reusing slots that have been released to minimize stack frame size.
#[derive(Clone, Debug)]
pub struct StackFrame {
    vector_bytes: u32,
    allocated_bytes: u32,
    free_pool: Vec<Slot>,
}

impl StackFrame {
    /// Create a new stack frame manager for vector slots of `vector_bytes` stride.
    #[inline]
    #[must_use]
    pub const fn new(vector_bytes: u32) -> Self {
        Self {
            vector_bytes,
            allocated_bytes: 0,
            free_pool: Vec::new(),
        }
    }

    /// Allocate a slot in the frame, reusing a previously freed slot if available.
    pub fn alloc_slot(&mut self) -> Result<Slot, crate::error::CompileError> {
        const MAX_FRAME: u32 = 2 * 1024 * 1024;
        if let Some(reused) = self.free_pool.pop() {
            Ok(reused)
        } else {
            if self.allocated_bytes > MAX_FRAME - self.vector_bytes {
                return Err(crate::error::CompileError::BudgetExceeded(
                    "spill frame overflow: exceeds 2MB stack limit",
                ));
            }
            let offset = self.allocated_bytes;
            self.allocated_bytes += self.vector_bytes;
            Ok(Slot::new(offset, self.vector_bytes))
        }
    }

    /// Release a slot back to the free pool so it can be reused by a later value.
    pub fn free_slot(&mut self, slot: Slot) {
        debug_assert_eq!(slot.bytes(), self.vector_bytes);
        self.free_pool.push(slot);
    }

    /// Total stack frame size in bytes, aligned to 16 bytes per standard ABI.
    #[inline]
    #[must_use]
    pub const fn frame_size(&self) -> u32 {
        (self.allocated_bytes + 15) & !15
    }

    /// The vector stride in bytes.
    #[inline]
    #[must_use]
    pub const fn vector_bytes(&self) -> u32 {
        self.vector_bytes
    }
}
