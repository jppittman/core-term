// pixelflow-core/src/bitwise.rs
//! Low-level bitwise operations and storage for primitive types.

use crate::backend::{BatchArithmetic, SimdBatch};
use crate::batch::{Batch, NativeBackend};
use core::fmt::Debug;

/// Trait for types that can be manipulated as bit patterns (u32/f32/etc).
pub trait Bitwise: Copy + Default + Debug + PartialEq + 'static + Send + Sync {
    /// Create from raw u32 bits.
    fn from_bits(v: u32) -> Self;

    /// Convert to raw u32 bits.
    fn to_bits(self) -> u32;

    /// Convert a batch of self to a batch of u32 (packed representation).
    fn batch_to_bits(batch: Batch<Self>) -> Batch<u32>;

    /// Convert a batch of u32 (packed representation) to a batch of self.
    fn batch_from_bits(batch: Batch<u32>) -> Batch<Self>;

    /// Gather values from a slice using indices.
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self>;

    /// Store a batch of values into a slice.
    fn batch_store(batch: Batch<Self>, slice: &mut [Self]);
}

impl Bitwise for u32 {
    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        v
    }
    #[inline(always)]
    fn to_bits(self) -> u32 {
        self
    }
    #[inline(always)]
    fn batch_to_bits(batch: Batch<Self>) -> Batch<u32> {
        batch
    }
    #[inline(always)]
    fn batch_from_bits(batch: Batch<u32>) -> Batch<Self> {
        batch
    }
    #[inline(always)]
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self> {
        BatchArithmetic::gather(slice, indices)
    }
    #[inline(always)]
    fn batch_store(batch: Batch<Self>, slice: &mut [Self]) {
        SimdBatch::store(&batch, slice);
    }
}

impl Bitwise for f32 {
    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        f32::from_bits(v)
    }
    #[inline(always)]
    fn to_bits(self) -> u32 {
        self.to_bits()
    }
    #[inline(always)]
    fn batch_to_bits(batch: Batch<Self>) -> Batch<u32> {
        <NativeBackend as crate::backend::Backend>::transmute_f32_to_u32(batch)
    }
    #[inline(always)]
    fn batch_from_bits(batch: Batch<u32>) -> Batch<Self> {
        <NativeBackend as crate::backend::Backend>::transmute_u32_to_f32(batch)
    }
    #[inline(always)]
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self> {
        // Gather as u32 (transmuted slice) to avoid float index issues if logic was flawed,
        // but here we want to gather f32s.
        // If underlying gather takes Self indices, passing u32 indices to f32 gather is type error.
        // So we interpret slice as u32, gather u32s, then transmute back to f32.
        let u32_slice: &[u32] = unsafe { core::mem::transmute(slice) };
        let gathered = BatchArithmetic::gather(u32_slice, indices);
        <NativeBackend as crate::backend::Backend>::transmute_u32_to_f32(gathered)
    }
    #[inline(always)]
    fn batch_store(batch: Batch<Self>, slice: &mut [Self]) {
        SimdBatch::store(&batch, slice);
    }
}

impl Bitwise for u8 {
    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        v as u8
    }
    #[inline(always)]
    fn to_bits(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn batch_to_bits(batch: Batch<Self>) -> Batch<u32> {
        <NativeBackend as crate::backend::Backend>::upcast_u8_to_u32(batch)
    }
    #[inline(always)]
    fn batch_from_bits(batch: Batch<u32>) -> Batch<Self> {
        <NativeBackend as crate::backend::Backend>::downcast_u32_to_u8(batch)
    }
    #[inline(always)]
    fn batch_gather(slice: &[Self], indices: Batch<u32>) -> Batch<Self> {
        let gathered = BatchArithmetic::gather_u8(slice, indices);
        <NativeBackend as crate::backend::Backend>::downcast_u32_to_u8(gathered)
    }
    #[inline(always)]
    fn batch_store(batch: Batch<Self>, slice: &mut [Self]) {
        SimdBatch::store(&batch, slice);
    }
}
