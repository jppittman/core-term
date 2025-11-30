extern crate alloc;
use alloc::sync::Arc;
use pixelflow_core::batch::{Batch, SimdFloatOps};
use pixelflow_core::pipe::Surface;
use crate::curves::Segment;

#[derive(Clone)]
pub struct CurveSurface {
    pub segments: Arc<[Segment]>,
    pub scale: f32,
}

impl Surface<u8> for CurveSurface {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        // Convert u32 coords to f32 + 0.5 (center of pixel)
        // x and y are u32. We need to cast to f32.
        // SimdVec<u32> to SimdVec<f32> conversion is usually cvt.
        // But we don't have a cast method from Batch<u32> to Batch<f32> yet.
        // We can use unsafe transmute if we assume layout is compatible (it is), but values are wrong.
        // We need `to_f32()` conversion method.
        // Or unpack and repack.
        // Or implement `from_u32` in SimdFloatOps.

        // Unpack for now, until we add conversion support to core.
        // "Fake SIMD" for conversion only?
        // Actually, x86 has `_mm_cvtepi32_ps`. We should expose it.
        // But I don't want to modify core again if I can avoid it.
        // Let's use unpack/repack for coord conversion (it's once per batch).

        let xa = x.to_array_usize();
        let ya = y.to_array_usize();
        let xf = Batch::<f32>::new(xa[0] as f32 + 0.5, xa[1] as f32 + 0.5, xa[2] as f32 + 0.5, xa[3] as f32 + 0.5);
        let yf = Batch::<f32>::new(ya[0] as f32 + 0.5, ya[1] as f32 + 0.5, ya[2] as f32 + 0.5, ya[3] as f32 + 0.5);

        let mut winding = Batch::<f32>::splat(0.0);
        let mut min_dist = Batch::<f32>::splat(1e9);

        for seg in self.segments.iter() {
            winding = winding + seg.winding_batch(xf, yf);

            let dist = seg.min_dist_batch(xf, yf);
            let dist_abs = dist.abs();
            let mask = dist_abs.lt(min_dist.abs());
            min_dist = dist.select(min_dist, mask);
        }

        // Finalize
        let scale_vec = Batch::splat(self.scale);
        let d = min_dist * scale_vec;

        // if winding != 0 (inside) -> -d.abs(), else d.abs()
        let inside = winding.abs().gt(Batch::splat(0.5)); // winding is integer-ish float (0.0, 1.0, -1.0)

        let signed_dist = d.abs().select(d.abs() * Batch::splat(-1.0), inside);

        let alpha = (Batch::splat(0.5) - signed_dist).max(Batch::splat(0.0)).min(Batch::splat(1.0));

        // Convert alpha (0.0..1.0) to u8 (0..255)
        let alpha255 = alpha * Batch::splat(255.0);

        // Cast back to u32 then u8.
        // Again, missing f32->u32 cast in batch.
        // Unpack/repack.
        // We can do `alpha255.cast::<u32>()` which does bitcast. That's wrong.
        // We need value conversion.

        // Hack: Unpack
        let mut buf = [0.0f32; 4];
        unsafe { alpha255.store(buf.as_mut_ptr()) };
        let b = Batch::<u32>::new(buf[0] as u32, buf[1] as u32, buf[2] as u32, buf[3] as u32);

        b.cast()
    }
}
