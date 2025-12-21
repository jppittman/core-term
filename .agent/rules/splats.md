---
trigger: always_on
---

Do not export leak SIMD from pixelflow-core

Batch::splat and Field::splat are pub(crate). DO NOT EXPOSE THEM. Do not expose SimdVecs. Do not expose anything that hints at lanes.

SIMD is an implemntation detail. pixelflow-core is an algebra. Writing it is supposed to look like Halide, not assembly.