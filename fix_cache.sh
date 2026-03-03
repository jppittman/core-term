#!/bin/bash
# Remove first 4 lines
sed -i '1,4d' pixelflow-graphics/src/fonts/cache.rs

# Insert after the last use crate::Grayscale;
sed -i '/use crate::Grayscale;/a\
\
const BUCKET_STEP_F32: f32 = 4.0;\
const BUCKET_STEP_SIZE: usize = 4;\
const MIN_BUCKET_SIZE_VAL: usize = 8;\
const BYTES_PER_PIXEL: usize = 4;\
' pixelflow-graphics/src/fonts/cache.rs
