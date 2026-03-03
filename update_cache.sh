#!/bin/bash
sed -i '1s/^/const BUCKET_STEP_F32: f32 = 4.0;\nconst BUCKET_STEP_SIZE: usize = 4;\nconst MIN_BUCKET_SIZE_VAL: usize = 8;\nconst BYTES_PER_PIXEL: usize = 4;\n\n/' pixelflow-graphics/src/fonts/cache.rs

sed -i 's/let bucket = ((size \/ 4.0).ceil() as usize) \* 4;/let bucket = ((size \/ BUCKET_STEP_F32).ceil() as usize) * BUCKET_STEP_SIZE;/g' pixelflow-graphics/src/fonts/cache.rs
sed -i 's/bucket.max(8)/bucket.max(MIN_BUCKET_SIZE_VAL)/g' pixelflow-graphics/src/fonts/cache.rs
sed -i 's/\.map(|g| g.width \* g.height \* 4) \/\/ f32 per pixel/.map(|g| g.width * g.height * BYTES_PER_PIXEL)/g' pixelflow-graphics/src/fonts/cache.rs
