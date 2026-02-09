#[cfg(test)]
mod debug_tests {
    use super::super::*;
    use pixelflow_core::combinators::{At, Texture};
    use pixelflow_core::{Field, Manifold, ManifoldExt};

    fn eval_scalar<M: Manifold<Output = Field>>(m: &M, x: f32, y: f32) -> f32 {
        let bound = At { inner: m, x, y, z: 0.0f32, w: 0.0f32 };
        let tex = Texture::from_manifold(&bound, 1, 1);
        tex.data()[0]
    }

    #[test]
    fn test_log2_behavior() {
        let u = 0.75f32;
        let v = 0.75f32;
        let max_uv = Field::from(u);

        let log2_val = max_uv.log2();
        let l2 = eval_scalar(&log2_val, u, v);

        let neg_log2 = log2_val.neg();
        let nl2 = eval_scalar(&neg_log2, u, v);

        let floor_val = neg_log2.floor();
        let fl = eval_scalar(&floor_val, u, v);

        let exp2_val = floor_val.exp2();
        let ex = eval_scalar(&exp2_val, u, v);

        println!("u={}", u);
        println!("log2(u) = {}", l2);
        println!("-log2(u) = {}", nl2);
        println!("floor(-log2(u)) = {}", fl);
        println!("exp2(floor(...)) = {}", ex);
    }
}
