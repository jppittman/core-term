use pixelflow_core::curve::Mat3;

fn naive_inverse(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-6 {
        return None;
    }

    let inv_det = 1.0 / det;
    let mut inv = [[0.0; 3]; 3];

    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;

    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

    Some(inv)
}

#[test]
fn test_mat3_inverse_correctness() {
    let test_cases = vec![
        // Identity
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        // Scaling
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        // Translation (affine)
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
        // Rotation/Shear mix
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        // Arbitrary
        [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]],
        // Singular
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]],
    ];

    for (i, &m_arr) in test_cases.iter().enumerate() {
        let mat = Mat3 { m: m_arr };
        let naive = naive_inverse(m_arr);
        let optimized = mat.inverse();

        match (naive, optimized) {
            (Some(n), Some(o)) => {
                for r in 0..3 {
                    for c in 0..3 {
                        let diff = (n[r][c] - o.m[r][c]).abs();
                        assert!(
                            diff < 1e-5,
                            "Mismatch at case #{}, elem [{}][{}]: naive={}, optimized={}",
                            i,
                            r,
                            c,
                            n[r][c],
                            o.m[r][c]
                        );
                    }
                }
            }
            (None, None) => {}
            _ => panic!("Mismatch in singularity detection at case #{}", i),
        }
    }
}
