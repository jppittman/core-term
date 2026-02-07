//! Forward and backward passes for NNUE training.
//!
//! Implements full backpropagation through the NNUE architecture:
//! - Sparse branch (HalfEP features -> L1)
//! - Dense branch (ILP features -> L_dense)
//! - Combined layers (L2 -> L3 -> Output)

use pixelflow_nnue::{
    DenseFeatures, HalfEPFeature, Nnue, OUTPUT_SCALE, QUANTIZATION_SHIFT, RELU_MAX,
};

// ============================================================================
// Training Hyperparameters
// ============================================================================

const LEAKY_RELU_SLOPE_DEAD: f32 = 0.01;
const LEAKY_RELU_SLOPE_SATURATED: f32 = 0.1;
const LR_SCALE_HIDDEN: f32 = 0.01;
const LR_SCALE_INPUT: f32 = 0.1;
const WEIGHT_CLIP_I8: f32 = 127.0;
const WEIGHT_CLIP_I16: f32 = 32767.0;

/// Clipped ReLU activation: (x >> 6).clamp(0, 127)
///
/// Returns both the output and the derivative.
/// Uses "leaky" gradient (0.01) for dead neurons to allow recovery during training.
/// Full gradient (1.0) in active region [0, 127].
/// Small gradient (0.1) in saturated region (>127) for soft clamping.
#[inline]
pub fn clipped_relu(x: i32) -> (f32, f32) {
    let shifted = x >> QUANTIZATION_SHIFT;
    let clamped = shifted.clamp(0, RELU_MAX);
    // Leaky gradient allows dead neurons to recover
    let deriv = if shifted <= 0 {
        LEAKY_RELU_SLOPE_DEAD // Dead neuron - small gradient for recovery
    } else if shifted >= RELU_MAX {
        LEAKY_RELU_SLOPE_SATURATED // Saturated - reduced but non-zero gradient
    } else {
        1.0 // Active region - full gradient
    };
    (clamped as f32, deriv)
}

/// Intermediate activations for sparse-only forward pass.
#[derive(Clone)]
pub struct ForwardState {
    pub l1_pre: Vec<i32>,
    pub l1_post: Vec<f32>,
    pub l2_pre: Vec<i32>,
    pub l2_post: Vec<f32>,
    pub l3_pre: Vec<i32>,
    pub l3_post: Vec<f32>,
    pub active_features: Vec<usize>,
}

/// Intermediate activations for hybrid forward pass (sparse + dense).
#[derive(Clone)]
pub struct HybridForwardState {
    pub l1_pre: Vec<i32>,
    pub l1_post: Vec<f32>,
    pub dense_input: Vec<i32>,  // Input to dense branch (for W_dense gradients)
    pub dense_pre: Vec<i32>,    // Pre-activation (for derivative)
    pub dense_post: Vec<f32>,
    pub l2_pre: Vec<i32>,
    pub l2_post: Vec<f32>,
    pub l3_pre: Vec<i32>,
    pub l3_post: Vec<f32>,
    pub active_features: Vec<usize>,
}

/// Forward pass (sparse-only) that stores intermediate activations.
pub fn forward_with_state(
    nnue: &Nnue,
    features: &[HalfEPFeature],
) -> (f32, ForwardState) {
    let l1_size = nnue.config.l1_size;
    let l2_size = nnue.config.l2_size;
    let l3_size = nnue.config.l3_size;

    let active_features: Vec<usize> = features
        .iter()
        .map(|f| f.to_index())
        .filter(|&idx| idx < HalfEPFeature::COUNT)
        .collect();

    // L1: Sparse input via accumulator
    let mut l1_pre = nnue.b1.clone();
    for &feature_idx in &active_features {
        let offset = feature_idx * l1_size;
        for i in 0..l1_size {
            l1_pre[i] += nnue.w1[offset + i] as i32;
        }
    }

    let mut l1_post = Vec::with_capacity(l1_size);
    for &x in &l1_pre {
        let (out, _) = clipped_relu(x);
        l1_post.push(out);
    }

    // L2
    let mut l2_pre = nnue.b2.clone();
    for i in 0..l1_size {
        let a = l1_post[i] as i8 as i32;
        for j in 0..l2_size {
            l2_pre[j] += a * (nnue.w2[i * l2_size + j] as i32);
        }
    }

    let mut l2_post = Vec::with_capacity(l2_size);
    for &x in &l2_pre {
        let (out, _) = clipped_relu(x);
        l2_post.push(out);
    }

    // L3
    let mut l3_pre = nnue.b3.clone();
    for i in 0..l2_size {
        let a = l2_post[i] as i8 as i32;
        for j in 0..l3_size {
            l3_pre[j] += a * (nnue.w3[i * l3_size + j] as i32);
        }
    }

    let mut l3_post = Vec::with_capacity(l3_size);
    for &x in &l3_pre {
        let (out, _) = clipped_relu(x);
        l3_post.push(out);
    }

    // Output
    let mut output = nnue.b_out;
    for i in 0..l3_size {
        let a = l3_post[i] as i8 as i32;
        output += a * (nnue.w_out[i] as i32);
    }

    let state = ForwardState {
        l1_pre,
        l1_post,
        l2_pre,
        l2_post,
        l3_pre,
        l3_post,
        active_features,
    };

    (output as f32 / OUTPUT_SCALE, state)
}

/// Forward pass with hybrid architecture (sparse + dense ILP features).
pub fn forward_with_state_hybrid(
    nnue: &Nnue,
    features: &[HalfEPFeature],
    dense: &DenseFeatures,
) -> (f32, HybridForwardState) {
    let l1_size = nnue.config.l1_size;
    let dense_size = nnue.config.dense_size;
    let l2_size = nnue.config.l2_size;
    let l3_size = nnue.config.l3_size;

    let active_features: Vec<usize> = features
        .iter()
        .map(|f| f.to_index())
        .filter(|&idx| idx < HalfEPFeature::COUNT)
        .collect();

    // Sparse branch: L1
    let mut l1_pre = nnue.b1.clone();
    for &feature_idx in &active_features {
        let offset = feature_idx * l1_size;
        for i in 0..l1_size {
            l1_pre[i] += nnue.w1[offset + i] as i32;
        }
    }

    let mut l1_post = Vec::with_capacity(l1_size);
    for &x in &l1_pre {
        let (out, _) = clipped_relu(x);
        l1_post.push(out);
    }

    // Dense branch - store inputs and pre-activations for backprop
    let dense_input: Vec<i32> = dense.values.to_vec();
    let mut dense_pre = Vec::with_capacity(dense_size);
    let mut dense_post = Vec::with_capacity(dense_size);
    for j in 0..dense_size {
        let mut sum = nnue.b_dense[j];
        for i in 0..DenseFeatures::COUNT {
            sum += dense.values[i] * (nnue.w_dense[i * dense_size + j] as i32);
        }
        dense_pre.push(sum);
        let (out, _) = clipped_relu(sum);
        dense_post.push(out);
    }

    // L2: Combined
    let mut l2_pre = nnue.b2.clone();

    for i in 0..l1_size {
        let a = l1_post[i] as i8 as i32;
        for j in 0..l2_size {
            l2_pre[j] += a * (nnue.w2[i * l2_size + j] as i32);
        }
    }

    for i in 0..dense_size {
        let a = dense_post[i] as i8 as i32;
        for j in 0..l2_size {
            l2_pre[j] += a * (nnue.w2[(l1_size + i) * l2_size + j] as i32);
        }
    }

    let mut l2_post = Vec::with_capacity(l2_size);
    for &x in &l2_pre {
        let (out, _) = clipped_relu(x);
        l2_post.push(out);
    }

    // L3
    let mut l3_pre = nnue.b3.clone();
    for i in 0..l2_size {
        let a = l2_post[i] as i8 as i32;
        for j in 0..l3_size {
            l3_pre[j] += a * (nnue.w3[i * l3_size + j] as i32);
        }
    }

    let mut l3_post = Vec::with_capacity(l3_size);
    for &x in &l3_pre {
        let (out, _) = clipped_relu(x);
        l3_post.push(out);
    }

    // Output
    let mut output = nnue.b_out;
    for i in 0..l3_size {
        let a = l3_post[i] as i8 as i32;
        output += a * (nnue.w_out[i] as i32);
    }

    let state = HybridForwardState {
        l1_pre,
        l1_post,
        dense_input,
        dense_pre,
        dense_post,
        l2_pre,
        l2_post,
        l3_pre,
        l3_post,
        active_features,
    };

    (output as f32 / OUTPUT_SCALE, state)
}

/// Backward pass for sparse-only architecture.
pub fn backward(
    nnue: &mut Nnue,
    state: &ForwardState,
    error: f32,
    lr: f32,
) {
    let l1_size = nnue.config.l1_size;
    let l2_size = nnue.config.l2_size;
    let l3_size = nnue.config.l3_size;

    let d_output = error;

    // Output layer
    nnue.b_out -= (d_output * lr * OUTPUT_SCALE) as i32;

    let mut d_l3_post = vec![0.0f32; l3_size];
    for i in 0..l3_size {
        d_l3_post[i] = d_output * (nnue.w_out[i] as f32);
        let grad_w_out = d_output * state.l3_post[i];
        let update = (grad_w_out * lr).clamp(-WEIGHT_CLIP_I8, WEIGHT_CLIP_I8) as i8;
        nnue.w_out[i] = nnue.w_out[i].saturating_sub(update);
    }

    // L3
    let mut d_l3_pre = vec![0.0f32; l3_size];
    for i in 0..l3_size {
        let (_, deriv) = clipped_relu(state.l3_pre[i]);
        d_l3_pre[i] = d_l3_post[i] * deriv;
    }

    for j in 0..l3_size {
        nnue.b3[j] -= (d_l3_pre[j] * lr * 1.0) as i32;
    }

    let mut d_l2_post = vec![0.0f32; l2_size];
    for i in 0..l2_size {
        for j in 0..l3_size {
            d_l2_post[i] += d_l3_pre[j] * (nnue.w3[i * l3_size + j] as f32);
            let grad_w3 = d_l3_pre[j] * state.l2_post[i];
            let update = (grad_w3 * lr * LR_SCALE_HIDDEN).clamp(-WEIGHT_CLIP_I8, WEIGHT_CLIP_I8)
                as i8;
            nnue.w3[i * l3_size + j] = nnue.w3[i * l3_size + j].saturating_sub(update);
        }
    }

    // L2
    let mut d_l2_pre = vec![0.0f32; l2_size];
    for i in 0..l2_size {
        let (_, deriv) = clipped_relu(state.l2_pre[i]);
        d_l2_pre[i] = d_l2_post[i] * deriv;
    }

    for j in 0..l2_size {
        nnue.b2[j] -= (d_l2_pre[j] * lr * 1.0) as i32;
    }

    let mut d_l1_post = vec![0.0f32; l1_size];
    for i in 0..l1_size {
        for j in 0..l2_size {
            d_l1_post[i] += d_l2_pre[j] * (nnue.w2[i * l2_size + j] as f32);
            let grad_w2 = d_l2_pre[j] * state.l1_post[i];
            let update = (grad_w2 * lr * LR_SCALE_HIDDEN).clamp(-WEIGHT_CLIP_I8, WEIGHT_CLIP_I8)
                as i8;
            nnue.w2[i * l2_size + j] = nnue.w2[i * l2_size + j].saturating_sub(update);
        }
    }

    // L1
    let mut d_l1_pre = vec![0.0f32; l1_size];
    for i in 0..l1_size {
        let (_, deriv) = clipped_relu(state.l1_pre[i]);
        d_l1_pre[i] = d_l1_post[i] * deriv;
    }

    for i in 0..l1_size {
        nnue.b1[i] -= (d_l1_pre[i] * lr * 1.0) as i32;
    }

    // W1 sparse update
    for &feature_idx in &state.active_features {
        let offset = feature_idx * l1_size;
        for i in 0..l1_size {
            let grad_w1 = d_l1_pre[i];
            let update = (grad_w1 * lr * LR_SCALE_INPUT)
                .clamp(-WEIGHT_CLIP_I16, WEIGHT_CLIP_I16) as i16;
            nnue.w1[offset + i] = nnue.w1[offset + i].saturating_sub(update);
        }
    }
}

/// Backward pass for hybrid architecture (sparse + dense).
pub fn backward_hybrid(
    nnue: &mut Nnue,
    state: &HybridForwardState,
    error: f32,
    lr: f32,
) {
    let l1_size = nnue.config.l1_size;
    let dense_size = nnue.config.dense_size;
    let l2_size = nnue.config.l2_size;
    let l3_size = nnue.config.l3_size;

    let d_output = error;

    // Output layer
    nnue.b_out -= (d_output * lr * OUTPUT_SCALE) as i32;

    let mut d_l3_post = vec![0.0f32; l3_size];
    for i in 0..l3_size {
        d_l3_post[i] = d_output * (nnue.w_out[i] as f32);
        let grad_w_out = d_output * state.l3_post[i];
        let update = (grad_w_out * lr).clamp(-WEIGHT_CLIP_I8, WEIGHT_CLIP_I8) as i8;
        nnue.w_out[i] = nnue.w_out[i].saturating_sub(update);
    }

    // L3
    let mut d_l3_pre = vec![0.0f32; l3_size];
    for i in 0..l3_size {
        let (_, deriv) = clipped_relu(state.l3_pre[i]);
        d_l3_pre[i] = d_l3_post[i] * deriv;
    }

    for j in 0..l3_size {
        nnue.b3[j] -= (d_l3_pre[j] * lr * 1.0) as i32;
    }

    let mut d_l2_post = vec![0.0f32; l2_size];
    for i in 0..l2_size {
        for j in 0..l3_size {
            d_l2_post[i] += d_l3_pre[j] * (nnue.w3[i * l3_size + j] as f32);
            let grad_w3 = d_l3_pre[j] * state.l2_post[i];
            let update = (grad_w3 * lr * LR_SCALE_HIDDEN).clamp(-WEIGHT_CLIP_I8, WEIGHT_CLIP_I8)
                as i8;
            nnue.w3[i * l3_size + j] = nnue.w3[i * l3_size + j].saturating_sub(update);
        }
    }

    // L2
    let mut d_l2_pre = vec![0.0f32; l2_size];
    for i in 0..l2_size {
        let (_, deriv) = clipped_relu(state.l2_pre[i]);
        d_l2_pre[i] = d_l2_post[i] * deriv;
    }

    for j in 0..l2_size {
        nnue.b2[j] -= (d_l2_pre[j] * lr * 1.0) as i32;
    }

    // Gradient into sparse branch
    let mut d_l1_post = vec![0.0f32; l1_size];
    for i in 0..l1_size {
        for j in 0..l2_size {
            d_l1_post[i] += d_l2_pre[j] * (nnue.w2[i * l2_size + j] as f32);
            let grad_w2 = d_l2_pre[j] * state.l1_post[i];
            let update = (grad_w2 * lr * LR_SCALE_HIDDEN).clamp(-WEIGHT_CLIP_I8, WEIGHT_CLIP_I8)
                as i8;
            nnue.w2[i * l2_size + j] = nnue.w2[i * l2_size + j].saturating_sub(update);
        }
    }

    // Gradient into dense branch
    let mut d_dense_post = vec![0.0f32; dense_size];
    for i in 0..dense_size {
        for j in 0..l2_size {
            d_dense_post[i] += d_l2_pre[j] * (nnue.w2[(l1_size + i) * l2_size + j] as f32);
            let grad_w2 = d_l2_pre[j] * state.dense_post[i];
            let update = (grad_w2 * lr * LR_SCALE_HIDDEN).clamp(-WEIGHT_CLIP_I8, WEIGHT_CLIP_I8)
                as i8;
            nnue.w2[(l1_size + i) * l2_size + j] =
                nnue.w2[(l1_size + i) * l2_size + j].saturating_sub(update);
        }
    }

    // L1 (sparse)
    let mut d_l1_pre = vec![0.0f32; l1_size];
    for i in 0..l1_size {
        let (_, deriv) = clipped_relu(state.l1_pre[i]);
        d_l1_pre[i] = d_l1_post[i] * deriv;
    }

    for i in 0..l1_size {
        nnue.b1[i] -= (d_l1_pre[i] * lr * 1.0) as i32;
    }

    // W1 sparse update
    for &feature_idx in &state.active_features {
        let offset = feature_idx * l1_size;
        for i in 0..l1_size {
            let grad_w1 = d_l1_pre[i];
            let update = (grad_w1 * lr * LR_SCALE_INPUT)
                .clamp(-WEIGHT_CLIP_I16, WEIGHT_CLIP_I16) as i16;
            nnue.w1[offset + i] = nnue.w1[offset + i].saturating_sub(update);
        }
    }

    // Dense branch: compute pre-activation gradients using leaky ReLU
    let mut d_dense_pre = vec![0.0f32; dense_size];
    for j in 0..dense_size {
        let (_, deriv) = clipped_relu(state.dense_pre[j]);
        d_dense_pre[j] = d_dense_post[j] * deriv;
    }

    // Update dense biases
    for j in 0..dense_size {
        nnue.b_dense[j] -= (d_dense_pre[j] * lr * 1.0) as i32;
    }

    // Update W_dense weights: grad = d_dense_pre * dense_input
    for i in 0..DenseFeatures::COUNT {
        let input_val = state.dense_input[i] as f32;
        if input_val == 0.0 {
            continue; // Skip zero inputs for efficiency
        }
        for j in 0..dense_size {
            let grad_w_dense = d_dense_pre[j] * input_val;
            let update = (grad_w_dense * lr * LR_SCALE_HIDDEN)
                .clamp(-WEIGHT_CLIP_I16, WEIGHT_CLIP_I16) as i16;
            nnue.w_dense[i * dense_size + j] =
                nnue.w_dense[i * dense_size + j].saturating_sub(update);
        }
    }
}
