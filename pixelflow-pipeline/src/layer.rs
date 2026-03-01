//! General purpose neural network layers.


use vec;

/// A simple linear layer with fixed-point or floating-point weights.
#[derive(Clone, Debug)]
pub struct Linear<W = i16, B = i32> {
    pub weights: Vec<W>,
    pub bias: Vec<B>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear<i16, i32> {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: vec![0; in_features * out_features],
            bias: vec![0; out_features],
            in_features,
            out_features,
        }
    }

    /// Integer forward pass (used in NNUE inference).
    pub fn forward(&self, input: &[i32], output: &mut [i32]) {
        output.copy_from_slice(&self.bias);
        for i in 0..self.in_features {
            let val = input[i];
            if val == 0 { continue; }
            let offset = i * self.out_features;
            for j in 0..self.out_features {
                output[j] += val * (self.weights[offset + j] as i32);
            }
        }
    }
}

/// Clipped ReLU activation: (x >> 6).clamp(0, 127)
pub struct ReLU;

impl ReLU {
    #[inline(always)]
    pub fn apply(x: i32) -> i8 {
        ((x >> 6).clamp(0, 127)) as i8
    }

    #[inline(always)]
    pub fn derivative(x: i32) -> f32 {
        let shifted = x >> 6;
        if shifted > 0 && shifted < 127 { 1.0 } else { 0.0 }
    }
}

/// An efficiently updatable accumulator for sparse input layers.
#[derive(Clone, Debug)]
pub struct Accumulator<W = i16, B = i32> {
    pub values: Vec<B>,
    pub weights: Vec<W>,
    pub bias: Vec<B>,
    pub size: usize,
}

impl Accumulator<i16, i32> {
    pub fn new(input_dim: usize, output_size: usize) -> Self {
        Self {
            values: vec![0; output_size],
            weights: vec![0; input_dim * output_size],
            bias: vec![0; output_size],
            size: output_size,
        }
    }

    pub fn reset(&mut self) {
        self.values.copy_from_slice(&self.bias);
    }

    pub fn add_feature(&mut self, feature_idx: usize) {
        let offset = feature_idx * self.size;
        for i in 0..self.size {
            self.values[i] += self.weights[offset + i] as i32;
        }
    }

    pub fn remove_feature(&mut self, feature_idx: usize) {
        let offset = feature_idx * self.size;
        for i in 0..self.size {
            self.values[i] -= self.weights[offset + i] as i32;
        }
    }
}

/// A standard dense layer for small models.
pub struct DenseLayer<W = i8, B = i32> {
    pub weights: Vec<W>,
    pub bias: Vec<B>,
}
