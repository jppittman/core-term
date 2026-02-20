//! # Generic NNUE Model
//!
//! A generic Efficiently Updatable Neural Network.
//! It maps sparse feature indices to a scalar output.

extern crate alloc;
use alloc::vec::Vec;

/// NNUE network configuration.
#[derive(Clone)]
pub struct NnueConfig {
    /// Input feature space size.
    pub feature_count: usize,
    /// Size of the first hidden layer.
    pub l1_size: usize,
    /// Size of the second hidden layer.
    pub l2_size: usize,
    /// Size of the third hidden layer.
    pub l3_size: usize,
}

impl Default for NnueConfig {
    fn default() -> Self {
        Self {
            feature_count: 0, // Must be set
            l1_size: 256,
            l2_size: 32,
            l3_size: 32,
        }
    }
}

/// The NNUE network.
#[derive(Clone)]
pub struct Nnue {
    pub config: NnueConfig,
    /// First layer weights: [feature_count, l1_size]
    pub w1: Vec<i16>,
    /// First layer biases: [l1_size]
    pub b1: Vec<i32>,
    /// Second layer weights: [l1_size, l2_size]
    pub w2: Vec<i8>,
    /// Second layer biases: [l2_size]
    pub b2: Vec<i32>,
    /// Third layer weights: [l2_size, l3_size]
    pub w3: Vec<i8>,
    /// Third layer biases: [l3_size]
    pub b3: Vec<i32>,
    /// Output layer weights: [l3_size]
    pub w_out: Vec<i8>,
    /// Output layer bias
    pub b_out: i32,
}

impl Nnue {
    pub fn new(config: NnueConfig) -> Self {
        Self {
            w1: alloc::vec![0i16; config.feature_count * config.l1_size],
            b1: alloc::vec![0i32; config.l1_size],
            w2: alloc::vec![0i8; config.l1_size * config.l2_size],
            b2: alloc::vec![0i32; config.l2_size],
            w3: alloc::vec![0i8; config.l2_size * config.l3_size],
            b3: alloc::vec![0i32; config.l3_size],
            w_out: alloc::vec![0i8; config.l3_size],
            b_out: 0,
            config,
        }
    }
}

/// Accumulator for incremental updates.
#[derive(Clone)]
pub struct Accumulator {
    pub values: Vec<i32>,
}

impl Accumulator {
    pub fn new(nnue: &Nnue) -> Self {
        Self {
            values: nnue.b1.clone(),
        }
    }

    pub fn reset(&mut self, nnue: &Nnue) {
        self.values.copy_from_slice(&nnue.b1);
    }

    #[inline]
    pub fn add_feature(&mut self, nnue: &Nnue, feature_idx: usize) {
        let l1_size = nnue.config.l1_size;
        let offset = feature_idx * l1_size;
        for i in 0..l1_size {
            self.values[i] += nnue.w1[offset + i] as i32;
        }
    }

    #[inline]
    pub fn remove_feature(&mut self, nnue: &Nnue, feature_idx: usize) {
        let l1_size = nnue.config.l1_size;
        let offset = feature_idx * l1_size;
        for i in 0..l1_size {
            self.values[i] -= nnue.w1[offset + i] as i32;
        }
    }

    pub fn forward(&self, nnue: &Nnue) -> i32 {
        let l1_size = nnue.config.l1_size;
        let l2_size = nnue.config.l2_size;
        let l3_size = nnue.config.l3_size;

        // L1 -> L2
        let mut l2 = nnue.b2.clone();
        for i in 0..l1_size {
            let a = (self.values[i] >> 6).clamp(0, 127) as i8;
            for j in 0..l2_size {
                l2[j] += (a as i32) * (nnue.w2[i * l2_size + j] as i32);
            }
        }

        // L2 -> L3
        let mut l3 = nnue.b3.clone();
        for i in 0..l2_size {
            let a = (l2[i] >> 6).clamp(0, 127) as i8;
            for j in 0..l3_size {
                l3[j] += (a as i32) * (nnue.w3[i * l3_size + j] as i32);
            }
        }

        // L3 -> Output
        let mut output = nnue.b_out;
        for i in 0..l3_size {
            let a = (l3[i] >> 6).clamp(0, 127) as i8;
            output += (a as i32) * (nnue.w_out[i] as i32);
        }

        output
    }
}
