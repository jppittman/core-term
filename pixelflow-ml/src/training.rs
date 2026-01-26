//! # Training Data Generation for NNUE Instruction Selection
//!
//! This module provides training data infrastructure for NNUE cost models.
//!
//! ## Recommended Workflow (Compiled Kernels)
//!
//! The preferred approach uses compiled SIMD kernels via criterion benchmarks:
//!
//! ```bash
//! # 1. Generate expression pairs and benchmark code
//! cargo run -p pixelflow-ml --example gen_kernels --features training -- --mode bwd
//!
//! # 2. Run benchmarks to measure actual compiled SIMD costs
//! cargo bench -p pixelflow-ml --bench generated_kernels
//!
//! # 3. Collect benchmark results into training data
//! cargo run -p pixelflow-ml --example collect_benchmark_costs --features training
//!
//! # 4. Train NNUE on the collected data
//! cargo run -p pixelflow-ml --example nnue_train_test --features training
//! ```
//!
//! This workflow measures **real compiled kernel costs** - no interpreter overhead.
//!
//! ## Binpack Format
//!
//! ```text
//! Header (16 bytes):
//!   - Magic: 0x4E4E5545 ("NNUE")
//!   - Version: u32
//!   - Sample count: u64
//!
//! Per sample:
//!   - Feature count: u16
//!   - Features: [u32; feature_count] (packed feature indices)
//!   - Cost (nanoseconds): u64
//!   - Best rewrite index: u16
//! ```

extern crate alloc;

use alloc::vec::Vec;

use crate::nnue::{HalfEPFeature};

// ============================================================================
// Training Data Structures
// ============================================================================

/// A training sample with real SIMD-measured cost.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    /// Packed feature indices (sorted for consistency).
    pub features: Vec<u32>,
    /// Execution cost in nanoseconds (from SIMD benchmark).
    pub cost_ns: u64,
    /// Index of the best rewrite found (or u16::MAX if none better).
    pub best_rewrite: u16,
    /// Cost improvement from best rewrite (negative = improvement).
    pub cost_delta_ns: i64,
}

impl TrainingSample {
    /// Create a new training sample from features and SIMD-measured cost.
    pub fn new(features: Vec<HalfEPFeature>, cost_ns: u64) -> Self {
        let mut packed: Vec<u32> = features.iter()
            .map(|f| f.to_index() as u32)
            .collect();
        packed.sort_unstable();
        packed.dedup();

        Self {
            features: packed,
            cost_ns,
            best_rewrite: u16::MAX,
            cost_delta_ns: 0,
        }
    }
}

// ============================================================================
// Binary Format I/O
// ============================================================================

/// Magic number for binpack files.
pub const BINPACK_MAGIC: u32 = 0x4E4E5545; // "NNUE"

/// Current binpack format version.
pub const BINPACK_VERSION: u32 = 2; // v2: real SIMD costs

/// Write samples to binpack format for offline training.
#[cfg(feature = "std")]
pub fn write_binpack(path: &std::path::Path, samples: &[TrainingSample]) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;

    // Header
    file.write_all(&BINPACK_MAGIC.to_le_bytes())?;
    file.write_all(&BINPACK_VERSION.to_le_bytes())?;
    file.write_all(&(samples.len() as u64).to_le_bytes())?;

    // Samples
    for sample in samples {
        file.write_all(&(sample.features.len() as u16).to_le_bytes())?;
        for &f in &sample.features {
            file.write_all(&f.to_le_bytes())?;
        }
        file.write_all(&sample.cost_ns.to_le_bytes())?;
        file.write_all(&sample.best_rewrite.to_le_bytes())?;
        file.write_all(&sample.cost_delta_ns.to_le_bytes())?;
    }

    Ok(())
}

/// Read samples from binpack format.
#[cfg(feature = "std")]
pub fn read_binpack(path: &std::path::Path) -> std::io::Result<Vec<TrainingSample>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;

    // Read header
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if u32::from_le_bytes(magic) != BINPACK_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid magic number",
        ));
    }

    let mut version = [0u8; 4];
    file.read_exact(&mut version)?;
    let ver = u32::from_le_bytes(version);
    if ver != BINPACK_VERSION && ver != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported version: {}", ver),
        ));
    }

    let mut count = [0u8; 8];
    file.read_exact(&mut count)?;
    let sample_count = u64::from_le_bytes(count) as usize;

    // Read samples
    let mut samples = Vec::with_capacity(sample_count);

    for _ in 0..sample_count {
        let mut fc = [0u8; 2];
        file.read_exact(&mut fc)?;
        let feature_count = u16::from_le_bytes(fc) as usize;

        let mut features = Vec::with_capacity(feature_count);
        for _ in 0..feature_count {
            let mut f = [0u8; 4];
            file.read_exact(&mut f)?;
            features.push(u32::from_le_bytes(f));
        }

        let mut c = [0u8; 8];
        file.read_exact(&mut c)?;
        let cost_ns = u64::from_le_bytes(c);

        let mut br = [0u8; 2];
        file.read_exact(&mut br)?;
        let best_rewrite = u16::from_le_bytes(br);

        let mut cd = [0u8; 8];
        file.read_exact(&mut cd)?;
        let cost_delta_ns = i64::from_le_bytes(cd);

        samples.push(TrainingSample {
            features,
            cost_ns,
            best_rewrite,
            cost_delta_ns,
        });
    }

    Ok(samples)
}

/// Append samples to an existing binpack file.
#[cfg(feature = "std")]
pub fn append_binpack(path: &std::path::Path, new_samples: &[TrainingSample]) -> std::io::Result<()> {
    // Read existing samples
    let mut existing = if path.exists() {
        read_binpack(path)?
    } else {
        Vec::new()
    };

    // Append new samples
    existing.extend(new_samples.iter().cloned());

    // Rewrite file
    write_binpack(path, &existing)
}

// ============================================================================
// Dataset Statistics
// ============================================================================

/// Statistics about a training dataset.
#[derive(Clone, Debug)]
pub struct DatasetStats {
    /// Number of samples.
    pub sample_count: usize,
    /// Average feature count per sample.
    pub avg_features: f64,
    /// Average cost in nanoseconds.
    pub avg_cost_ns: f64,
    /// Min cost observed.
    pub min_cost_ns: u64,
    /// Max cost observed.
    pub max_cost_ns: u64,
    /// Samples with improving rewrites.
    pub samples_with_improvement: usize,
    /// Average cost improvement when there is one.
    pub avg_improvement_ns: f64,
}

impl DatasetStats {
    /// Compute statistics from a dataset.
    pub fn from_samples(samples: &[TrainingSample]) -> Self {
        if samples.is_empty() {
            return Self {
                sample_count: 0,
                avg_features: 0.0,
                avg_cost_ns: 0.0,
                min_cost_ns: 0,
                max_cost_ns: 0,
                samples_with_improvement: 0,
                avg_improvement_ns: 0.0,
            };
        }

        let total_features: usize = samples.iter().map(|s| s.features.len()).sum();
        let total_cost: u64 = samples.iter().map(|s| s.cost_ns).sum();
        let min_cost = samples.iter().map(|s| s.cost_ns).min().unwrap_or(0);
        let max_cost = samples.iter().map(|s| s.cost_ns).max().unwrap_or(0);

        let improving: Vec<_> = samples.iter()
            .filter(|s| s.cost_delta_ns < 0)
            .collect();

        let total_improvement: i64 = improving.iter()
            .map(|s| -s.cost_delta_ns)
            .sum();

        Self {
            sample_count: samples.len(),
            avg_features: total_features as f64 / samples.len() as f64,
            avg_cost_ns: total_cost as f64 / samples.len() as f64,
            min_cost_ns: min_cost,
            max_cost_ns: max_cost,
            samples_with_improvement: improving.len(),
            avg_improvement_ns: if improving.is_empty() {
                0.0
            } else {
                total_improvement as f64 / improving.len() as f64
            },
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_sample_dedup() {
        let features = vec![
            HalfEPFeature { perspective_op: 0, descendant_op: 1, depth: 0, path: 0 },
            HalfEPFeature { perspective_op: 0, descendant_op: 1, depth: 0, path: 0 }, // duplicate
            HalfEPFeature { perspective_op: 0, descendant_op: 2, depth: 0, path: 0 },
        ];
        let sample = TrainingSample::new(features, 100);
        assert_eq!(sample.features.len(), 2);
    }

    #[test]
    fn test_dataset_stats() {
        let samples = vec![
            TrainingSample {
                features: vec![1, 2, 3],
                cost_ns: 100,
                best_rewrite: 0,
                cost_delta_ns: -10,
            },
            TrainingSample {
                features: vec![4, 5],
                cost_ns: 200,
                best_rewrite: u16::MAX,
                cost_delta_ns: 0,
            },
        ];

        let stats = DatasetStats::from_samples(&samples);
        assert_eq!(stats.sample_count, 2);
        assert!((stats.avg_features - 2.5).abs() < 0.01);
        assert!((stats.avg_cost_ns - 150.0).abs() < 0.01);
        assert_eq!(stats.samples_with_improvement, 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_binpack_roundtrip() {
        let samples = vec![
            TrainingSample {
                features: vec![1, 2, 3],
                cost_ns: 42,
                best_rewrite: 5,
                cost_delta_ns: -10,
            },
        ];

        let path = std::path::Path::new("/tmp/test_binpack.bin");
        write_binpack(path, &samples).unwrap();
        let loaded = read_binpack(path).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].features, vec![1, 2, 3]);
        assert_eq!(loaded[0].cost_ns, 42);
        assert_eq!(loaded[0].best_rewrite, 5);
        assert_eq!(loaded[0].cost_delta_ns, -10);

        std::fs::remove_file(path).ok();
    }
}
