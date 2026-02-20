//! DataSource implementations.

extern crate alloc;

use alloc::vec::Vec;

use crate::nnue::{Expr, ExprGenConfig, ExprGenerator, BwdGenConfig, BwdGenerator};
use super::DataSource;

/// Generate random expressions.
pub struct GeneratedData {
    generator: ExprGenerator,
    generated: usize,
    max: Option<usize>,
}

impl GeneratedData {
    pub fn new(seed: u64, config: ExprGenConfig, max: Option<usize>) -> Self {
        Self {
            generator: ExprGenerator::new(seed, config),
            generated: 0,
            max,
        }
    }
}

impl DataSource for GeneratedData {
    fn next_batch(&mut self, count: usize) -> Vec<Expr> {
        let actual = match self.max {
            Some(max) => count.min(max.saturating_sub(self.generated)),
            None => count,
        };
        self.generated += actual;
        (0..actual).map(|_| self.generator.generate()).collect()
    }

    fn reset(&mut self) {
        self.generated = 0;
    }

    fn len(&self) -> Option<usize> {
        self.max
    }
}

/// Generate backward (optimized → unoptimized) expression pairs.
pub struct BwdGeneratedData {
    generator: BwdGenerator,
    generated: usize,
    max: Option<usize>,
}

impl BwdGeneratedData {
    pub fn new(seed: u64, config: BwdGenConfig, max: Option<usize>) -> Self {
        Self {
            generator: BwdGenerator::new(seed, config),
            generated: 0,
            max,
        }
    }
}

impl DataSource for BwdGeneratedData {
    fn next_batch(&mut self, count: usize) -> Vec<Expr> {
        let actual = match self.max {
            Some(max) => count.min(max.saturating_sub(self.generated)),
            None => count,
        };
        self.generated += actual;
        (0..actual)
            .map(|_| self.generator.generate().unoptimized)
            .collect()
    }

    fn reset(&mut self) {
        self.generated = 0;
    }

    fn len(&self) -> Option<usize> {
        self.max
    }
}

/// Load expressions from a pre-loaded vector.
pub struct VecData {
    exprs: Vec<Expr>,
    index: usize,
}

impl VecData {
    pub fn new(exprs: Vec<Expr>) -> Self {
        Self { exprs, index: 0 }
    }
}

impl DataSource for VecData {
    fn next_batch(&mut self, count: usize) -> Vec<Expr> {
        let remaining = self.exprs.len().saturating_sub(self.index);
        let actual = count.min(remaining);
        let batch = self.exprs[self.index..self.index + actual].to_vec();
        self.index += actual;
        batch
    }

    fn reset(&mut self) {
        self.index = 0;
    }

    fn len(&self) -> Option<usize> {
        Some(self.exprs.len())
    }
}

/// Chain multiple data sources.
pub struct ChainedData {
    sources: Vec<Box<dyn DataSource>>,
    current: usize,
}

impl ChainedData {
    pub fn new(sources: Vec<Box<dyn DataSource>>) -> Self {
        Self { sources, current: 0 }
    }
}

impl DataSource for ChainedData {
    fn next_batch(&mut self, count: usize) -> Vec<Expr> {
        let mut result = Vec::with_capacity(count);
        while result.len() < count && self.current < self.sources.len() {
            let batch = self.sources[self.current].next_batch(count - result.len());
            if batch.is_empty() {
                self.current += 1;
            } else {
                result.extend(batch);
            }
        }
        result
    }

    fn reset(&mut self) {
        self.current = 0;
        for source in &mut self.sources {
            source.reset();
        }
    }

    fn len(&self) -> Option<usize> {
        let mut total = 0usize;
        for source in &self.sources {
            match source.len() {
                Some(n) => total += n,
                None => return None,
            }
        }
        Some(total)
    }
}
