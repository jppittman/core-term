# Deep Learning for Symbolic Mathematics

**Authors:** Guillaume Lample, François Charton
**Affiliation:** Facebook AI Research
**Published:** ICLR 2020
**arXiv:** [1912.01412](https://arxiv.org/abs/1912.01412)
**Code:** [github.com/facebookresearch/SymbolicMathematics](https://github.com/facebookresearch/SymbolicMathematics)

## Abstract

Neural networks have a reputation for being better at solving statistical or
approximate problems than at performing calculations or working with symbolic
data. This paper shows they can be surprisingly good at more elaborated tasks
in mathematics, such as symbolic integration and solving differential equations.

The authors propose:
1. A syntax for representing mathematical problems as sequences
2. Methods for generating large datasets for training seq2seq models
3. Results that outperform commercial CAS (Matlab, Mathematica)

## Key Insight: Backward Generation (BWD)

The central contribution relevant to our work is **Backward Generation (BWD)**
for creating training data.

### The Problem with Forward Generation (FWD)

Forward generation creates training pairs by:
1. Generate random expression f
2. Apply symbolic solver to get result g
3. Train on (f, g)

**Issues:**
- Depends on external symbolic solver
- Dataset limited to what the solver can handle
- Expensive computation for complex expressions

### Backward Generation Solution

BWD reverses the process:
1. Generate random function f (the "answer")
2. Apply a cheap, deterministic transformation to get f' (the "question")
3. Train on (f', f)

**For integration:**
- Generate random f
- Compute derivative f' (always possible, always fast)
- Train model to integrate: given f', predict f

**Key properties:**
- No external solver needed
- Unlimited data generation
- Transformation is deterministic and fast
- Naturally generates diverse examples

## Results

| Dataset | FWD-trained | BWD-trained | Combined |
|---------|-------------|-------------|----------|
| FWD test | 91.5% | 27.5% | 56.1% |
| BWD test | 31.6% | 99.6% | 97.2% |
| IBP test | 60.0% | 60.0% | 89.1% |

BWD achieves near-perfect accuracy on BWD test data, demonstrating the
effectiveness of the approach when test distribution matches training.

## Application to PixelFlow NNUE

### Our Adaptation

For NNUE kernel optimization, we apply BWD as follows:

**Forward (current approach):**
1. Generate random expression tree
2. Benchmark to get cost
3. Train to predict cost from structure

**Backward (new approach):**
1. Generate "optimized" expression (using fused ops: MulAdd, MulRsqrt)
2. Apply reverse rewrites (unfusing) to create unoptimized version
3. Train on (unoptimized_features, optimization_gain)

### The Correspondence

| Lample & Charton | PixelFlow NNUE |
|------------------|----------------|
| Random function f | Optimized expression (fused ops) |
| Derivative f' | Unoptimized expression (unfused) |
| Integration task | Optimization task |
| (f', f) pairs | (unoptimized, optimized) pairs |

### Why This Works

1. **Generating optimized expressions is easy** - just include fused ops
2. **Unfusing is deterministic** - apply inverse rewrite rules
3. **Training signal is clear** - cost difference is the optimization gain
4. **Covers optimization space** - generates exactly the patterns we want to recognize

### Implementation

```rust
// Generate optimized expression (with fused ops)
let optimized = generator.generate_with_fused_ops();

// Apply reverse rewrites to "de-optimize"
let unoptimized = apply_unfusing_rewrites(&optimized);

// Training pair: (unoptimized_features, cost_delta)
// Model learns: "this unoptimized pattern can be optimized to save X cycles"
```

## References

```bibtex
@inproceedings{lample2019deep,
  title={Deep Learning for Symbolic Mathematics},
  author={Lample, Guillaume and Charton, Fran{\c{c}}ois},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
