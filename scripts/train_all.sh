#!/bin/bash
# NNUE Training Pipeline
#
# This script runs the complete training pipeline for the PixelFlow NNUE:
# 1. Generate benchmark expressions
# 2. Run SIMD benchmarks
# 3. Collect Judge training data
# 4. Train Judge (Value Head)
# 5. Collect search trajectories
# 6. Train Guide (Search Head)
#
# Usage:
#   ./scripts/train_all.sh          # Full pipeline
#   ./scripts/train_all.sh --quick  # Quick test run

set -e  # Exit on error

# Configuration
QUICK_MODE=false
COUNT=100
EPOCHS_JUDGE=100
EPOCHS_GUIDE=50

# Parse arguments
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            COUNT=20
            EPOCHS_JUDGE=20
            EPOCHS_GUIDE=10
            ;;
        --count=*)
            COUNT="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--quick] [--count=N]"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "NNUE Training Pipeline"
echo "=================================="
echo "Mode: $([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Full")"
echo "Expressions: $COUNT"
echo "Judge epochs: $EPOCHS_JUDGE"
echo "Guide epochs: $EPOCHS_GUIDE"
echo ""

# Step 1: Generate benchmark expressions via e-graph saturation
# Uses all_training_rules() = math rules + fusion rules (FMA, rsqrt, mul_rsqrt)
echo "=== Step 1: Generate benchmark expressions ==="
cargo run -p pixelflow-pipeline --example gen_egraph_variants --release -- \
    --count "$COUNT" \
    --variants 5 \
    --targeted \
    --seed 42
echo ""

# Step 2: Run SIMD benchmarks
echo "=== Step 2: Run SIMD benchmarks ==="
if [ "$QUICK_MODE" = true ]; then
    # Quick mode: fewer iterations
    cargo bench -p pixelflow-pipeline --bench generated_kernels -- \
        --warm-up-time 1 \
        --measurement-time 2 \
        --sample-size 10
else
    cargo bench -p pixelflow-pipeline --bench generated_kernels
fi
echo ""

# Step 3: Collect Judge training data
echo "=== Step 3: Collect Judge training data ==="
cargo run -p pixelflow-pipeline --bin collect_judge_data --release
echo ""

# Step 4: Train Judge (Value Head)
echo "=== Step 4: Train Judge (Value Head) ==="
cargo run -p pixelflow-pipeline --bin train_judge --release --features training -- \
    --epochs "$EPOCHS_JUDGE" \
    --learning-rate 0.001
echo ""

# Step 5: Collect search trajectories
echo "=== Step 5: Collect search trajectories ==="
SEARCH_COUNT=$((COUNT / 2))
if [ "$QUICK_MODE" = true ]; then
    cargo run -p pixelflow-pipeline --bin collect_search_data --release -- \
        --count "$SEARCH_COUNT" \
        --short-expansions 20 \
        --long-expansions 50 \
        --judge-model pixelflow-pipeline/data/judge.bin
else
    cargo run -p pixelflow-pipeline --bin collect_search_data --release -- \
        --count "$SEARCH_COUNT" \
        --short-expansions 50 \
        --long-expansions 200 \
        --judge-model pixelflow-pipeline/data/judge.bin
fi
echo ""

# Step 6: Train Guide (Search Head)
echo "=== Step 6: Train Guide (Search Head) ==="
cargo run -p pixelflow-pipeline --bin train_guide --release --features training -- \
    --epochs "$EPOCHS_GUIDE" \
    --learning-rate 0.0001 \
    --judge-model pixelflow-pipeline/data/judge.bin
echo ""

# Summary
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo ""
echo "Models saved to:"
echo "  - pixelflow-pipeline/data/judge.bin (Value Head)"
echo "  - pixelflow-pipeline/data/guide.bin (Search Head)"
echo ""
echo "To use the trained models in the compiler:"
echo "  let nnue = DualHeadNnue::load(\"pixelflow-pipeline/data/guide.bin\");"
echo ""
