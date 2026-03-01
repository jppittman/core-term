#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Overnight training pipeline for NNUE Guide (actor-critic OPD)
#
# Stages:
#   1. Collect replay buffer data (Rust: perturbed-actor rollouts)
#   2. Train actor-critic (Python: GNN critic + NNUE actor distillation)
#   3. Export guide weights (Python -> Rust binary format)
#
# Usage:
#   cd core-term && bash pixelflow-pipeline/scripts/train_overnight.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/pixelflow-pipeline/data"
SCRIPTS_DIR="$PROJECT_ROOT/pixelflow-pipeline/scripts"
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/train_overnight_${TIMESTAMP}.log"

# Tee everything to log file AND stdout
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "  NNUE Guide Overnight Training Pipeline"
echo "  Started: $(date)"
echo "  Log: $LOG"
echo "================================================================"
echo ""

# ── Configuration ─────────────────────────────────────────────────────────
# Collection
COLLECT_COUNT=500
COLLECT_PERTURBATIONS=3
COLLECT_SIGMA=0.1
COLLECT_THRESHOLD=0.0
COLLECT_MAX_EPOCHS=30
COLLECT_SEED=42

# Training
TRAIN_EPOCHS=80
TRAIN_BATCH_SIZE=128
TRAIN_LR=3e-4
TRAIN_EMA_DECAY=0.999
TRAIN_HIDDEN=128
TRAIN_HEADS=4
TRAIN_LAYERS=4

# Output paths
REPLAY_BUFFER="$DATA_DIR/guide_training.jsonl"
TEACHER_CKPT="$DATA_DIR/graph_teacher_v2.pt"
STUDENT_BIN="$DATA_DIR/guide_v2.bin"

echo "Configuration:"
echo "  Collection: $COLLECT_COUNT expressions, $COLLECT_PERTURBATIONS perturbations"
echo "  Training: $TRAIN_EPOCHS epochs, batch=$TRAIN_BATCH_SIZE, lr=$TRAIN_LR"
echo "  Output: $STUDENT_BIN"
echo ""

# ── Stage 1: Collect Replay Buffer ────────────────────────────────────────
echo "================================================================"
echo "  Stage 1: Collecting replay buffer"
echo "  $(date)"
echo "================================================================"

# Fresh start — clear old data
: > "$REPLAY_BUFFER"

cd "$PROJECT_ROOT"

echo "Building collector..."
cargo build --release -p pixelflow-pipeline --bin collect_guide_data --features training 2>&1 | tail -3

echo ""
echo "Running collection..."
COLLECT_START=$(date +%s)

cargo run --release -p pixelflow-pipeline --bin collect_guide_data --features training -- \
    --count "$COLLECT_COUNT" \
    --perturbations "$COLLECT_PERTURBATIONS" \
    --sigma "$COLLECT_SIGMA" \
    --threshold "$COLLECT_THRESHOLD" \
    --max-epochs "$COLLECT_MAX_EPOCHS" \
    --seed "$COLLECT_SEED"

COLLECT_END=$(date +%s)
COLLECT_DURATION=$((COLLECT_END - COLLECT_START))
RECORD_COUNT=$(wc -l < "$REPLAY_BUFFER")

echo ""
echo "Collection complete: $RECORD_COUNT records in ${COLLECT_DURATION}s"
echo ""

if [ "$RECORD_COUNT" -lt 50 ]; then
    echo "ERROR: Too few records ($RECORD_COUNT). Something went wrong."
    exit 1
fi

# ── Stage 2: Train Actor-Critic ───────────────────────────────────────────
echo "================================================================"
echo "  Stage 2: Training actor-critic OPD"
echo "  $(date)"
echo "================================================================"

cd "$SCRIPTS_DIR"

TRAIN_START=$(date +%s)

uv run graph_teacher.py train-policy \
    --data "$REPLAY_BUFFER" \
    --teacher-checkpoint "$TEACHER_CKPT" \
    --student-checkpoint "$STUDENT_BIN" \
    --epochs "$TRAIN_EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --lr "$TRAIN_LR" \
    --ema-decay "$TRAIN_EMA_DECAY" \
    --hidden "$TRAIN_HIDDEN" \
    --heads "$TRAIN_HEADS" \
    --layers "$TRAIN_LAYERS"

TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

echo ""
echo "Training complete in ${TRAIN_DURATION}s"
echo ""

# ── Stage 3: Verify Export ────────────────────────────────────────────────
echo "================================================================"
echo "  Stage 3: Verify exported weights"
echo "  $(date)"
echo "================================================================"

if [ -f "$STUDENT_BIN" ]; then
    BYTES=$(wc -c < "$STUDENT_BIN")
    # 9 features: (9*32 + 32 + 32 + 1) * 4 = 1412 bytes
    EXPECTED=1412
    if [ "$BYTES" -eq "$EXPECTED" ]; then
        echo "Guide weights OK: $STUDENT_BIN ($BYTES bytes, $EXPECTED expected)"
    else
        echo "WARNING: Guide weights size mismatch: $BYTES bytes, expected $EXPECTED"
    fi
else
    echo "WARNING: Guide weights not found at $STUDENT_BIN"
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_START=$((COLLECT_END - COLLECT_DURATION))
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "================================================================"
echo "  Pipeline Complete"
echo "  Finished: $(date)"
echo "  Total time: ${TOTAL_DURATION}s"
echo "================================================================"
echo ""
echo "Artifacts:"
echo "  Replay buffer:  $REPLAY_BUFFER ($RECORD_COUNT records)"
echo "  Teacher ckpt:   $TEACHER_CKPT"
echo "  Guide weights:  $STUDENT_BIN"
echo "  Log:            $LOG"
