#!/usr/bin/env bash
# Phase 3: scale-up of the Phase 1/2 winner to the full N=100, T=1200
# configuration that collapsed without intervention. Single config (PCA
# warm-start alone, the simplest winner from Phase 1). If Phase 2
# reveals a meaningfully better combination, edit the COMMON args below
# before launching.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/experiment}"
PYTHON="${PYTHON:-python3}"
ENTRY="$REPO_ROOT/examples/ensemble_limit_cycle/experiment.py"
RESULTS_DIR="$REPO_ROOT/results_phase3"
mkdir -p "$RESULTS_DIR"

COMMON=(
  --likelihood poisson
  --device cuda
  --n 100
  --per-step 25
  --batch 16
  --dim-emb 2
  --alpha 0.01
  --mean-rate 0.05
  --max-rate 0.5
  --snr-db 15
  --n-neurons-min 800
  --n-neurons-max 1200
  --num-trials 64
  --num-timepoints 1200
  --steps 4000
  --eval-every 200
  --log-every 100
  --snapshot-every 500
  --warm-start-C pca
)

NAME="${PHASE3_NAME:-P3_pca_full_scale}"
EXTRA="${PHASE3_EXTRA:-}"

OUT="$RESULTS_DIR/$NAME"
LOG="$RESULTS_DIR/${NAME}.log"
SWEEP_LOG="$RESULTS_DIR/sweep.log"
: > "$SWEEP_LOG"
START=$(date +%s)
echo "[phase3] start $(date -Iseconds)" | tee -a "$SWEEP_LOG"
echo "[phase3] config=$NAME extra=$EXTRA" | tee -a "$SWEEP_LOG"
echo "$NAME" > "$RESULTS_DIR/current_config.txt"

"$PYTHON" -u "$ENTRY" "${COMMON[@]}" $EXTRA --out-dir "$OUT" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
END=$(date +%s)
echo "[phase3] exit=$RC dur=$((END-START))s" | tee -a "$SWEEP_LOG"
echo "PHASE3_DONE" | tee -a "$SWEEP_LOG"
