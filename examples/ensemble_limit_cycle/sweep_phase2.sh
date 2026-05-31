#!/usr/bin/env bash
# Phase 2: targeted cross of warm-start with the other knobs. Confirms
# whether weight decay or lower readout lr improve on PCA warm-start
# alone (Phase 1 winner). Same small scale, same 1500 steps.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/experiment}"
PYTHON="${PYTHON:-python3}"
ENTRY="$REPO_ROOT/examples/ensemble_limit_cycle/experiment.py"
RESULTS_DIR="$REPO_ROOT/results_phase2"
mkdir -p "$RESULTS_DIR"

COMMON=(
  --likelihood poisson
  --device cuda
  --n 30
  --per-step 15
  --batch 16
  --dim-emb 2
  --alpha 0.01
  --mean-rate 0.05
  --max-rate 0.5
  --snr-db 15
  --n-neurons-min 400
  --n-neurons-max 400
  --num-trials 64
  --num-timepoints 200
  --steps 1500
  --eval-every 100
  --log-every 50
  --snapshot-every 250
  --warm-start-C pca
)

# (name, extra)
CONFIGS=(
  "P2_pca_lr01     --readout-lr-scale 0.1"
  "P2_pca_wd       --readout-weight-decay 1e-2"
  "P2_pca_lr01_wd  --readout-lr-scale 0.1 --readout-weight-decay 1e-2"
)

SWEEP_LOG="$RESULTS_DIR/sweep.log"
: > "$SWEEP_LOG"
START_ALL=$(date +%s)
echo "[phase2] start $(date -Iseconds)" | tee -a "$SWEEP_LOG"

for entry in "${CONFIGS[@]}"; do
  read -r NAME EXTRA <<< "$entry"
  OUT="$RESULTS_DIR/$NAME"
  LOG="$RESULTS_DIR/${NAME}.log"
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== [phase2] $NAME ($EXTRA) $(date -Iseconds) =====" | tee -a "$SWEEP_LOG"
  echo "$NAME" > "$RESULTS_DIR/current_config.txt"
  T0=$(date +%s)
  "$PYTHON" -u "$ENTRY" "${COMMON[@]}" $EXTRA --out-dir "$OUT" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  echo "[phase2] $NAME exit=$RC dur=$((T1-T0))s" | tee -a "$SWEEP_LOG"
done

END_ALL=$(date +%s)
echo "[phase2] total dur=$((END_ALL-START_ALL))s" | tee -a "$SWEEP_LOG"
echo "PHASE2_DONE" | tee -a "$SWEEP_LOG"
