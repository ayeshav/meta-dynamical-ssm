#!/usr/bin/env bash
# Phase 1 of the small-scale-then-scale-up strategy (see STRATEGY.md).
# Five one-axis configs at small scale on A100 (~30 min total). The
# winners are promoted to Phase 2.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/experiment}"
PYTHON="${PYTHON:-python3}"
ENTRY="$REPO_ROOT/examples/ensemble_limit_cycle/experiment.py"
RESULTS_DIR="$REPO_ROOT/results"
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
)

# Each config touches exactly one knob vs the baseline.
# (name, extra args)
CONFIGS=(
  "P1_baseline    "
  "P1_oracle      --warm-start-C oracle"
  "P1_pca         --warm-start-C pca"
  "P1_wd          --readout-weight-decay 1e-2"
  "P1_lrscale     --readout-lr-scale 0.1"
)

SWEEP_LOG="$RESULTS_DIR/sweep.log"
: > "$SWEEP_LOG"
START_ALL=$(date +%s)
echo "[sweep] start $(date -Iseconds)" | tee -a "$SWEEP_LOG"

for entry in "${CONFIGS[@]}"; do
  read -r NAME EXTRA <<< "$entry"
  OUT="$RESULTS_DIR/$NAME"
  LOG="$RESULTS_DIR/${NAME}.log"
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== [sweep] $NAME ($EXTRA) $(date -Iseconds) =====" | tee -a "$SWEEP_LOG"
  echo "$NAME" > "$RESULTS_DIR/current_config.txt"
  T0=$(date +%s)
  "$PYTHON" -u "$ENTRY" "${COMMON[@]}" $EXTRA --out-dir "$OUT" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  echo "[sweep] $NAME exit=$RC dur=$((T1-T0))s" | tee -a "$SWEEP_LOG"
done

END_ALL=$(date +%s)
echo "[sweep] total dur=$((END_ALL-START_ALL))s" | tee -a "$SWEEP_LOG"
echo "SWEEP_DONE" | tee -a "$SWEEP_LOG"
