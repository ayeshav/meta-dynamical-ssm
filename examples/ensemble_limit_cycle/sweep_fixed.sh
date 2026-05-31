#!/usr/bin/env bash
# Local-CPU 1h diagnostic: does fixing the observation matrix (one C, b
# shared across datasets, so the only per-dataset variation is omega)
# break the Poisson collapse? Two configs, ~11 min each on CPU.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-uv run --with torch --with numpy --with scipy --with matplotlib python}"
ENTRY="$REPO_ROOT/examples/ensemble_limit_cycle/experiment.py"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/gcp_runs/local_frozen_readout/results}"
mkdir -p "$RESULTS_DIR"

COMMON=(
  --likelihood poisson
  --device cpu
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
  --steps 2000
  --eval-every 100
  --log-every 50
  --snapshot-every 250
)

# (name, extra_args)
CONFIGS=(
  "poisson_frozen_true  --freeze-readout-to-true"
  "poisson_per_ds_C     "
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
  $PYTHON "$ENTRY" "${COMMON[@]}" $EXTRA --out-dir "$OUT" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  echo "[sweep] $NAME exit=$RC dur=$((T1-T0))s" | tee -a "$SWEEP_LOG"
done

END_ALL=$(date +%s)
echo "[sweep] total dur=$((END_ALL-START_ALL))s" | tee -a "$SWEEP_LOG"
echo "SWEEP_DONE" | tee -a "$SWEEP_LOG"
