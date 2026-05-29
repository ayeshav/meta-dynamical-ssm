#!/usr/bin/env bash
# Poisson SNR sweep on GPU. Three configs at varying SNR / neuron counts,
# fitting in a ~4h L4 budget. alpha and dim_embedding fixed at the
# Gaussian-sweep optimum.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/experiment}"
PYTHON="${PYTHON:-python3}"
ENTRY="$REPO_ROOT/examples/ensemble_limit_cycle/experiment.py"
RESULTS_DIR="$REPO_ROOT/results"
mkdir -p "$RESULTS_DIR"

COMMON=(
  --likelihood poisson
  --device cuda
  --n 100
  --per-step 50
  --batch 16
  --dim-emb 2
  --alpha 0.01
  --mean-rate 0.05
  --max-rate 0.5
  --eval-every 100
  --log-every 50
  --snapshot-every 200
)

# (name, snr_db, n_min, n_max, steps)
CONFIGS=(
  "poisson_snr20 20 1800 2400 2000"
  "poisson_snr15 15  800 1200 2000"
  "poisson_snr10 10  300  500 1000"
)

cd "$REPO_ROOT"
SWEEP_LOG="$RESULTS_DIR/sweep.log"
: > "$SWEEP_LOG"
START_ALL=$(date +%s)
echo "[sweep] start $(date -Is)" | tee -a "$SWEEP_LOG"
nvidia-smi | head -10 | tee -a "$SWEEP_LOG"

for entry in "${CONFIGS[@]}"; do
  read -r NAME SNR NMIN NMAX STEPS <<< "$entry"
  OUT="$RESULTS_DIR/$NAME"
  LOG="$RESULTS_DIR/${NAME}.log"
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== [sweep] $NAME (snr=$SNR dB, n=[$NMIN,$NMAX], steps=$STEPS) $(date -Is) =====" | tee -a "$SWEEP_LOG"
  echo "$NAME" > "$RESULTS_DIR/current_config.txt"
  T0=$(date +%s)
  "$PYTHON" -u "$ENTRY" "${COMMON[@]}" \
    --snr-db "$SNR" \
    --n-neurons-min "$NMIN" --n-neurons-max "$NMAX" \
    --steps "$STEPS" \
    --out-dir "$OUT" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  echo "[sweep] $NAME exit=$RC dur=$((T1-T0))s" | tee -a "$SWEEP_LOG"
done

END_ALL=$(date +%s)
echo "" | tee -a "$SWEEP_LOG"
echo "[sweep] total dur=$((END_ALL-START_ALL))s" | tee -a "$SWEEP_LOG"
echo "SWEEP_DONE" | tee -a "$SWEEP_LOG"
