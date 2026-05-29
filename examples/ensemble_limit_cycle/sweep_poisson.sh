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
  --per-step 25
  --batch 16
  --dim-emb 2
  --alpha 0.01
  --mean-rate 0.05
  --max-rate 0.5
  --snr-db 15
  --n-neurons-min 800
  --n-neurons-max 1200
  --eval-every 200
  --log-every 100
  --snapshot-every 500
)

# Long-trial follow-up after the L4 collapse: at fixed (n ~ 1000, SNR 15
# dB), test whether more spike data per trial (longer T) gives the
# encoder enough signal to escape the mean-rate basin. Two configs:
#   poisson_T1200: 12x longer than the L4 sweep (T_eff = 400)
#   poisson_T2400: 24x longer (T_eff = 800)
# (name, T, steps)
CONFIGS=(
  "poisson_T1200 1200 4000"
  "poisson_T2400 2400 3000"
)

cd "$REPO_ROOT"
SWEEP_LOG="$RESULTS_DIR/sweep.log"
: > "$SWEEP_LOG"
START_ALL=$(date +%s)
echo "[sweep] start $(date -Is)" | tee -a "$SWEEP_LOG"
nvidia-smi | head -10 | tee -a "$SWEEP_LOG"

for entry in "${CONFIGS[@]}"; do
  read -r NAME TVAL STEPS <<< "$entry"
  OUT="$RESULTS_DIR/$NAME"
  LOG="$RESULTS_DIR/${NAME}.log"
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== [sweep] $NAME (T=$TVAL, steps=$STEPS) $(date -Is) =====" | tee -a "$SWEEP_LOG"
  echo "$NAME" > "$RESULTS_DIR/current_config.txt"
  T0=$(date +%s)
  "$PYTHON" -u "$ENTRY" "${COMMON[@]}" \
    --num-timepoints "$TVAL" \
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
