#!/usr/bin/env bash
# D1 + D2 diagnostic sweep on A100. Disentangles whether the Poisson
# collapse is caused by (a) the readout scale (large D_obs creating an
# imbalanced joint optimization) or (b) Poisson-specific signal
# properties (sparsity + exp() nonlinearity at low rate).
#
#   D1 = Gaussian at Poisson-scale D_obs.  If it collapses, the issue
#        is scale, not Poisson-vs-Gaussian.
#   D2 = Poisson at high firing rate (mean 1.0, max 5.0). If it works,
#        the issue is low-rate sparsity.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/experiment}"
PYTHON="${PYTHON:-python3}"
ENTRY="$REPO_ROOT/examples/ensemble_limit_cycle/experiment.py"
RESULTS_DIR="$REPO_ROOT/results"
mkdir -p "$RESULTS_DIR"

# Common across both: match the architecture and the trial structure of
# the L4 baseline that collapsed (N=100, T=100, 32 trials, 2000 steps,
# alpha=0.01, dim_emb=2). Only D_obs / likelihood / firing rate change.
COMMON=(
  --device cuda
  --n 100
  --per-step 50
  --batch 16
  --dim-emb 2
  --alpha 0.01
  --num-trials 32
  --num-timepoints 100
  --steps 2000
  --eval-every 100
  --log-every 50
  --snapshot-every 250
  --n-neurons-min 1800
  --n-neurons-max 2400
)

# (name, extra_args)
CONFIGS=(
  "D1_gaussian_large_Dobs  --likelihood gaussian --snr-db 30"
  "D2_poisson_high_rate    --likelihood poisson --snr-db 20 --mean-rate 1.0 --max-rate 5.0"
)

SWEEP_LOG="$RESULTS_DIR/sweep.log"
: > "$SWEEP_LOG"
START_ALL=$(date +%s)
echo "[sweep] start $(date -Iseconds)" | tee -a "$SWEEP_LOG"
nvidia-smi | head -10 | tee -a "$SWEEP_LOG"

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
