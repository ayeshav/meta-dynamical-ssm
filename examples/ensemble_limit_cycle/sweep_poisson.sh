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
  --num-trials 320
  --num-timepoints 400
  --steps 4000
  --eval-every 200
  --log-every 100
  --snapshot-every 500
)

# Warm-start + 10x trials follow-up. After longer trials (T=1200/2400)
# didn't break the Poisson collapse, the next interventions are:
#   - 10x more trials (32 -> 320) for denser velocity-field sampling
#   - warm-start MlpDynamics on true latent trajectories for K steps
# Two configs, control vs warm-start, to isolate the warm-start effect.
# (name, warm_start_steps)
CONFIGS=(
  "poisson_warm   1000"
  "poisson_nowarm    0"
)

cd "$REPO_ROOT"
SWEEP_LOG="$RESULTS_DIR/sweep.log"
: > "$SWEEP_LOG"
START_ALL=$(date +%s)
echo "[sweep] start $(date -Is)" | tee -a "$SWEEP_LOG"
nvidia-smi | head -10 | tee -a "$SWEEP_LOG"

for entry in "${CONFIGS[@]}"; do
  read -r NAME WSTART <<< "$entry"
  OUT="$RESULTS_DIR/$NAME"
  LOG="$RESULTS_DIR/${NAME}.log"
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== [sweep] $NAME (warm_start=$WSTART) $(date -Is) =====" | tee -a "$SWEEP_LOG"
  echo "$NAME" > "$RESULTS_DIR/current_config.txt"
  T0=$(date +%s)
  "$PYTHON" -u "$ENTRY" "${COMMON[@]}" \
    --warm-start-steps "$WSTART" \
    --out-dir "$OUT" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  echo "[sweep] $NAME exit=$RC dur=$((T1-T0))s" | tee -a "$SWEEP_LOG"
done

END_ALL=$(date +%s)
echo "" | tee -a "$SWEEP_LOG"
echo "[sweep] total dur=$((END_ALL-START_ALL))s" | tee -a "$SWEEP_LOG"
echo "SWEEP_DONE" | tee -a "$SWEEP_LOG"
