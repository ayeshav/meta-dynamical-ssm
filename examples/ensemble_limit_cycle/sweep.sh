#!/usr/bin/env bash
# Sequential hyperparameter sweep over (alpha, dim_embedding) for the
# meta-dynamical SSM. Designed for one long-running VM. Writes per-config
# logs and results, then a top-level SWEEP_DONE marker so we can poll.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/experiment}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
ENTRY="$REPO_ROOT/examples/ensemble_limit_cycle/experiment.py"
RESULTS_DIR="$REPO_ROOT/results"
mkdir -p "$RESULTS_DIR"

# Common args
COMMON=(
  --n 100
  --steps 2000
  --per-step 50
  --batch 16
  --snr-db 30
  --eval-every 100
  --log-every 50
  --snapshot-every 200
)

# (config_name, alpha, dim_emb)
CONFIGS=(
  "alpha0p00_dim2 0.0  2"
  "alpha0p01_dim2 0.01 2"
  "alpha0p10_dim2 0.1  2"
  "alpha0p00_dim1 0.0  1"
  "alpha0p00_dim4 0.0  4"
  "alpha0p01_dim1 0.01 1"
)

cd "$REPO_ROOT"
SWEEP_LOG="$RESULTS_DIR/sweep.log"
: > "$SWEEP_LOG"

START_ALL=$(date +%s)
echo "[sweep] start $(date -Is)" | tee -a "$SWEEP_LOG"
echo "[sweep] python=$PYTHON entry=$ENTRY" | tee -a "$SWEEP_LOG"

for entry in "${CONFIGS[@]}"; do
  read -r NAME ALPHA DIM <<< "$entry"
  OUT="$RESULTS_DIR/$NAME"
  LOG="$RESULTS_DIR/${NAME}.log"

  echo "" | tee -a "$SWEEP_LOG"
  echo "===== [sweep] $NAME (alpha=$ALPHA, dim_emb=$DIM) $(date -Is) =====" | tee -a "$SWEEP_LOG"
  echo "$NAME" > "$RESULTS_DIR/current_config.txt"

  T0=$(date +%s)
  "$PYTHON" -u "$ENTRY" "${COMMON[@]}" \
    --alpha "$ALPHA" --dim-emb "$DIM" \
    --out-dir "$OUT" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  echo "[sweep] $NAME exit=$RC dur=$((T1-T0))s" | tee -a "$SWEEP_LOG"
done

END_ALL=$(date +%s)
echo "" | tee -a "$SWEEP_LOG"
echo "[sweep] total dur=$((END_ALL-START_ALL))s" | tee -a "$SWEEP_LOG"
echo "SWEEP_DONE" | tee -a "$SWEEP_LOG"
