# Reproducing the Poisson sweeps

All runs are seed-pinned at `seed=20260528` and run from this repo with
the `external/neurofisherSNR` submodule. Each run writes per-step
snapshots so the embedding/dynamics evolution can be re-analyzed without
re-training.

## L4 sweep (collapsed, 2026-05-29)

Artifacts: `gcp_runs/exp-20260529-171902-poisson-gpu/results/`.

Single VM (L4 / g2-standard-4) in `europe-west2-b`. Run the same sweep
from a fresh checkout:

```bash
git clone --recurse-submodules \
  --branch dev https://github.com/ayeshav/meta-dynamical-ssm.git
cd meta-dynamical-ssm
git checkout 9e4c863          # commit pinned for this sweep
bash examples/ensemble_limit_cycle/sweep_poisson.sh
```

(The L4-version `sweep_poisson.sh` had 3 SNR configs; that exact file is
the one at commit `9e4c863`. Check it out with
`git show 9e4c863:examples/ensemble_limit_cycle/sweep_poisson.sh`.)

## A100 long-trial sweep (this run, 2026-05-30)

Artifacts: `gcp_runs/exp-20260529-212015-poisson-a100/results/`.

Single VM (A100 / a2-highgpu-1g) in `europe-west4-a`. Two configs at
fixed `(n_neurons in [800, 1200], SNR target 15 dB, dim_emb=2,
alpha=0.01)`, varying trial length and step count:

```bash
git clone --recurse-submodules \
  --branch dev https://github.com/ayeshav/meta-dynamical-ssm.git
cd meta-dynamical-ssm
git checkout 6aeea7b           # commit pinned for this sweep
pip install scipy matplotlib   # numpy + torch preinstalled in image
bash examples/ensemble_limit_cycle/sweep_poisson.sh
```

Equivalent direct invocation (one config):

```bash
python3 examples/ensemble_limit_cycle/experiment.py \
  --likelihood poisson --device cuda \
  --n 100 --per-step 25 --batch 16 --dim-emb 2 --alpha 0.01 \
  --mean-rate 0.05 --max-rate 0.5 --snr-db 15 \
  --n-neurons-min 800 --n-neurons-max 1200 \
  --num-timepoints 1200 --steps 4000 \
  --eval-every 200 --log-every 100 --snapshot-every 500 \
  --out-dir results/poisson_T1200
```

## What each saved file is

Each `poisson_T*/` directory contains:

- `summary.json`: full CLI config + final scalar metrics + elapsed time.
- `metrics.json`: per-step loss + R^2 trajectory (sparser sampling).
- `state.pt`: final model `state_dict`, loadable via `torch.load(...,
  map_location='cpu')` then `build_model(...).load_state_dict(...)`.
- `diagnostics.pt`: final per-dataset `mu_e`, `mu_q`, `delta_norms`,
  `z_true`, `y_hat`, `rates`. Used for `embedding.png`, `dynamics.png`,
  `reconstruction.png`. Same shape across runs so post-hoc rescoring is
  one-line.
- `snapshots/step_*.pt`: `state_dict` + per-dataset `mu_e` +
  `delta_norms` + scalar diagnostics at every snapshot step. Use these
  to re-derive trajectory metrics or to warm-start a new run.

## Cost log

- L4 sweep: 2.04 h * ~$1.0/h = ~$2.30
- A100 sweep: TBD (filled in after VM is torn down)

## Post-hoc rescoring (no GPU needed)

To recompute the rotation-invariant |omega| Spearman across all
snapshots locally:

```bash
uv run --with torch python - <<'EOF'
import torch
from pathlib import Path
def spear(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float), torch.as_tensor(b, dtype=torch.float)
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra*rb).sum() / (torch.sqrt((ra**2).sum()*(rb**2).sum())+1e-12))
def pc1(mu_e):
    M = torch.as_tensor(mu_e, dtype=torch.float)
    if M.shape[1] == 1: return M[:, 0]
    Mc = M - M.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(Mc, full_matrices=False)
    return Mc @ v[0]
root = Path('gcp_runs/exp-20260529-212015-poisson-a100/results')
for cfg in sorted(root.iterdir()):
    snap_dir = cfg / 'snapshots'
    if not snap_dir.exists(): continue
    for sp in sorted(snap_dir.glob('step_*.pt')):
        try:
            s = torch.load(sp, weights_only=False, map_location='cpu')
        except Exception:
            continue
        om = torch.as_tensor(s['summary']['omegas'])
        p1 = pc1(s['summary']['mu_e'])
        print(f"{cfg.name:>15} step={s['step']:>5} |omega|_rho={spear(p1.abs(), om.abs()):+.3f} r2={s['diag']['r2_median']:+.3f}")
EOF
```
