# Poisson experiment 3 - 10x trials + warm-start, A100

**Result in one line:** 10x trials and warm-starting `MlpDynamics` on
the ground-truth latent trajectories had **no effect on the collapse**.
Both the warm-start and the control (no warm-start) configs land at the
same R^2 = -0.054 and the same per-step loss trajectory, indistinguishable
beyond the third decimal at most steps.

Artifacts:
  `gcp_runs/exp-20260530-213415-warmstart-a100/results/warm_summary.png`
plus per-config plots, snapshots, and `warm_start.json` (the warm-start
training loss). GCP cost ~ $19 on A100 a2-highgpu-1g in
`europe-west4-a` for 4.66 h of sweep wall time.

## 1. Setup

Common (both configs):
- N = 100 datasets, Poisson likelihood, log-linear `lambda = exp(C z + b)`
  with learned bias (matches generator exactly).
- `num_trials = 320` (10x the previous 32).
- `num_timepoints = 400`, stride 3 -> `T_eff = 133`.
- `n_neurons in [800, 1200]`, `target_snr_db = 15`.
- `alpha = 0.01`, `dim_emb = 2`, `per_step = 25`, `batch = 16`.
- 4000 main training steps; snapshots every 500 steps.

The two configs differ only in:

| name             | warm-start steps | main steps |
|------------------|-----------------:|-----------:|
| `poisson_warm`   |            1000  |       4000 |
| `poisson_nowarm` |               0  |       4000 |

Warm-start (`warm_start_dynamics()` in `experiment.py`): per step, pick
a random dataset, sample 64 trials, compute MSE between
`MlpDynamics(z[:, :-1], deltas=None)` and `z[:, 1:]`. Only `MlpDynamics`
parameters are updated. Optimizer Adam, lr 1e-2. The warm-start loop
sees only ground-truth normalized latents (`data.latents[key] = z_norm`),
i.e. the model gets a near-oracle initialization of the *base* shared
dynamics before any Poisson gradient is applied.

## 2. Result

![warm summary](../../gcp_runs/exp-20260530-213415-warmstart-a100/results/warm_summary.png)

Final scalars:

| config         | step | loss      | rate R^2  | signed_rho | |omega|_rho | emb_range |
|----------------|-----:|----------:|----------:|-----------:|------------:|----------:|
| `poisson_warm`   | 4000 |  6.79e+05 |   -0.055 |     +0.028 |       0.003 |     3.79  |
| `poisson_nowarm` | 4000 |  6.79e+05 |   -0.054 |     -0.090 |       0.179 |     4.06  |

Panel (a-c) show the convergence trajectories overlay almost exactly
between the two configs - same loss curve, same R^2 climb, same flat
|omega|-Spearman. The control run's |omega|-Spearman ticks up to 0.18
in the last 500 steps, slightly *better* than warm-start; this is
within seed noise. The bottom-row embedding scatters look essentially
the same: spread out but not omega-ordered.

Panel (e) shows the warm-start did its job before main training began:
MSE on true z trajectories dropped to ~5e-3 within 50 steps and
stayed there. The base `MlpDynamics` weights at the end of warm-start
predict the next latent state from the current latent state with very
small error. That state was then overwritten by the Poisson NLL
gradient in main training, leaving the model in the same collapse
basin as the control.

## 3. Why warm-start did not help

Three (non-exclusive) explanations consistent with the data:

1. **Symmetric omega averaging.** Warm-start trains the *shared* base
   `MlpDynamics` without dataset conditioning. The omega range
   is `[-5, 5]`, symmetric, so the average rotation rate is zero. The
   warm-started MLP learns the radial pull toward the limit cycle
   (which is shared) and zero net rotation. Once main training starts,
   the rotation needs to come from the hypernet deltas - and those are
   driven by the embedding, which still collapses for the same reason
   as before.

2. **Gradient overwrite.** Poisson NLL on `B * T * n_neurons ~ 16 * 133
   * 1000 ~ 2e6` terms produces a gradient ~4-5 orders of magnitude
   larger than the MSE gradient that warm-start used. In the first
   few hundred main steps the dynamics MLP is pulled toward whatever
   minimizes the Poisson NLL given the current encoder/readout state -
   which is to flatten predictions, not to preserve the limit-cycle
   structure. The warm-start init is in the basin of attraction of
   the same flat solution.

3. **Encoder is the bottleneck, not the dynamics.** Even with a
   perfectly initialized MlpDynamics, training still requires the
   latent encoder to extract dataset-distinguishing z(t) from spike
   counts. The encoder is randomly initialized in both configs and
   collapses to a 1-D embedding regardless of where the dynamics MLP
   starts.

If (3) is right, the dynamics warm-start can never break the collapse
on its own; it would need to be combined with an encoder warm-start or
a different latent identifiability mechanism.

## 4. Cumulative picture from the three Poisson experiments

| run                     | hardware | n_neurons    | T    | trials | warm-start | R^2 final | |omega|_rho final |
|-------------------------|----------|--------------|-----:|-------:|-----------:|----------:|------------------:|
| L4 `poisson_snr20`      | L4       | [1800, 2400] |  100 |     32 |          0 |   +0.019  |              0.21 |
| A100 `poisson_T1200`    | A100     | [ 800, 1200] | 1200 |     32 |          0 |   -0.065  |              0.05 |
| A100 `poisson_T2400`    | A100     | [ 800, 1200] | 2400 |     32 |          0 |   -0.359  |              0.15 |
| A100 `poisson_warm`     | A100     | [ 800, 1200] |  400 |    320 |       1000 |   -0.055  |              0.00 |
| A100 `poisson_nowarm`   | A100     | [ 800, 1200] |  400 |    320 |          0 |   -0.054  |              0.18 |

Nothing in [n, T, num_trials, warm-start] has moved the metrics out of
the collapse band. The Gaussian sweep on the same family hit
R^2 = 0.998 and |omega|-Spearman = 0.99.

## 5. What is left to try (per user's earlier steers)

Ranked by expected effort vs payoff:

1. **Diversify initial conditions of trials** (`x0 ~ N(0, sigma^2 I)`
   with `sigma` swept in [0.3, 3]). Not yet tested. Would force trials
   to sample the *velocity field off-cycle*. This was Memming's
   recent framing about dynamics-SNR. Cheap to try (no architecture
   change). If this also fails, the encoder-bottleneck hypothesis
   (#3 above) becomes the leading explanation.
2. **Encoder warm-start.** Pre-train the latent encoder on per-dataset
   (`y`, `z_norm`) pairs with MSE loss before main training. This is
   the analogue of warm-starting `MlpDynamics` but for the part of
   the model that is most likely to be the bottleneck.
3. **Gaussian -> Poisson curriculum**. Train to convergence on the
   Gaussian likelihood (works), then swap to Poisson with a low lr
   to continue. The warm Gaussian model has a working encoder and
   dynamics; the swap reveals whether Poisson is genuinely
   incompatible or just hard to train from scratch.

## 6. Repeatability

- Code: branch `dev`, commit `6fefc0a`.
- Submodule: `external/neurofisherSNR` pinned at `96e83f6`.
- Seed: `20260528`. CLI in `examples/ensemble_limit_cycle/sweep_poisson.sh`.
- Artifacts (all local): full `state.pt`, `diagnostics.pt`,
  `metrics.json`, `warm_start.json`, 8 snapshots per config,
  per-config logs, top-level `sweep.log`.
- See `REPRODUCE.md` for exact commands.
