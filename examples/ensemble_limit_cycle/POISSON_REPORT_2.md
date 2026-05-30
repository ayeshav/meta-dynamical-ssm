# Poisson experiment 2 - longer trials, A100

**Result in one line:** at the requested settings
(`n in [800, 1200]`, mean rate 0.05, peak ~0.5, log-linear Poisson),
**increasing trial length 12-24x did not break the collapse.** Both
A100 runs land at the same R^2/|omega|-Spearman as the L4 baseline, on
the same trajectory shape. R^2 of rate recovery stays <= 0,
|Spearman(|PC1(mu_e)|, |omega|)| stays <= 0.15 throughout.

Artifacts:
  `gcp_runs/exp-20260529-212015-poisson-a100/results/comparison_summary.png`
plus per-config plots and snapshots in `poisson_T1200/`, `poisson_T2400/`.
GCP cost ~ $60 on A100 a2-highgpu-1g in `europe-west4-a`, 15.5 h of
sweep wall time.

## 1. Setup

- Hardware: NVIDIA A100 40 GB, a2-highgpu-1g, `europe-west4-a`.
- Latents, dynamics, normalization, calibration: all unchanged from
  the L4 baseline (see POISSON_REPORT.md and POISSON_PLAN.md).
- Architecture: same model (`alpha = 0.01`, `dim_emb = 2`),
  PoissonLikelihood already updated to log-linear `exp(C z + b)` with
  learned per-neuron bias.
- Optimizer: AdamW, lr = 1e-3 (unchanged).
- Sweep:

  | name             | n_neurons   | T    | T_eff | steps |
  |------------------|-------------|-----:|------:|------:|
  | `poisson_T1200`  | [800, 1200] | 1200 |   400 |  4000 |
  | `poisson_T2400`  | [800, 1200] | 2400 |   800 |  3000 |

  Both at `target_snr_db = 15` (realized 14.8 dB), `per_step = 25`.

## 2. Pre-flight bug audit

Before launching the long-running A100 sweep, I re-checked the Poisson
pipeline:

- `PoissonLikelihood.forward`: `rate = exp(clamp(C z + b, max=8))` -
  matches the generator's `lambda = exp(C z_norm + b)` exactly. Bias
  is learned via the `nn.Linear` readout.
- `encode_dataset()` (for diagnostic R^2): returns
  `exp(lik.readout(mu_q).clamp(max=lik.log_rate_clamp))` for Poisson.
  R^2 compares predicted rate to ground-truth rate.
- `_clean_target()`: returns `data.rates[key]` (the actual generator-
  computed rate), not spike counts.
- Generator `latents[key] = z_norm` is consistent with the rates
  (`exp(z_norm C^T + b)`), so the model's R^2 target is the true rate.
- No leftover Gaussian-only assumptions in `MetaDynamicalSSM`,
  `_run_one_dataset`, or the variational helpers.

Two non-bug observations: `n_samples = 1` in posterior sampling (could
add Monte Carlo if useful later); and `latents` storage convention
differs between Gaussian (raw `x`) and Poisson (normalized `z_norm`)
generators (purely cosmetic for the visualization).

**Conclusion: no bugs.** The collapse is real.

## 3. Result

![comparison](../../gcp_runs/exp-20260529-212015-poisson-a100/results/comparison_summary.png)

Final scalars (this run + the L4 baseline for comparison):

| run                       | n_neurons   | T    | steps | loss_final |  R^2  | |omega|_rho | emb_range |
|---------------------------|-------------|-----:|------:|-----------:|------:|------------:|----------:|
| L4 baseline `poisson_snr20` | [1800,2400] |  100 |  2000 |    7.26e+5 | +0.019 |        0.21 |      3.81 |
| A100 `poisson_T1200`        | [ 800,1200] | 1200 |  4000 |    2.02e+6 | -0.065 |        0.05 |      4.91 |
| A100 `poisson_T2400`        | [ 800,1200] | 2400 |  3000 |    4.30e+6 | -0.359 |        0.15 |      3.95 |

Per-step trajectories (panel a-c above): the three runs land on the
same shape of loss curve; R^2 climbs from very negative toward 0 from
below but never crosses; |omega|-Spearman stays in the [-0.15, +0.15]
noise band the entire training. The bottom row shows the final
embeddings: L4 has the original collapse + outliers; A100 T1200 has a
wide blob with no `omega` ordering; A100 T2400 has clean *modes*
(four clusters) but `omega` is not the labeling axis.

The A100 T1200 model is *worse* at fitting rate than the L4 baseline
at step 2000, even with double the training. Longer T (more spike data
per inference) did not let the encoder extract more dynamics signal -
the optimization is determined by the step count, not the data volume
per step.

## 4. Why longer trials did not help

Memming's framing made this precise after the run completed: the
neurofisherSNR formula bounds **state-SNR** (how well x_t can be
inferred from y_t per bin). It says nothing about **dynamics-SNR**
(how well omega can be inferred from the velocity field of x).
Dynamics-SNR depends on:

- the time dimension - more bins = more samples of x_t -> x_{t+1};
- the initial-condition distribution - whether trials sample the
  velocity field at many (z, dz/dt) points or repeatedly traverse the
  same orbit.

T1200/T2400 increased the time dimension 12-24x. But the dynamics-
inference problem is bottlenecked by what the encoder can extract per
step. The R^2 trajectories tracking the *same curve at the same step
count* across the three runs is direct evidence: each gradient step
moves the optimizer the same amount regardless of how much data per
trial. The encoder is not bandwidth-limited on a single trial; it is
optimization-limited.

## 5. Initial conditions are not stressed by the current generator

In `examples/ensemble_limit_cycle/synthetic.py::generate_ensemble_poisson`:

```python
x0 = torch.randn(num_trials, latent_dim, generator=generator, device=device)
```

`x0 ~ N(0, I)`, latent_dim = 2. Limit-cycle radius is
`sqrt(radius_scale=2) ~ 1.41`, so trials start inside the cycle and
spiral *out* over the first ~30-50 unstrided steps. After that all
trials are on-cycle and the only difference between them is phase.

So 32 trials per dataset effectively give 32 random phases of the same
cycle traversal. The *radial* axis of the dynamics (off-cycle
behavior, return rate to the cycle) is barely sampled. The 2-D
velocity field is observed only on a thin annulus around the limit
cycle.

This is a plausible explanation for why the encoder cannot tell omega
from a per-dataset constant: the trials don't probe the velocity field
in a way that disambiguates rotation rate from per-dataset random
features.

## 6. What to try next (per user steer)

Given longer trials did not help, the user's plan-B was to
**initialize the dynamics better**. Two concrete options for the next
experiment, in order of expected impact:

1. **Diversify initial conditions to probe the velocity field.**
   Sample `x0` from `N(0, sigma^2 I)` with `sigma in [0.3, 3]` so each
   trial spirals in/out from a different region. This forces the
   encoder to see (z, dz/dt) at off-cycle points where the dynamics
   curl is more informative about omega.
2. **Warm-start the shared `MlpDynamics` from a known limit-cycle
   solution** (or pre-train on the Gaussian likelihood and then swap
   in the Poisson). The hypothesis is that random init has the wrong
   topology and the optimizer cannot reorganize through the
   posterior-collapse basin once trapped.

(I am NOT recommending changing the likelihood, per user's earlier
correction.)

## 7. Honest limitations

- Single seed per config; the collapse pattern is identical across the
  three runs (L4, A100-T1200, A100-T2400), which makes a seed-effect
  unlikely, but a 2-3 seed sweep would close that loop.
- We never tried `n_samples > 1` in the posterior sampler. That is a
  cheap Monte Carlo lever for high-D Poisson likelihoods and worth a
  try if (1) and (2) above don't move the needle.
- "Bug-free" is a statement about the code paths I read; the model
  could still have an *architectural* mismatch with the Poisson regime
  (e.g., the GRU latent encoder's hidden state never sees the spike-
  count statistic that disambiguates omega). I cannot rule this out
  without an architecture-level intervention.
