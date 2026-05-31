# Lessons learned - Poisson Meta-SSM ensemble experiments

Cumulative findings from the Gaussian-success-then-Poisson-collapse
sequence of experiments. Written 2026-05-31 after seven sweeps (one
Gaussian, six Poisson) on the limit-cycle ensemble.

## TL;DR

1. **The architecture works** under Gaussian observations and recovers
   the 1-D omega family at R^2 ~ 0.99 (rotation-invariant) across a
   wide range of D_obs and hyperparameters.
2. **Under log-linear Poisson observations the same architecture
   collapses to a "predict mean rate via readout bias" solution**
   regardless of trial length, neuron count, number of trials, or
   warm-starting the dynamics MLP.
3. **Two independent results pinpoint the readout pathway as the
   bottleneck**:
   - Freezing the per-dataset readout to the true `(C, b)` jumps
     R^2 from 0.08 to 0.96 and signed Spearman(PC1(mu_e), omega)
     from 0.06 to 1.00 in 30 min of local CPU.
   - Boosting firing rate (mean 0.05 -> 1.0) lifts R^2 from -0.05 to
     0.62 - large effect from a likelihood-side change.
4. **`D_obs` scale alone is not the cause.** Gaussian at the same
   D_obs (1800-2400) reaches R^2 = 0.996.
5. **The Poisson collapse is genuinely Poisson-specific**, driven by:
   - sparse low-rate counts (most bins are zero; per-bin information
     is weak)
   - `exp()` nonlinearity scaling `dNLL/dC` by the rate (~0.05),
     so the gradient on `C` is tiny in the low-rate regime
   - a "trivial mean-rate solution" basin where the readout bias
     `b` absorbs the loss without any latent-state involvement

## Sanity check on physical units

Originally interpreted `mean_rate=0.05` as "0.05 per bin" with an
implicit 10 ms bin (= 5 spikes/sec, biologically reasonable cortical
mean) and `max_rate=0.5` as 50 spikes/sec peak. **Trial length in the
first sweeps was set to T_eff = 33 (~ 0.33 s)** which is way too short
for any dynamics inference. T_eff = 400 (T=1200) was attempted later
(~4 s, reasonable) but the optimization still didn't escape.
Recommended setup: T_eff >= 500 (5 s @ 10 ms bins).

## What we tried and what we learned

| experiment | finding | hardware |
|------------|---------|----------|
| Gaussian sweep (D_obs 20-50, 30 dB)              | R^2 = 0.99 | L4 |
| Poisson SNR sweep (n=300-2400, T=100, rate 0.05) | collapse: R^2 ~ 0  | L4 |
| Long-trial Poisson (T=1200, 2400)                | collapse, same R^2 trajectory vs step | A100 |
| 10x trials + dynamics MLP warm-start             | no effect (dynamics warm-start overwritten by NLL gradient) | A100 |
| Fixed shared C (calibrated once)                 | marginal improvement, still collapsed | local |
| **Frozen-true readout (C, b)**                   | **R^2 = 0.96, signed rho = 1.00** | local |
| D1: Gaussian at large D_obs (1800-2400)          | works (R^2 = 0.996); D_obs scale is not the issue | A100 |
| D2: Poisson high rate (mean 1.0)                 | partial recovery R^2 = 0.62 | A100 |

## What does *not* work, things to avoid trying again

- **More trials alone.** 10x trials (32 -> 320) does not change R^2 or
  embedding range in the collapse regime.
- **Longer trials alone.** T=2400 vs T=400 ends at the same R^2 = -0.36
  vs -0.07 (slightly worse, identical trajectory shape per step).
- **Warm-starting MlpDynamics on true z trajectories.** The dynamics
  MLP learns the true dynamics (MSE -> 5e-3) but the warm-started
  weights are overwritten by the Poisson NLL gradient in the first
  few hundred main-training steps.
- **Sharing C across datasets (data-side, learnable).** Calibrating
  one (C, b) on dataset 0 and reusing it for all datasets gives the
  same R^2 as per-dataset random C.
- **Changing the PoissonLikelihood functional form.** It is now
  log-linear `exp(C z + b)` matching the generator (this was the
  right fix and was already applied early on).

## What does work or shows promise

- **Frozen-true readout** demonstrates the architecture works once
  the per-dataset readout is anchored.
- **High firing rate** (D2) gets ~half the way back. Confirms the
  low-rate gradient pathology is real.
- **Untried but promising** (next experiment): per-dataset readout
  warm-start from data alone (spike-triggered linear regression on
  PCA-projected smoothed spikes), plus weight decay and/or smaller
  learning rate on the readout.

## The biggest single insight

The neurofisherSNR formula bounds **state-SNR** (how well x_t can be
inferred from y_t per bin assuming C is known). It says nothing about
how identifiable C itself is. With ~10^4 readout parameters per
dataset and effectively ~10^3 independent observations per training
step, C is **under-determined per step** even at high state-SNR.
Number of timepoints needs to scale with the number of unknowns in C,
not just with state-SNR. This is why regularization on C is a
reasonable lever.

## Recommended defaults for future Poisson runs

(Until the readout-side interventions are landed.)

- `mean_rate >= 0.5` (5+ spikes/sec at 10 ms bins) for safer
  optimization.
- `T_eff >= 500` (5 s at 10 ms) — enough cycles for dynamics + enough
  observations for C identifiability.
- `num_trials >= 64` per dataset.
- For diagnostic runs: use `--freeze-readout-to-true` to verify the
  rest of the pipeline is working before debugging optimization.

## Pointers

- Reports: `POISSON_REPORT.md`, `POISSON_REPORT_2.md`, `POISSON_REPORT_3.md`
- Reproduce commands: `REPRODUCE.md`
- Frozen-true diagnostic: commit `abedb58`,
  `gcp_runs/local_frozen_readout/results/frozen_readout_summary.png`
- D1+D2 sweep: commit `9e74aba`,
  `gcp_runs/exp-20260531-111911-d1d2-a100/results/d1d2_summary.png`
