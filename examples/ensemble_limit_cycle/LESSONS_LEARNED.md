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
- **PCA warm-start on (C, b) per dataset, from data alone** (no
  oracle). PFA / PLDS / GPFA-style: Gaussian-smooth spikes in time,
  log-transform, PCA across neurons, least-squares fit (C, b). At
  small scale this matches the oracle warm-start within seed noise
  (R^2 = 0.94, |omega|-rho = 0.98 vs oracle's 0.997). After fitting,
  rescale (C, b) so the latent has unit variance per dim, matching
  the dynamics MLP's working scale. References: Macke et al. NeurIPS
  2011 (PLDS init); Yu et al. J Neurophysiol 2009 (GPFA, the
  PCA-on-smoothed-rates subroutine).

### Caveat - sign-flip ambiguity

PCA principal components are sign-ambiguous: PC1 and -PC1 are both
valid. After warm-start, each dataset's latent may be sign-flipped
along any subset of latent dimensions, with no way to disambiguate
from data alone (each per-dataset latent space is its own).

For the limit cycle this is harmless: omega <-> -omega gives the
same orbit (rotational symmetry), so a per-dataset sign flip just
maps to a per-dataset omega-sign flip in the embedding, which is
already a rotation-invariance of the architecture (see the Gaussian
sweep that recovered |omega| at rho = 0.99 but not signed omega).

For general dynamics (asymmetric oscillators, fixed-point systems,
non-symmetric flows) the sign-flip is a discontinuous nuisance. The
shared dynamics MLP would have to learn the right and the sign-flipped
versions simultaneously, and gradient descent cannot bridge the
discontinuity. **The PCA warm-start trick is therefore not a general
fix; it is leveraging the limit-cycle symmetry here.** For non-
symmetric problems, a sign-disambiguation step would be needed
(e.g., align the sign of the second moment of dz/dt, or use a
manifold-preserving alignment to a canonical dataset's PCs).

### One-axis ablation, no combinations

Phase 1 tested each knob (`--warm-start-C oracle`, `--warm-start-C pca`,
`--readout-weight-decay 1e-2`, `--readout-lr-scale 0.1`) versus a
no-intervention baseline. Weight decay alone and lr scale alone do not
help. The Phase 2 cross of {warm-start} x {wd, lr-scale, both} was
proposed in STRATEGY.md but skipped because PCA warm-start alone
already hit the success criterion.

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
