# Poisson observation experiment plan

## Goal

Demonstrate that the meta-dynamical SSM recovers the 1-D |omega| family
under a *Poisson spike-count* observation model with biologically realistic
firing rates, matching the Gaussian-case result of |omega|-Spearman ~= 0.99
shown in `gcp_runs/exp-20260528-200902-sweep/results/sweep_summary.png`.

Concrete success criterion: median |Spearman(|PC1(mu_e)|, |omega|)| >= 0.9
across the N=100 datasets at convergence, on at least one (alpha, dim_emb,
n_neurons) configuration.

## Generative model

For dataset d with latent trajectory z_d(t) in R^2 (limit cycle, angular
velocity omega_d), draw spike counts at each bin as

    lambda_i(t) = exp( C_d[i, :] @ z_d(t) + b_d[i] )
    y_i(t)      ~ Poisson(lambda_i(t))

per the log-linear Poisson model used in neurofisherSNR
(arXiv:2408.08752, EUSIPCO 2024). The latent z_d is normalized to zero
mean and unit variance per dimension before being passed to the
observation model.

Per-dataset firing-rate constraints (target across datasets, with
small jitter):
- mean rate per bin: 0.05
- max rate per bin:  0.5

Per-dataset variation: n_neurons drawn from [n_min, n_max] (default
[500, 1500]) so the readin adapter has to absorb dimensionality change.

## SNR definition (from neurofisherSNR)

Per dataset, with C in R^{n x 2}, b in R^{1 x n}, latents z (T x 2):

    Lambda(z)        = exp(z C^T + b)               # (T x n)
    I^pop(z)         = C^T diag(Lambda(z)) C        # (2 x 2)
    invFI            = mean_t trace(I^pop(z(t))^{-1})
    SNR_dB           = 10 * log10( E_t[||z(t)||^2] / invFI )

The Gaussian sweep ran at 30 dB. Per the neurofisherSNR README, the upper
bound for the suggested-SNR knob is

    SNR_dB_bound ~ 10 * log10( tgt_rate * n / d_latent * 2 * log(max_rate/tgt_rate) )

At tgt_rate=0.05, max_rate=0.5, d_latent=2:

    30 dB => n ~ 8700 neurons (per dataset) -- too costly with N=100
    25 dB => n ~ 2750
    20 dB => n ~  870
    15 dB => n ~  275

The plan is to scan a few SNRs around the tractable range and find the
elbow where |omega|-recovery hits the 0.9 target.

## Code changes (this branch)

- `examples/ensemble_limit_cycle/synthetic.py`: add
  `generate_ensemble_poisson(...)` that mirrors `generate_ensemble`,
  but:
  - normalizes the per-dataset latent z before applying the observation
    model;
  - draws C with random Gaussian entries, scales C and chooses b via a
    two-stage procedure (bias to hit `tgt_rate`, then scale rows of C
    to hit `tgt_snr`); this is a minimal reimplementation of
    neurofisherSNR's `gen_poisson_observations` to avoid a network
    dependency in the experiment script and to run end-to-end on the GPU
    with torch tensors;
  - returns observations as `float` (PoissonLikelihood expects float input
    to its `log_prob`).
- `examples/ensemble_limit_cycle/builders.py`: add `build_model_poisson`
  wiring `PoissonLikelihood` instead of `GaussianLikelihood`.
- `examples/ensemble_limit_cycle/experiment.py`: add
  `--likelihood {gaussian,poisson}`, `--n-neurons-min`, `--n-neurons-max`,
  `--mean-rate`, `--max-rate`, `--target-snr-db`. Selects the right
  generator + builder.
- `examples/ensemble_limit_cycle/sweep_poisson.sh`: GPU-targeted sweep
  config.

## Compute plan (4h GPU budget)

1. Benchmark first: spin up smallest GPU VM (g2-standard-4 with L4),
   train one config for 50 steps to measure steps/sec on a representative
   (n_neurons, batch, datasets-per-step) point. Compare to CPU steps/sec
   from the e2-standard-8 sweep (~0.47 step/sec at d_obs<=50 Gaussian).
   Decide:
   - whether L4 is enough or to escalate to a single A100 (a2-highgpu-1g);
   - which configs fit in the remaining budget.
2. Sweep (best guess given Gaussian-sweep findings: alpha is largely
   irrelevant for recovery, but keep dim_emb=2 and alpha=0.01 as the
   anchor):

   | target_snr_dB | n_neurons range | configs            |
   |---------------|-----------------|--------------------|
   | 15            | 200-400         | alpha=0.01, dim=2  |
   | 20            | 500-1000        | alpha=0.01, dim=2  |
   | 25            | 1500-2500       | alpha=0.01, dim=2  |
   | 30 (stretch)  | 5000-10000      | alpha=0.01, dim=2  |

   Drop 30 dB if the benchmark shows it's too costly.

3. 2000 training steps per config (matches Gaussian sweep);
   snapshots every 200 steps; same N=100, num_trials=32,
   num_timepoints=100, stride=3.

## Reporting

Single 3-page report `examples/ensemble_limit_cycle/POISSON_REPORT.md`:
1. Setup + generative model + SNR definition (~1 page).
2. Results: cross-SNR `sweep_summary.png`-style figure with rectified
   embedding scatter, plus convergence curves (~1 page).
3. Comparison with Gaussian-30dB result + caveats (rotation invariance,
   bias-cancellation degeneracy from log-linear model) (~1 page).

## Open questions before execution

(Resolved in this iteration:)
- The bias `b_d` is a learned per-dataset parameter via the
  `nn.Linear(num_latents, num_observations)` inside `PoissonLikelihood`
  (it has both weight and bias). No extra plumbing needed.
- `PoissonLikelihood` is updated to use `rate = exp(C z + b)` (with a
  log-rate clamp at 8 for stability) so it matches the generator
  exactly. (Previously it used `softplus`, which mismatched the
  log-linear model in neurofisherSNR.)
