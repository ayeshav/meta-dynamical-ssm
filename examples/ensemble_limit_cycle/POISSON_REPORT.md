# Poisson observation experiment - report

**Result in one line:** at the firing rates the user requested (mean
0.05, peak ~0.5), the meta-SSM **fails to recover** the 1-D
|omega| family under Poisson observations -- the model lands in a
posterior-collapse mode regardless of SNR (10, 15, or 20 dB). Rate
R^2 ~ 0 and |omega|-Spearman <= 0.2 against the target of 0.9. The
Gaussian-observation baseline on the same family hits R^2 = 0.998 and
|omega|-rho = 0.99 with the same model.

Artifacts: `gcp_runs/exp-20260529-171902-poisson-gpu/results/poisson_summary.png`
(single summary figure), per-config plots in `poisson_snr*/`, and
`poisson_summary.json` (cross-config scalar table).

## 1. Setup

- Ensemble: N = 100 limit-cycle datasets along the angular-velocity
  family omega in [-5, 5] (matching the Gaussian sweep at
  `gcp_runs/exp-20260528-200902-sweep/`).
- Observation model (log-linear Poisson, per
  neurofisherSNR, Jeon & Park, EUSIPCO 2024):

  ```
  lambda_i(t) = exp( C_d[i, :] @ z_norm(t) + b_d[i] )
  y_i(t)      ~ Poisson(lambda_i(t))
  ```

  z is normalized to zero mean / unit variance per dataset before
  being passed to the calibrator. C and b are fit by the lib's
  `gen_poisson_observations` with `tgt_rate_per_bin = 0.05`,
  `max_rate_per_bin = 0.5`, and `tgt_snr` set per config.
- Likelihood in the model: `meta_ssm.nn.PoissonLikelihood` updated to
  `rate = exp(W z + b)` (log-linear, matching the generator). Bias is
  the learned `nn.Linear` bias on the per-dataset readout. Log-rate
  clamp at 8 for stability.
- Sweep (single VM, NVIDIA L4 / g2-standard-4, europe-west2-b):

  | name            | target SNR | realized SNR | n_neurons      | steps |
  |-----------------|-----------:|-------------:|----------------|------:|
  | `poisson_snr20` | 20 dB      | 19.4 dB      | [1800, 2400]   | 2000  |
  | `poisson_snr15` | 15 dB      | 14.7 dB      | [ 800, 1200]   | 2000  |
  | `poisson_snr10` | 10 dB      |  9.6 dB      | [ 300,  500]   | 1000  |

  Common: alpha = 0.01, dim_emb = 2, 50 datasets per step, batch = 16,
  AdamW lr = 1e-3, seed = 20260528. Snapshots every 200 steps.
- Compute: 7337 s = 2.04 h of L4 wall time. **GCP cost ~ $2.30**
  (g2-standard-4 + 1 L4 + 100 GB pd-balanced).

## 2. Results

![sweep summary](../../gcp_runs/exp-20260529-171902-poisson-gpu/results/poisson_summary.png)

Final scalar metrics:

| config          | loss      | rate R^2 | |omega|-rho | emb_range |
|-----------------|----------:|---------:|------------:|----------:|
| `poisson_snr20` |   725 519 |   +0.019 |       0.21  |     3.81  |
| `poisson_snr15` |   333 869 |   -0.044 |       0.03  |     1.62  |
| `poisson_snr10` |   138 599 |   -3.65  |       0.16  |     5.04  |

Panel (a) shows loss collapses cleanly and plateaus by step 1200-1500
for all three configs - the optimizer is not stuck. Panel (b) shows
median R^2 of the inferred rate against the ground-truth rate barely
climbs out of negative territory by step 2000; the 0.9 target line is
far off-screen. Panel (c) shows |Spearman(|PC1(mu_e)|, |omega|)|
stays under 0.2 across the entire trajectory at every SNR.

Panel (d) makes the failure mode explicit: at the highest-SNR config
the per-dataset embeddings collapse to a single cluster (with two
single-dataset outliers). The lower-SNR configs' embedding scatters
show similar collapse (not shown to keep the figure to four panels;
files `poisson_snr*/embedding.png` are in the artifacts).

For comparison, the Gaussian sweep at 30 dB on the same family achieved
R^2 = 0.998 and |omega|-rho = 0.985-0.994 across alpha and dim_emb.

## 3. Diagnosis - posterior collapse driven by likelihood scale

Two readings of the diagnostic plots converge on the same explanation:

1. **`poisson_snr20/dynamics.png`** (per-config artifact, not in the
   summary figure for space): the inferred posterior-mean trajectory
   `mu_q` is a *line segment* in 2-D latent space at every omega, not
   a rotating circle. The encoder is sending all observations into a
   degenerate 1-D path.
2. **`poisson_snr20/embedding.png`**: 96 of the 100 dataset embeddings
   are stacked in one cluster with sub-unit radius; only 4 are off in
   isolated outliers. The hypernet has no informative input.

The cause is well-known for VAEs with very high-D observation
likelihoods: in our worst case the per-step NLL is summed over
`batch x T x n_neurons = 16 * 33 * 2400 ~ 1.3M` log-Poisson terms,
while the dynamics KL is summed over `16 x 33 x 2` and the embedding
KL is one scalar per dataset. The likelihood gradient signal
dominates the KL by 4-5 orders of magnitude. Once the readout's
per-neuron bias `b` is at `log(mean_rate)`, predicting a constant
mean rate is already a very low-loss solution, and there is little
pressure to use the latent state. The encoder responds by collapsing
the posterior.

This is consistent with the *loss-decreases-but-R^2-stays-near-zero*
pattern in panels (a) and (b): the model is learning, just learning
the wrong thing (the per-neuron, per-dataset mean rate).

## 4. What to try next

In rough order of expected impact (priorities for the next run):

1. **Bias-initialize the Poisson readout at `log(mean_rate)`** so the
   model does not need to use weights to absorb the per-neuron mean.
   This removes the easy collapse target and makes the latent-driven
   variation worth fitting.
2. **Free-bits on the dynamics KL** (clip KL_t per dim at a floor like
   0.5 nats) so the posterior cannot push KL to zero by collapsing.
3. **Curriculum / pre-train**. Train first on a Gaussian likelihood
   over the same loadings (or a square-root spike-count proxy), then
   swap in the Poisson likelihood. This gives the latent encoder a
   working representation to anchor.
4. **Reduce D_obs** -- repeat the n=300 (`poisson_snr10`) config at
   higher SNR (boost `tgt_snr` by adding more trials/timepoints
   instead of more neurons). The likelihood-vs-KL ratio shrinks.
5. **Smaller `lr` / longer schedule**. Current AdamW at 1e-3 is the
   Gaussian setting; Poisson with this loss magnitude may need 3-5x
   smaller or a warmup.

If item 1 alone closes the gap (we should know from a 30 min L4 run),
the rotation-invariant 1-D recovery picture established in the
Gaussian sweep should carry over directly.

## 5. Honest limitations

- Three SNR points only, single seed. The collapse pattern is the
  same at all three, which is informative, but a per-config
  seed-sweep would be needed before claiming "no Poisson recovery
  at this firing rate."
- The `poisson_snr20/diagnostics.pt` file was lost when the periodic
  rsync caught the run mid-checkpoint just before tear-down. The
  summary figure's panel (d) uses the embedding from the
  step-2000 snapshot (which did persist) rather than the
  `collect_final` trajectories. The conclusion is unchanged.
- Goal: 0.9 R^2 of |omega| recovery. **Not met.** This report
  documents the failure mode and the next experiment that should
  resolve it.
