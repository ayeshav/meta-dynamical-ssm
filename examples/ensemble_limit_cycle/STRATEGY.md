# Strategy - small-scale hyperparameter search before scale-up

## Why this document exists

We have identified three independent levers that *should* help the
Poisson optimization escape the mean-rate-prediction basin:

1. **Warm-start `C`** per dataset from a cheap data-only estimate
   (spike-triggered linear regression on a smoothed-and-PCA'd version
   of the spike trains).
2. **Regularization on `C`** (L2 weight decay on the readout, since
   `C` is under-determined per training step at the current `D_obs`
   and trial count).
3. **Lower learning rate on `C` and `b`** than on the encoder, so the
   encoder can move first and `C` settles around the encoder's
   estimate rather than the other way around.

These are all standard VAE/posterior-collapse fixes. Each adds
hyperparameters. Naive grid over even a coarse 3 x 3 x 3 product is 27
configurations -> infeasible at our usual per-config GPU budget.

The strategy below uses fast small-scale runs to *rank* configurations
by convergence speed, then scales up only the top 2-3 to full size.

## Phase 0 - reuse the small-scale recipe we already have

Use the same scale as the fixed-C / freeze-readout local diagnostics:

- N = 30 datasets, T = 200, n_neurons = 400, num_trials = 64
- 1000-1500 main steps (collapse, if it happens, has happened by then
  in the small-scale baselines)
- alpha = 0.01, dim_emb = 2
- Local CPU; each run is ~5-12 min.

## Phase 1 - one-axis-at-a-time first

Don't sweep the full cross-product. First test each axis alone vs the
small-scale baseline, to see which knob is moving the needle.

| # | warm-start C | weight-decay on readout | readout lr scale | hypothesis tested |
|---|---|---:|---:|---|
| 0 | none           | 0    | 1.0  | small-scale baseline (re-run; R^2 ~ 0.08 expected) |
| 1 | oracle         | 0    | 1.0  | init from true (C, b), allow training. Tests whether a good init alone is enough. |
| 2 | sta (PCA+lstsq)| 0    | 1.0  | data-only init. Tests whether realistic warm-start works. |
| 3 | none           | 1e-2 | 1.0  | regularization alone |
| 4 | none           | 0    | 0.1  | slower readout lr alone |

5 configs * ~10 min each = ~50 min local CPU. **Pick any winners
(R^2 > 0.3 by step 1000 is the bar).**

## Phase 2 - small targeted cross of the winners

If, say, configs 2 and 4 are both promising:

| # | warm-start C | weight-decay | lr scale |
|---|---|---:|---:|
| 5 | sta            | 0    | 0.1  |
| 6 | sta            | 1e-2 | 1.0  |
| 7 | sta            | 1e-2 | 0.1  |

Pick the best from {5, 6, 7}. ~30 min local CPU.

## Phase 3 - scale-up

Take the single best config from Phase 1 + Phase 2 and scale to:

- N = 100, T = 1200 (5 s at 10 ms bins), n in [800, 1200],
  num_trials = 64, 4000 main steps
- A100 GPU, ~5-7 h
- This is the "real" experiment. Single config (or two: best + a
  close-second seed-control) to keep budget at ~1/2 day.

## Decision criteria

At each phase, the deciding metric is:

- **Primary**: R^2 of rate at step 1000 (Phase 1) / step 2000 (Phase 2)
- **Secondary**: emb_range > 1.0 (i.e., embedding pathway hasn't
  collapsed)
- **Tertiary**: |Spearman(|PC_1(mu_e)|, |omega|)| trajectory monotone-
  ish upward

If the primary metric is similar across configs, prefer the cheapest
intervention (fewer non-default knobs touched).

## Cost ceiling

Phase 1 + Phase 2: ~80 min local CPU, $0.
Phase 3: ~$25 GPU.

If nothing in Phase 1+2 crosses R^2 > 0.3 on the small scale, do NOT
escalate to Phase 3. Instead, report and replan.

## Stopping criteria for the whole effort

We declare success if a single config (without freezing C to oracle)
achieves R^2 >= 0.9 and |omega|-rho >= 0.7 in the Phase 3 scale-up.

We declare partial success if R^2 >= 0.5 and emb_range >= 1.0 — the
encoder is doing something useful and the rest can be tuned.

We declare failure and re-plan if Phase 3 lands at R^2 < 0.3.
