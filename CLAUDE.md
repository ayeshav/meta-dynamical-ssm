# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PyTorch implementation of a meta-learning state-space model that shares
dynamics across multiple datasets with different observation dimensionalities.
Each dataset gets its own readin/likelihood "adapter" plus a per-dataset
embedding `e`; a hypernetwork turns `e` into low-rank (LoRA-style) deltas that
adapt a shared MLP transition function. Training optimizes an ELBO with KL
terms over latents, initial condition, and the embedding prior.

There is no `pyproject.toml`, `setup.py`, or lockfile — the package is used in
place via `sys.path` insertion in `examples/limit_cycle/train.py`. Dependencies
are `torch`, `hydra-core`, `omegaconf`. Per global rules, run Python through
`uv` (`uv run ...`).

## Running the example

```
uv run python examples/limit_cycle/train.py
```

The example is a Hydra app. Config root is `examples/limit_cycle/configs/`
(`config.yaml` + `model/meta_ssm.yaml`). Outputs land in
`outputs/<date>/<time>/` (Hydra `chdir: true`). Override config from the CLI,
e.g. `uv run python examples/limit_cycle/train.py training.steps=100 device=cuda`.

## Testing

```
uv run pytest tests/ -v
```

`tests/synthetic.py` is the ensemble data generator with calibrated SNR
(N_ensemble datasets along a 1-D angular-velocity family, varying observation
dims, per-dataset Gaussian noise sized to a target SNR in dB).
`tests/builders.py` constructs `MetaDynamicalSSM` directly (no Hydra),
mirroring `examples/limit_cycle/configs/model/meta_ssm.yaml`.
`tests/test_high_snr.py` runs a short training loop on a high-SNR ensemble
and checks (a) loss is finite and decreases, (b) median reconstruction R^2
on held-out datasets, (c) Spearman correlation of the inferred 1-D embedding
with the true angular velocity.

## Reporting

Post a 1-3 line TL;DR to Slack `#context-dependent-dynamics` after each
meaningful plan or milestone. Use `slack_send_message`. Format: what
changed / what's next.

## External dependencies

`external/neurofisherSNR` (git submodule, catniplab/neurofisherSNR) is
used for SNR-matched Poisson observation generation. The library is
numpy-based; the wrapper in `examples/ensemble_limit_cycle/synthetic.py`
moves between torch and numpy at the boundary. Two performance traps to
remember when calling its `gen_poisson_observations`:

1. **Pre-seed C.** Pass `C=...` instead of leaving it `None`, otherwise
   the lib's `initialize_C` runs a 15000-iteration coherence
   optimization that never converges for `d_latent=2` (max coherence of
   unit vectors on a 1-D circle is 1).
2. **Subsample latents.** `SNR_bound_instantaneous` iterates
   `np.linalg.inv` over every row of `x`; pass ~60 representative rows,
   not the full (B * T) flatten.

With these two fixes, generation is sub-second per dataset even at
`n_neurons = 2500`.

## Compute / GCP notes

- L4 GPU images: use `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` from
  `deeplearning-platform-release`. The older `common-cu123-debian-12`
  family is gone. Image needs >= 100 GB boot disk.
- GPU stockouts in `europe-west1-b`/`-c`/`europe-west4` are common; the
  cheapest reliable fallback we found for L4 was `europe-west2-b`.
- For this model size (~1-3 M params, ensemble of 100 small datasets
  serialized through the dataset loop in
  `MetaDynamicalSSM.forward`), L4 gives only ~2x speedup vs the same
  VM's 4 vCPUs. The bottleneck is the serial per-dataset loop, not the
  per-dataset matmul. If we ever need a real GPU win, batching across
  datasets via padding would be the lever.
- PyTorch on GCE Deep Learning images is preinstalled at
  `/usr/bin/python3`; `pip install scipy matplotlib` is needed for
  neurofisherSNR + plotting. `numpy` is preinstalled.

## Architecture

The whole forward pass lives in `meta_ssm/model.py::MetaDynamicalSSM._run_one_dataset`.
Reading that method top-to-bottom is the fastest way to understand the model.
Steps (numbered to match the source):

1. **Per-dataset readin** (`Adapters.readin[ds]`) maps `y` from its
   dataset-specific observation dim to `dim_y_bar`, then `shared_readin` projects
   to `dim_shared` with a `LayerNorm`. This is what makes datasets with
   different observation dimensionalities trainable jointly.
2. **Embedding encoder** (GRU over `y_bar`) produces a per-dataset latent
   `e ~ q(e|y)` of size `dim_embedding`. Note `EmbeddingEncoder` pools across
   the batch (`mean(0)`) — the embedding is a property of the dataset, not the
   trial.
3. **Hypernetwork** (`LoRAHypernet`) maps `e` to a dict of low-rank weight
   deltas `{linear_i: U V^T}` for the layers listed in `adapt_layers`, plus an
   `init_head` for `(mu_0, var_0)` when `common_init_condition=False`.
4. **Latent encoder** (`LatentDynamicsEncoderDKF` or `DVBF`) produces
   `q(z_t|y)`. DKF uses a bidirectional GRU; DVBF uses a reversed-time GRU.
   When `concat_embedding=True`, the embedding `e` is concatenated to the
   encoder input (detached in DKF, attached in DVBF).
5. **Masked posterior sampling** (`utils.variational.masked_posterior_sampler`):
   for each timestep, with probability `p_mask` the sample comes from the
   prior dynamics rather than the posterior. This is the "scheme" knob — DKF
   vs DVBF and various ablations are realized by combining encoder choice
   with the mask probability.
6. **Per-dataset likelihood** (`Adapters.likelihood[ds]`) — `Gaussian` or
   `Poisson` — closes the loop back to the original observation dim.
7. **Dynamics KL** uses `MlpDynamics(z[:-1], deltas)` as the prior. The
   `deltas` are applied as `base + z @ delta` (rank-r additive perturbation
   to the linear layer); `MlpDynamics.adapt_layers` must match the keys in
   the hypernet output.

The loss is `-ELBO + alpha * ||delta||`, summed across datasets in
`forward()`. Batches are `Dict[dataset_key, {"y": tensor, "u": optional}]` —
the model iterates datasets and sums losses, so a batch with N datasets does
N forward passes. There is no batching across datasets.

## Editing rules specific to this repo

- The dynamics adaptation is tightly coupled: `LoRAHypernet.adapt_layers`,
  `MlpDynamics.adapt_layers`, and the `linear_i` key naming in `deltas` must
  stay in sync. If you change one, change all three.
- `MlpDynamics.forward` does `z @ delta` where `delta` is `[dim_in, dim_out]`,
  built as `U V^T` via `einsum('bij, bkj -> bik')`. The `.squeeze()` on the
  result will drop the batch dim if `B == 1` — be careful when adding
  batch-size-1 code paths.
- `common_init_condition` has redundant code paths: the hypernet computes
  `(mu_0, var_0)` from `init_head`, but `_run_one_dataset` overwrites it with
  the shared `self.mu_0` / `self.log_variance_0` when the flag is true. The
  hypernet's `init_head` is still constructed regardless.
- Per global instructions: equations (the ELBO terms in `model.py`, KL in
  `utils/variational.py`, dynamics update rule) are equation code — ask before
  changing them, and update any describing document first.
