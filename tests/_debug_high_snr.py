"""Standalone debug runner for the high-SNR ensemble.

Not a pytest. Prints R^2 / Spearman / loss every K steps so we can find
the budget and thresholds at which the model recovers structure.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.builders import build_model
from tests.synthetic import generate_ensemble, sample_batch


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (torch.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def predict_clean(model, ds: str, y_obs: torch.Tensor) -> torch.Tensor:
    y_bar = model.adapters.readin[ds](y_obs)
    y_bar = model.shared_readin(y_bar)
    mu_e, _ = model.embedding_encoder(y_bar)
    e = mu_e if model.concat_embedding else None
    mu_q, _ = model.latent_encoder(y_bar, e)
    z = model.shared_readout(mu_q)
    return model.adapters.likelihood[ds].readout(z)


@torch.no_grad()
def diagnostics(model, data) -> dict:
    model.eval()
    keys = list(data.observations.keys())

    embeddings = []
    omegas = []
    r2_values = []
    for key in keys:
        y_obs = data.observations[key]
        x_true = data.latents[key]
        C = data.observation_matrices[key]
        y_clean = x_true @ C

        y_bar = model.adapters.readin[key](y_obs)
        y_bar = model.shared_readin(y_bar)
        mu_e, _ = model.embedding_encoder(y_bar)
        embeddings.append(mu_e.flatten()[0].item())
        omegas.append(data.omegas[key])

        y_hat = predict_clean(model, key, y_obs)
        r2_values.append(r2(y_hat, y_clean))

    model.train()
    r2_sorted = sorted(r2_values)
    return {
        "spearman": spearman(torch.tensor(embeddings), torch.tensor(omegas)),
        "r2_median": r2_sorted[len(r2_sorted) // 2],
        "r2_min": r2_sorted[0],
        "r2_max": r2_sorted[-1],
        "emb_range": max(embeddings) - min(embeddings),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="N_ensemble")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--per-step", type=int, default=0, help="0 = all")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dim-emb", type=int, default=1)
    parser.add_argument("--snr-db", type=float, default=30.0)
    parser.add_argument("--p-mask", type=float, default=0.2)
    parser.add_argument("--num-trials", type=int, default=32)
    parser.add_argument("--num-timepoints", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"args: {vars(args)}")
    t0 = time.time()
    data = generate_ensemble(
        num_ensemble=args.n,
        num_trials=args.num_trials,
        num_timepoints=args.num_timepoints,
        snr_db=args.snr_db,
        seed=args.seed,
    )
    print(f"data: {len(data.observations)} datasets, obs_dims={list(data.observation_dims.values())}")
    print(f"omegas: min={min(data.omegas.values()):.2f}, max={max(data.omegas.values()):.2f}")

    model = build_model(data.observation_dims, dim_embedding=args.dim_emb, alpha=args.alpha)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batch_gen = torch.Generator(device="cpu")
    batch_gen.manual_seed(args.seed + 1)

    keys = list(data.observations.keys())
    per_step = args.per_step if args.per_step > 0 else len(keys)
    rng = random.Random(args.seed + 2)

    print(f"\n{'step':>5} {'loss':>10} {'rho':>7} {'r2_med':>8} {'r2_min':>8} {'r2_max':>8} {'emb_rng':>8}")
    diag0 = diagnostics(model, data)
    print(f"{'init':>5} {' ':>10} {diag0['spearman']:>7.3f} {diag0['r2_median']:>8.3f} "
          f"{diag0['r2_min']:>8.3f} {diag0['r2_max']:>8.3f} {diag0['emb_range']:>8.3f}")

    for step in range(1, args.steps + 1):
        subset = rng.sample(keys, per_step) if per_step < len(keys) else keys
        obs_subset = {k: data.observations[k] for k in subset}
        batch = sample_batch(obs_subset, args.batch, generator=batch_gen)

        optimizer.zero_grad(set_to_none=True)
        out = model(batch, p_mask=args.p_mask)
        loss = out["loss"].mean()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            diag = diagnostics(model, data)
            print(f"{step:>5} {float(loss):>10.2f} {diag['spearman']:>7.3f} {diag['r2_median']:>8.3f} "
                  f"{diag['r2_min']:>8.3f} {diag['r2_max']:>8.3f} {diag['emb_range']:>8.3f}")

    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
