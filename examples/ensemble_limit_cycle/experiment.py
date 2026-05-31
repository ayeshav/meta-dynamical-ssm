"""Ensemble high-SNR experiment for MetaDynamicalSSM.

Trains the model on N_ensemble synthetic limit-cycle datasets spanned by a
1-D angular-velocity family at calibrated SNR, then writes metrics,
checkpoints, and diagnostic plots to a results directory.

Outputs (under --out-dir, default ./results):
  metrics.json          history of loss / R^2 / Spearman across training
  summary.json          final scalar metrics + config
  state.pt              final model state_dict
  diagnostics.pt        final per-dataset {mu_e, mu_q, deltas, omegas}
  convergence.png       loss + R^2 + Spearman vs step
  dynamics.png          true vs inferred latent trajectories, 6 datasets
  embedding.png         mu_e geometry colored by omega
  delta_norms.png       per-dataset hypernet delta magnitudes vs omega
  reconstruction.png    y_clean vs y_hat for sampled trials/channels
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.ensemble_limit_cycle.builders import build_model
from examples.ensemble_limit_cycle.synthetic import (
    generate_ensemble,
    generate_ensemble_poisson,
    sample_batch,
)


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
def encode_dataset(model, ds: str, y_obs: torch.Tensor):
    """Return (mu_e, mu_q, clean_prediction).

    `clean_prediction` is the mean of the likelihood (Gaussian: linear
    readout; Poisson: exp(log-rate)). Matches the noise-free target used
    in `diagnostics`.
    """
    from meta_ssm.nn import PoissonLikelihood
    y_bar = model.adapters.readin[ds](y_obs)
    y_bar = model.shared_readin(y_bar)
    mu_e, _ = model.embedding_encoder(y_bar)
    e = mu_e if model.concat_embedding else None
    mu_q, _ = model.latent_encoder(y_bar, e)
    z = model.shared_readout(mu_q)
    lik = model.adapters.likelihood[ds]
    raw = lik.readout(z)
    if isinstance(lik, PoissonLikelihood):
        y_hat = torch.exp(raw.clamp(max=lik.log_rate_clamp))
    else:
        y_hat = raw
    return mu_e, mu_q, y_hat


def _clean_target(data, key: str) -> torch.Tensor:
    """Ground-truth noise-free target for the given dataset.

    Poisson: true rate lambda = exp(C z + b). Gaussian: y_clean = z C.
    """
    if hasattr(data, "rates"):
        return data.rates[key]
    return data.latents[key] @ data.observation_matrices[key]


@torch.no_grad()
def diagnostics(model, data) -> dict:
    model.eval()
    keys = list(data.observations.keys())
    embeddings, omegas, r2_values = [], [], []
    for key in keys:
        y_obs = data.observations[key]
        y_clean = _clean_target(data, key)
        mu_e, _, y_hat = encode_dataset(model, key, y_obs)
        embeddings.append(mu_e.flatten().tolist())
        omegas.append(data.omegas[key])
        r2_values.append(r2(y_hat, y_clean))
    model.train()
    # for >1D embedding, Spearman against PC1 of mu_e
    emb_tensor = torch.tensor(embeddings)
    if emb_tensor.shape[1] == 1:
        emb_1d = emb_tensor[:, 0]
    else:
        emb_centered = emb_tensor - emb_tensor.mean(0, keepdim=True)
        _, _, v = torch.linalg.svd(emb_centered, full_matrices=False)
        emb_1d = emb_centered @ v[0]
    r2_sorted = sorted(r2_values)
    return {
        "spearman": spearman(emb_1d, torch.tensor(omegas)),
        "r2_median": r2_sorted[len(r2_sorted) // 2],
        "r2_min": r2_sorted[0],
        "r2_max": r2_sorted[-1],
        "emb_range": float(emb_1d.max() - emb_1d.min()),
    }


@torch.no_grad()
def collect_summary(model, data) -> dict:
    """Lightweight per-dataset state for snapshots (no trajectories)."""
    model.eval()
    keys = list(data.observations.keys())
    mu_e_all, delta_norms_all = [], []
    for key in keys:
        y_obs = data.observations[key]
        y_bar = model.adapters.readin[key](y_obs)
        y_bar = model.shared_readin(y_bar)
        mu_e, _ = model.embedding_encoder(y_bar)
        deltas, _, _ = model.hypernetwork(mu_e)
        mu_e_all.append(mu_e.flatten().tolist())
        delta_norms_all.append(
            [float(torch.linalg.norm(d).item()) for _, d in sorted(deltas.items())]
        )
    model.train()
    return {
        "keys": keys,
        "omegas": [data.omegas[k] for k in keys],
        "mu_e": mu_e_all,
        "delta_norms": delta_norms_all,
    }


@torch.no_grad()
def collect_final(model, data) -> dict:
    """Per-dataset arrays for plotting."""
    model.eval()
    keys = list(data.observations.keys())
    out = {
        "keys": keys,
        "omegas": [data.omegas[k] for k in keys],
        "obs_dims": [data.observation_dims[k] for k in keys],
        "mu_e": [],
        "mu_q": [],          # [num_trials, T, num_latents]
        "z_true": [],        # [num_trials, T, latent_dim]
        "y_hat": [],         # [num_trials, T, obs_dim]
        "y_clean": [],       # [num_trials, T, obs_dim]
        "delta_norms": [],   # list of per-layer norms
    }
    for key in keys:
        y_obs = data.observations[key]
        x_true = data.latents[key]
        y_clean = _clean_target(data, key)
        mu_e, mu_q, y_hat = encode_dataset(model, key, y_obs)
        deltas, _, _ = model.hypernetwork(mu_e)
        out["mu_e"].append(mu_e.flatten().tolist())
        out["mu_q"].append(mu_q.cpu())
        out["z_true"].append(x_true.cpu())
        out["y_hat"].append(y_hat.cpu())
        out["y_clean"].append(y_clean.cpu())
        out["delta_norms"].append(
            [float(torch.linalg.norm(d).item()) for _, d in sorted(deltas.items())]
        )
    model.train()
    return out


def warm_start_dynamics(model, data, num_steps: int, lr: float, batch: int,
                        rng: random.Random, log_every: int = 50) -> list[dict]:
    """Pre-train `model.dynamics` (no deltas) to fit one-step prediction on the
    ground-truth latent trajectories.

    Pulls z_norm from `data.latents` (already shape [num_trials, T, d_latent]
    per dataset), samples a random dataset + a random batch of trials per
    step, and minimizes MSE between `dynamics(z[:, :-1])` and `z[:, 1:]`.

    Only `model.dynamics` parameters are updated. The deltas pathway is not
    used (deltas=None), so this just gives the *base* MlpDynamics a good
    initialization in the limit-cycle basin.
    """
    opt = torch.optim.Adam(model.dynamics.parameters(), lr=lr)
    keys = list(data.observations.keys())
    history = []
    print(f"\n[warm-start] {num_steps} steps, lr={lr}, batch={batch}, "
          f"learning base MlpDynamics on true latents")
    for step in range(1, num_steps + 1):
        key = rng.choice(keys)
        z = data.latents[key]
        idx = torch.randint(z.shape[0], (min(batch, z.shape[0]),), device=z.device)
        z_batch = z[idx]                       # [b, T, d_latent]
        z_prev = z_batch[:, :-1]
        z_next = z_batch[:, 1:]

        opt.zero_grad(set_to_none=True)
        z_pred, _ = model.dynamics(z_prev, deltas=None)
        loss = ((z_pred - z_next) ** 2).mean()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == num_steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
            print(f"[warm-start] {step:>5} loss={loss.item():.4f}")
    return history


def make_plots(metrics, final, out_dir, dim_embedding):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

    # --- 1. Convergence ---
    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    eval_steps = [m["step"] for m in metrics if "r2_median" in m]
    r2_med = [m["r2_median"] for m in metrics if "r2_median" in m]
    r2_min = [m["r2_min"] for m in metrics if "r2_median" in m]
    r2_max = [m["r2_max"] for m in metrics if "r2_median" in m]
    rho = [abs(m["spearman"]) for m in metrics if "r2_median" in m]

    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    axes[0].plot(steps, losses, lw=0.8, color="black")
    axes[0].set_ylabel("loss")
    axes[1].fill_between(eval_steps, r2_min, r2_max, alpha=0.2, color="C0", lw=0)
    axes[1].plot(eval_steps, r2_med, color="C0", lw=1.2)
    axes[1].set_ylabel("R^2 (median; min-max band)")
    axes[1].set_ylim(-0.5, 1.05)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[2].plot(eval_steps, rho, color="C3", lw=1.2)
    axes[2].set_ylabel("|Spearman rho|")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].axhline(0.7, color="gray", lw=0.5, ls="--")
    axes[2].set_xlabel("step")
    fig.savefig(out_dir / "convergence.png")
    plt.close(fig)

    # --- 2. Snapshot dynamics ---
    omegas = final["omegas"]
    order = sorted(range(len(omegas)), key=lambda i: omegas[i])
    pick = [order[round(i * (len(order) - 1) / 5)] for i in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.5))
    for ax, idx in zip(axes.flat, pick):
        z_true = final["z_true"][idx]
        mu_q = final["mu_q"][idx]
        n_show = min(6, z_true.shape[0])
        for tr in range(n_show):
            ax.plot(z_true[tr, :, 0], z_true[tr, :, 1], color="gray", lw=0.7, alpha=0.7)
            ax.plot(mu_q[tr, :, 0], mu_q[tr, :, 1], color="C0", lw=0.7)
        ax.set_title(f"omega={omegas[idx]:+.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle("true (gray) vs inferred posterior mean (blue)", fontsize=10)
    fig.savefig(out_dir / "dynamics.png")
    plt.close(fig)

    # --- 3. Embedding geometry ---
    mu_e = torch.tensor(final["mu_e"])
    omegas_t = torch.tensor(omegas)
    fig, ax = plt.subplots(figsize=(5, 4))
    if mu_e.shape[1] == 1:
        ax.scatter(omegas_t, mu_e[:, 0], c=omegas_t, cmap="viridis", s=20)
        ax.set_xlabel("true omega")
        ax.set_ylabel("mu_e")
    else:
        sc = ax.scatter(mu_e[:, 0], mu_e[:, 1], c=omegas_t, cmap="viridis", s=20)
        ax.set_xlabel("mu_e[0]")
        ax.set_ylabel("mu_e[1]")
        plt.colorbar(sc, ax=ax, label="omega")
    fig.savefig(out_dir / "embedding.png")
    plt.close(fig)

    # --- 4. Delta norms vs omega ---
    delta_norms = torch.tensor(final["delta_norms"])  # [N, num_layers]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for j in range(delta_norms.shape[1]):
        ax.scatter(omegas_t, delta_norms[:, j], s=15, label=f"layer {j}")
    ax.set_xlabel("omega")
    ax.set_ylabel("||delta||")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "delta_norms.png")
    plt.close(fig)

    # --- 5. Reconstruction snapshot ---
    fig, axes = plt.subplots(3, 2, figsize=(7, 6), sharex=True)
    pick2 = [order[0], order[len(order) // 2], order[-1]]
    channels = [0, 1]
    for row, idx in enumerate(pick2):
        y_clean = final["y_clean"][idx][0]
        y_hat = final["y_hat"][idx][0]
        for col, ch in enumerate(channels):
            ch_idx = min(ch, y_clean.shape[1] - 1)
            ax = axes[row, col]
            ax.plot(y_clean[:, ch_idx], color="gray", lw=0.8)
            ax.plot(y_hat[:, ch_idx], color="C0", lw=0.8)
            if col == 0:
                ax.set_ylabel(f"omega={omegas[idx]:+.2f}", fontsize=9)
            if row == 0:
                ax.set_title(f"channel {ch_idx}", fontsize=9)
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    fig.suptitle("y_clean (gray) vs y_hat (blue), trial 0", fontsize=10)
    fig.savefig(out_dir / "reconstruction.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--per-step", type=int, default=0, help="0 = all")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dim-emb", type=int, default=2)
    parser.add_argument("--snr-db", type=float, default=30.0)
    parser.add_argument("--p-mask", type=float, default=0.2)
    parser.add_argument("--num-trials", type=int, default=32)
    parser.add_argument("--num-timepoints", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="Save state_dict + per-dataset mu_e + delta norms every K steps (0 = off)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--likelihood", choices=("gaussian", "poisson"),
                        default="gaussian")
    parser.add_argument("--n-neurons-min", type=int, default=500,
                        help="Poisson only: min neurons per dataset")
    parser.add_argument("--n-neurons-max", type=int, default=1500,
                        help="Poisson only: max neurons per dataset")
    parser.add_argument("--mean-rate", type=float, default=0.05,
                        help="Poisson only: target mean firing rate per bin")
    parser.add_argument("--max-rate", type=float, default=0.5,
                        help="Poisson only: target peak firing rate per bin")
    parser.add_argument("--device", default="cpu",
                        help="cpu, cuda, or cuda:N")
    parser.add_argument("--warm-start-steps", type=int, default=0,
                        help="Pre-train MlpDynamics on true (z_t -> z_{t+1}) "
                             "for K gradient steps before main training.")
    parser.add_argument("--warm-start-lr", type=float, default=1e-2)
    parser.add_argument("--warm-start-batch", type=int, default=64,
                        help="Trials per warm-start step (sampled per dataset).")
    parser.add_argument("--fixed-obs", action="store_true",
                        help="Poisson only: share C, b across datasets so the "
                             "only per-dataset difference is omega. Uses "
                             "n_neurons_min for all datasets.")
    parser.add_argument("--freeze-readout-to-true", action="store_true",
                        help="Poisson only: copy the true generator (C, b) "
                             "into each per-dataset PoissonLikelihood readout "
                             "and disable gradients on them. Forces the model "
                             "latent to match z_norm.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = vars(args).copy()
    config["out_dir"] = str(config["out_dir"])

    device = torch.device(args.device)

    t0 = time.time()
    if args.likelihood == "gaussian":
        data = generate_ensemble(
            num_ensemble=args.n,
            num_trials=args.num_trials,
            num_timepoints=args.num_timepoints,
            obs_dim_range=(args.n_neurons_min, args.n_neurons_max),
            snr_db=args.snr_db,
            seed=args.seed,
            device=device,
        )
        print(f"data: N={len(data.observations)} (gaussian, D_obs in "
              f"[{args.n_neurons_min}, {args.n_neurons_max}]), snr_db={data.snr_db}")
    else:
        data = generate_ensemble_poisson(
            num_ensemble=args.n,
            num_trials=args.num_trials,
            num_timepoints=args.num_timepoints,
            n_neurons_range=(args.n_neurons_min, args.n_neurons_max),
            target_mean_rate=args.mean_rate,
            target_max_rate=args.max_rate,
            target_snr_db=args.snr_db,
            seed=args.seed,
            device=device,
            fixed_obs=args.fixed_obs,
        )
        snrs = list(data.realized_snrs.values())  # type: ignore[attr-defined]
        print(f"data: N={len(data.observations)} (poisson), target_snr_db={data.snr_db}, "
              f"realized SNR mean={sum(snrs)/len(snrs):.2f} dB, "
              f"min={min(snrs):.2f}, max={max(snrs):.2f}")
    print(f"omegas: {min(data.omegas.values()):+.2f} .. {max(data.omegas.values()):+.2f}")
    print(f"obs_dims: {min(data.observation_dims.values())} .. {max(data.observation_dims.values())}")

    model = build_model(
        data.observation_dims,
        dim_embedding=args.dim_emb,
        alpha=args.alpha,
        likelihood=args.likelihood,
    ).to(device)

    if args.freeze_readout_to_true:
        if args.likelihood != "poisson":
            raise ValueError("--freeze-readout-to-true only supports --likelihood poisson")
        with torch.no_grad():
            for ds in data.observations:
                C_true = data.observation_matrices[ds].to(device)   # (n_obs, d_latent)
                b_true = data.biases[ds].to(device).squeeze(0)       # (n_obs,)  type: ignore[attr-defined]
                readout = model.adapters.likelihood[ds].readout
                assert readout.weight.shape == C_true.shape, \
                    f"shape mismatch {readout.weight.shape} vs {C_true.shape}"
                assert readout.bias.shape == b_true.shape, \
                    f"shape mismatch {readout.bias.shape} vs {b_true.shape}"
                readout.weight.copy_(C_true)
                readout.bias.copy_(b_true)
                readout.weight.requires_grad = False
                readout.bias.requires_grad = False
        n_frozen = sum(
            p.numel() for ds in data.observations
            for p in model.adapters.likelihood[ds].readout.parameters()
        )
        print(f"froze {n_frozen:,} parameters in PoissonLikelihood readouts "
              f"to ground-truth (C, b)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters on {device} ({n_trainable:,} trainable)")

    keys = list(data.observations.keys())
    per_step = args.per_step if args.per_step > 0 else len(keys)
    rng = random.Random(args.seed + 2)

    warm_history = []
    if args.warm_start_steps > 0:
        warm_history = warm_start_dynamics(
            model, data,
            num_steps=args.warm_start_steps,
            lr=args.warm_start_lr,
            batch=args.warm_start_batch,
            rng=random.Random(args.seed + 3),
        )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
    )
    batch_gen = torch.Generator(device=device)
    batch_gen.manual_seed(args.seed + 1)

    metrics: list[dict] = []
    print(f"\n{'step':>6} {'loss':>10} {'r2_med':>7} {'rho':>7}")
    diag0 = diagnostics(model, data)
    metrics.append({"step": 0, "loss": float("nan"), **diag0})
    print(f"{'init':>6} {' ':>10} {diag0['r2_median']:>7.3f} {diag0['spearman']:>7.3f}")

    for step in range(1, args.steps + 1):
        subset = rng.sample(keys, per_step) if per_step < len(keys) else keys
        obs_subset = {k: data.observations[k] for k in subset}
        batch = sample_batch(obs_subset, args.batch, generator=batch_gen)

        optimizer.zero_grad(set_to_none=True)
        out = model(batch, p_mask=args.p_mask)
        loss = out["loss"].mean()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if step % args.log_every == 0:
            metrics.append({"step": step, "loss": loss_value})

        if step % args.eval_every == 0 or step == args.steps:
            diag = diagnostics(model, data)
            metrics[-1].update(diag)
            print(f"{step:>6} {loss_value:>10.2f} {diag['r2_median']:>7.3f} {diag['spearman']:>7.3f}")

        if args.snapshot_every > 0 and (step % args.snapshot_every == 0 or step == args.steps):
            snap_dir = out_dir / "snapshots"
            snap_dir.mkdir(exist_ok=True)
            diag_snap = diagnostics(model, data)
            summary = collect_summary(model, data)
            torch.save(
                {
                    "step": step,
                    "loss": loss_value,
                    "diag": diag_snap,
                    "summary": summary,
                    "state_dict": model.state_dict(),
                },
                snap_dir / f"step_{step:05d}.pt",
            )

    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")

    final = collect_final(model, data)

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    if warm_history:
        with (out_dir / "warm_start.json").open("w") as f:
            json.dump(warm_history, f, indent=2)
    summary = {
        "config": config,
        "elapsed_sec": elapsed,
        "n_params": n_params,
        "warm_start_steps": args.warm_start_steps,
        "final": {k: v for k, v in metrics[-1].items() if isinstance(v, (int, float))},
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), out_dir / "state.pt")
    torch.save(final, out_dir / "diagnostics.pt")

    make_plots(metrics, final, out_dir, args.dim_emb)
    print(f"wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
