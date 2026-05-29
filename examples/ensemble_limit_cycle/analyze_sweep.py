"""Cross-config analysis for the meta-SSM hyperparameter sweep.

Reads each config dir under <sweep-dir>, extracts the final metrics +
per-dataset delta norms and embeddings, and writes:
  cross_config_table.json
  cross_config_summary.png   (alpha x [R^2, |rho|, ||delta||])
  embeddings_grid.png        (final mu_e per config, 6-panel grid)
  embedding_evolution_*.png  (per-config snapshot evolution)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def load_config(cfg_dir: Path) -> dict:
    summary = json.loads((cfg_dir / "summary.json").read_text())
    diag = torch.load(cfg_dir / "diagnostics.pt", weights_only=False)
    return {
        "name": cfg_dir.name,
        "alpha": summary["config"]["alpha"],
        "dim_emb": summary["config"]["dim_emb"],
        "elapsed_sec": summary["elapsed_sec"],
        "final": summary["final"],
        "omegas": diag["omegas"],
        "mu_e": diag["mu_e"],
        "delta_norms": diag["delta_norms"],
    }


def load_snapshots(cfg_dir: Path) -> list[dict]:
    snap_dir = cfg_dir / "snapshots"
    if not snap_dir.exists():
        return []
    out = []
    for path in sorted(snap_dir.glob("step_*.pt")):
        snap = torch.load(path, weights_only=False)
        out.append(
            {
                "step": snap["step"],
                "diag": snap["diag"],
                "mu_e": snap["summary"]["mu_e"],
                "delta_norms": snap["summary"]["delta_norms"],
                "omegas": snap["summary"]["omegas"],
            }
        )
    return out


def cfg_label(c: dict) -> str:
    return f"alpha={c['alpha']:g}, dim_emb={c['dim_emb']}"


def cross_config_plots(configs: list[dict], out_dir: Path):
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})
    n = len(configs)

    # --- summary table panel: alpha x dim_emb -> R^2, |rho|, mean ||delta|| ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    alphas = [c["alpha"] for c in configs]
    r2 = [c["final"]["r2_median"] for c in configs]
    rho_abs = [abs(c["final"]["spearman"]) for c in configs]
    delta_mean = [
        float(torch.tensor(c["delta_norms"]).mean()) for c in configs
    ]
    dim_embs = [c["dim_emb"] for c in configs]
    color_map = {1: "C0", 2: "C1", 4: "C2"}
    colors = [color_map[d] for d in dim_embs]

    # tiny x-jitter so overlapping alpha values don't collide
    eps = {1: -0.12, 2: 0.0, 4: 0.12}
    x_jitter = [(a if a > 0 else 0.005) * (1 + eps[d]) for a, d in zip(alphas, dim_embs)]

    for ax, vals, title, yscale in [
        (axes[0], r2, "R^2 median", "linear"),
        (axes[1], rho_abs, "|Spearman rho|", "linear"),
        (axes[2], delta_mean, "mean ||delta||", "log"),
    ]:
        for x, y, c, d in zip(x_jitter, vals, colors, dim_embs):
            ax.scatter(x, y, s=70, color=c, edgecolor="black", lw=0.4, zorder=3,
                       label=f"dim_emb={d}")
        ax.set_xscale("symlog", linthresh=0.005)
        ax.set_xlabel("alpha")
        ax.set_title(title, fontsize=10)
        if yscale == "log":
            ax.set_yscale("log")
        # dedup legend
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.0)
    axes[1].axhline(0.7, color="gray", lw=0.6, ls="--")
    fig.suptitle("Sweep summary (final values at step 2000)", fontsize=11)
    fig.savefig(out_dir / "cross_config_summary.png")
    plt.close(fig)

    # --- embeddings grid: final mu_e per config ---
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows),
                             squeeze=False)
    axes_flat = list(axes.flat)
    for ax, c in zip(axes_flat, configs):
        mu_e = torch.tensor(c["mu_e"])
        omegas = torch.tensor(c["omegas"])
        if mu_e.shape[1] == 1:
            sc = ax.scatter(omegas, mu_e[:, 0], c=omegas, cmap="viridis", s=18)
            ax.set_xlabel("omega")
            ax.set_ylabel("mu_e")
        else:
            sc = ax.scatter(mu_e[:, 0], mu_e[:, 1], c=omegas, cmap="viridis", s=18)
            ax.set_xlabel("mu_e[0]")
            ax.set_ylabel("mu_e[1]")
        rho = c["final"]["spearman"]
        ax.set_title(f"{cfg_label(c)}\nR^2={c['final']['r2_median']:.3f}, rho={rho:+.2f}",
                     fontsize=9)
    for ax in axes_flat[len(configs):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "embeddings_grid.png")
    plt.close(fig)


def per_config_evolution(cfg_dir: Path, out_dir: Path):
    cfg = load_config(cfg_dir)
    snaps = load_snapshots(cfg_dir)
    if not snaps:
        return
    nshow = min(6, len(snaps))
    idx = [int(round(i * (len(snaps) - 1) / (nshow - 1))) for i in range(nshow)]
    fig, axes = plt.subplots(1, nshow, figsize=(2.6 * nshow, 2.8))
    for ax, k in zip(axes, idx):
        snap = snaps[k]
        mu_e = torch.tensor(snap["mu_e"])
        omegas = torch.tensor(snap["omegas"])
        if mu_e.shape[1] == 1:
            ax.scatter(omegas, mu_e[:, 0], c=omegas, cmap="viridis", s=10)
        else:
            ax.scatter(mu_e[:, 0], mu_e[:, 1], c=omegas, cmap="viridis", s=10)
        rho = snap["diag"]["spearman"]
        ax.set_title(f"step {snap['step']}\nrho={rho:+.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Embedding evolution: {cfg_label(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"embedding_evolution_{cfg['name']}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    args = ap.parse_args()

    cfg_dirs = sorted(p for p in args.sweep_dir.iterdir()
                      if p.is_dir() and (p / "summary.json").exists())
    configs = [load_config(d) for d in cfg_dirs]
    configs.sort(key=lambda c: (c["alpha"], c["dim_emb"]))

    out_dir = args.sweep_dir
    cross_config_plots(configs, out_dir)

    for d in cfg_dirs:
        per_config_evolution(d, out_dir)

    # table
    table = []
    for c in configs:
        delta_t = torch.tensor(c["delta_norms"])
        table.append({
            "name": c["name"],
            "alpha": c["alpha"],
            "dim_emb": c["dim_emb"],
            "elapsed_sec": c["elapsed_sec"],
            "loss": c["final"]["loss"],
            "r2_median": c["final"]["r2_median"],
            "r2_min": c["final"]["r2_min"],
            "spearman": c["final"]["spearman"],
            "emb_range": c["final"]["emb_range"],
            "delta_mean": float(delta_t.mean()),
            "delta_max": float(delta_t.max()),
        })
    (out_dir / "cross_config_table.json").write_text(json.dumps(table, indent=2))

    # print table
    keys = ["name", "alpha", "dim_emb", "r2_median", "spearman", "emb_range", "delta_mean"]
    print(" | ".join(f"{k:>10}" for k in keys))
    for row in table:
        print(" | ".join(
            f"{row[k]:>10.4f}" if isinstance(row[k], float) else f"{str(row[k]):>10}"
            for k in keys
        ))


if __name__ == "__main__":
    main()
