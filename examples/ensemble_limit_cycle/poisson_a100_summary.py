"""Compare the two A100 long-trial Poisson runs to the L4 baseline.

Single figure:
  top row: convergence of (loss, rate R^2, |omega|-Spearman) for the
    three runs L4-T100, A100-T1200, A100-T2400 vs training step.
  bottom row: final embeddings (PC1 vs PC2) colored by omega for each
    of the three runs.

The negative result is the point: T1200 and T2400 collapse on the same
curve, just slower than the L4 baseline (which already collapsed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def spearman(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float), torch.as_tensor(b, dtype=torch.float)
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (torch.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def pc1(mu_e):
    M = torch.as_tensor(mu_e, dtype=torch.float)
    if M.shape[1] == 1:
        return M[:, 0]
    Mc = M - M.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(Mc, full_matrices=False)
    return Mc @ v[0]


def trajectory(cfg_dir: Path):
    snaps = sorted((cfg_dir / "snapshots").glob("step_*.pt"))
    rows = []
    for sp in snaps:
        try:
            s = torch.load(sp, weights_only=False, map_location="cpu")
        except Exception:
            continue
        om = s["summary"]["omegas"]
        p1 = pc1(s["summary"]["mu_e"])
        rows.append({
            "step": s["step"],
            "loss": float(s["loss"]),
            "r2": s["diag"]["r2_median"],
            "abs_rho": spearman(p1.abs(), torch.as_tensor(om).abs()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("gcp_runs/exp-20260529-212015-poisson-a100/results/comparison_summary.png"))
    args = ap.parse_args()

    runs = [
        ("L4 baseline (T=100)", "tab:gray",
         Path("gcp_runs/exp-20260529-171902-poisson-gpu/results/poisson_snr20")),
        ("A100 (T=1200)", "tab:blue",
         Path("gcp_runs/exp-20260529-212015-poisson-a100/results/poisson_T1200")),
        ("A100 (T=2400)", "tab:red",
         Path("gcp_runs/exp-20260529-212015-poisson-a100/results/poisson_T2400")),
    ]

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))

    for label, color, cfg_dir in runs:
        traj = trajectory(cfg_dir)
        if not traj:
            continue
        steps = [t["step"] for t in traj]
        axes[0, 0].plot(steps, [t["loss"] for t in traj], color=color, lw=1.3, label=label)
        axes[0, 1].plot(steps, [t["r2"] for t in traj], color=color, lw=1.3)
        axes[0, 2].plot(steps, [t["abs_rho"] for t in traj], color=color, lw=1.3)

    axes[0, 0].set_yscale("log"); axes[0, 0].set_ylabel("loss (log)")
    axes[0, 0].set_title("(a) loss", fontsize=10, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")

    axes[0, 1].set_ylabel(r"R$^2$ (rate, median)"); axes[0, 1].set_ylim(-1, 1.05)
    axes[0, 1].axhline(0, color="gray", lw=0.5)
    axes[0, 1].axhline(0.9, color="gray", lw=0.5, ls=":")
    axes[0, 1].set_title("(b) rate R$^2$ (target 0.9)", fontsize=10, loc="left")

    axes[0, 2].set_ylabel(r"$|$Spearman$(|PC_1(\mu_e)|, |\omega|)|$")
    axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].axhline(0.9, color="gray", lw=0.5, ls=":")
    axes[0, 2].set_title("(c) |omega| recovery", fontsize=10, loc="left")
    for ax in axes[0]:
        ax.set_xlabel("step")

    # Bottom row: final embedding scatter per run.
    for ax, (label, color, cfg_dir) in zip(axes[1], runs):
        # Use the last available snapshot for embedding.
        snaps = sorted((cfg_dir / "snapshots").glob("step_*.pt"))
        snap = None
        for sp in snaps[::-1]:
            try:
                snap = torch.load(sp, weights_only=False, map_location="cpu")
                break
            except Exception:
                continue
        if snap is None:
            ax.axis("off"); continue
        mu_e = torch.as_tensor(snap["summary"]["mu_e"])
        om = torch.as_tensor(snap["summary"]["omegas"])
        ax.scatter(mu_e[:, 0], mu_e[:, 1], c=om, cmap="viridis", s=18, edgecolor="black", lw=0.2)
        ax.set_xlabel(r"$\mu_e[0]$"); ax.set_ylabel(r"$\mu_e[1]$")
        ax.set_title(f"{label} (step {snap['step']})", fontsize=9, loc="left")

    fig.suptitle("Poisson observation experiment: longer trials do not break the collapse",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
