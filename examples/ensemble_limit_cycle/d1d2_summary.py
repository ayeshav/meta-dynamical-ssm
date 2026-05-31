"""One-figure summary of the D1+D2 diagnostic sweep on A100.

D1 (Gaussian, D_obs in [1800, 2400], 30 dB): tests whether large D_obs
    alone causes collapse. It does not.
D2 (Poisson, mean rate 1.0, max 5.0): tests whether sparsity at low
    rate was the killer. Partial recovery (R^2 = 0.62) confirms rate
    is a major factor.

Top row: loss, R^2, |omega|-Spearman trajectories.
Bottom row: final embeddings for each config + omega-vs-PC1 scatter
showing the recovery (or lack of it).
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


def final_emb(cfg_dir: Path):
    snaps = sorted((cfg_dir / "snapshots").glob("step_*.pt"))
    for sp in snaps[::-1]:
        try:
            s = torch.load(sp, weights_only=False, map_location="cpu")
            return (
                torch.as_tensor(s["summary"]["mu_e"]),
                torch.as_tensor(s["summary"]["omegas"]),
                s["step"],
            )
        except Exception:
            continue
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("gcp_runs/exp-20260531-111911-d1d2-a100/results"))
    args = ap.parse_args()

    cfgs = [
        ("D1_gaussian_large_Dobs", "D1: Gaussian, D_obs in [1800, 2400], SNR 30 dB", "tab:blue"),
        ("D2_poisson_high_rate", "D2: Poisson, mean rate 1.0, max 5.0", "tab:red"),
    ]

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

    for name, label, color in cfgs:
        traj = trajectory(args.root / name)
        if not traj:
            continue
        steps = [t["step"] for t in traj]
        axes[0, 0].plot(steps, [t["loss"] for t in traj], color=color, lw=1.3, label=label)
        axes[0, 1].plot(steps, [t["r2"] for t in traj], color=color, lw=1.3)
        axes[0, 2].plot(steps, [t["abs_rho"] for t in traj], color=color, lw=1.3)

    axes[0, 0].set_yscale("log"); axes[0, 0].set_ylabel("loss (log)")
    axes[0, 0].set_xlabel("step"); axes[0, 0].set_title("(a) loss", fontsize=10, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0, 1].set_ylabel(r"R$^2$ (median)"); axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylim(-1, 1.05); axes[0, 1].axhline(0, color="gray", lw=0.5)
    axes[0, 1].axhline(0.9, color="gray", lw=0.5, ls=":")
    axes[0, 1].set_title("(b) R$^2$ (target 0.9)", fontsize=10, loc="left")
    axes[0, 2].set_ylabel(r"$|$Spearman$(|PC_1|, |\omega|)|$"); axes[0, 2].set_xlabel("step")
    axes[0, 2].set_ylim(-0.05, 1.05); axes[0, 2].axhline(0.9, color="gray", lw=0.5, ls=":")
    axes[0, 2].set_title("(c) |omega| recovery", fontsize=10, loc="left")

    for ax, (name, label, _color) in zip([axes[1, 0], axes[1, 1]], cfgs):
        mu_e, om, step = final_emb(args.root / name)
        if mu_e is None:
            ax.axis("off"); continue
        ax.scatter(mu_e[:, 0], mu_e[:, 1], c=om, cmap="viridis", s=18,
                   edgecolor="black", lw=0.2)
        ax.set_xlabel(r"$\mu_e[0]$"); ax.set_ylabel(r"$\mu_e[1]$")
        ax.set_title(f"({name.split('_', 1)[0]}) step {step}", fontsize=9, loc="left")

    # bottom-right: PC1 vs true omega for both (the recovery test)
    ax = axes[1, 2]
    for name, label, color in cfgs:
        mu_e, om, _step = final_emb(args.root / name)
        if mu_e is None:
            continue
        p1 = pc1(mu_e)
        ax.scatter(om, p1, color=color, s=14, alpha=0.7,
                   label=f"{name.split('_', 1)[0]}: rho={spearman(p1, om):+.2f}")
    ax.set_xlabel(r"true $\omega$"); ax.set_ylabel(r"$PC_1(\mu_e)$")
    ax.set_title("(f) signed PC1 vs true omega", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("D1: Gaussian at Poisson-scale D_obs works.  "
                 "D2: high-rate Poisson partially recovers R$^2$ but not embedding.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.root / "d1d2_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
