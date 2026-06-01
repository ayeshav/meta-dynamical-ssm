"""Phase 3 full-scale Poisson sweep summary.

Two parallel A100 configs (N=100, T=1200, n in [800, 1200], 4000 steps):
  P3_pca_full_scale:    PCA warm-start, no other intervention
  P3_pca_wd_full_scale: PCA + readout-weight-decay 1e-2

Plus a baseline reference (A100 long-trial sweep without warm-start)
showing the collapsed trajectory.
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
    cfgs = [
        ("PCA warm-start", "tab:blue",
         Path("gcp_runs/exp-20260531-182842-phase23-a100/results_phase3/P3_pca_full_scale")),
        ("PCA + wd 1e-2", "tab:red",
         Path("gcp_runs/exp-20260531-212657-phase3wd-a100/results_phase3/P3_pca_wd_full_scale")),
        ("baseline (no warm-start, prior run)", "tab:gray",
         Path("gcp_runs/exp-20260529-212015-poisson-a100/results/poisson_T1200")),
    ]

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 3)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_rho = fig.add_subplot(gs[0, 2])
    ax_emb_a = fig.add_subplot(gs[1, 0])
    ax_emb_b = fig.add_subplot(gs[1, 1])
    ax_scatter = fig.add_subplot(gs[1, 2])

    for label, color, cfg_dir in cfgs:
        traj = trajectory(cfg_dir)
        if not traj:
            continue
        steps = [t["step"] for t in traj]
        ax_loss.plot(steps, [t["loss"] for t in traj], color=color, lw=1.3, label=label)
        ax_r2.plot(steps, [t["r2"] for t in traj], color=color, lw=1.3)
        ax_rho.plot(steps, [t["abs_rho"] for t in traj], color=color, lw=1.3)

    ax_loss.set_yscale("log"); ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss (log)")
    ax_loss.set_title("(a) loss", fontsize=10, loc="left")
    ax_loss.legend(frameon=False, fontsize=7, loc="upper right")
    ax_r2.set_xlabel("step"); ax_r2.set_ylabel(r"R$^2$ (median)")
    ax_r2.set_ylim(-1, 1.05); ax_r2.axhline(0, color="gray", lw=0.5)
    ax_r2.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax_r2.set_title("(b) rate R$^2$", fontsize=10, loc="left")
    ax_rho.set_xlabel("step"); ax_rho.set_ylabel(r"$|$Spearman$(|PC_1|, |\omega|)|$")
    ax_rho.set_ylim(-0.05, 1.05); ax_rho.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax_rho.set_title("(c) |omega| recovery", fontsize=10, loc="left")

    # bottom-left + bottom-mid: final embeddings of the two warm-start configs
    for ax, (label, _color, cfg_dir) in zip([ax_emb_a, ax_emb_b], cfgs[:2]):
        mu_e, om, step = final_emb(cfg_dir)
        if mu_e is None:
            ax.axis("off"); continue
        ax.scatter(mu_e[:, 0], mu_e[:, 1], c=om, cmap="viridis", s=18,
                   edgecolor="black", lw=0.2)
        ax.set_xlabel(r"$\mu_e[0]$"); ax.set_ylabel(r"$\mu_e[1]$")
        ax.set_title(f"({label}) step {step}", fontsize=9, loc="left")

    # bottom-right: |PC1| vs |omega| scatter for the winning config
    mu_e, om, step = final_emb(cfgs[0][2])
    if mu_e is not None:
        p1 = pc1(mu_e)
        ax_scatter.scatter(om.abs(), p1.abs(), c=om.abs(), cmap="viridis",
                            s=18, edgecolor="black", lw=0.2)
        rho = spearman(p1.abs(), om.abs())
        ax_scatter.set_xlabel(r"$|\omega|$")
        ax_scatter.set_ylabel(r"$|PC_1(\mu_e)|$")
        ax_scatter.set_title(f"(f) folded recovery, |omega| Spearman = {rho:+.3f}",
                              fontsize=9, loc="left")

    fig.suptitle("Phase 3: PCA warm-start recovers the 1-D omega family at full Poisson scale",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path("gcp_runs/exp-20260531-182842-phase23-a100/results_phase3/phase3_summary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
