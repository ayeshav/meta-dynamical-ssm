"""Phase 1 one-axis sweep summary for the Poisson collapse fix.

5 configs at (N=30, T=200, n=400, num_trials=64, 1500 steps):
baseline / oracle warm-start / STA warm-start / weight decay / lr scale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def spearman(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float), torch.as_tensor(b, dtype=torch.float)
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
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
            "r2": s["diag"]["r2_median"],
            "abs_rho": spearman(p1.abs(), torch.as_tensor(om).abs()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("gcp_runs/exp-20260531-145934-phase1-a100/results"))
    args = ap.parse_args()

    cfgs = [
        ("P1_baseline", "baseline (no intervention)", "tab:gray"),
        ("P1_oracle",   "oracle init (C,b)",          "tab:green"),
        ("P1_sta",      "STA init (data only)",       "tab:blue"),
        ("P1_wd",       "weight-decay 1e-2",          "tab:orange"),
        ("P1_lrscale",  "readout lr x 0.1",           "tab:red"),
    ]

    plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, label, color in cfgs:
        traj = trajectory(args.root / name)
        if not traj:
            continue
        steps = [t["step"] for t in traj]
        axes[0].plot(steps, [t["r2"] for t in traj], color=color, lw=1.5, label=label)
        axes[1].plot(steps, [t["abs_rho"] for t in traj], color=color, lw=1.5, label=label)

    axes[0].set_ylabel(r"R$^2$ (rate, median)")
    axes[0].set_xlabel("step")
    axes[0].set_ylim(-1, 1.05)
    axes[0].axhline(0.9, color="gray", lw=0.5, ls=":")
    axes[0].set_title("(a) rate R$^2$", fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    axes[1].set_ylabel(r"$|$Spearman$(|PC_1|, |\omega|)|$")
    axes[1].set_xlabel("step")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(0.9, color="gray", lw=0.5, ls=":")
    axes[1].set_title("(b) |omega| recovery", fontsize=11, loc="left")

    fig.suptitle("Phase 1: oracle and STA warm-start both reach R² ≈ 0.94, "
                 "|omega| rho ≈ 0.99.\nWeight decay alone and lr-scale alone don't help.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = args.root / "phase1_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
