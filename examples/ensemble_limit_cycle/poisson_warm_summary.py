"""One-figure summary of the warm-start + 10x trials A100 sweep.

Single figure showing warm vs no-warm side-by-side on:
  (a) loss trajectory (plus warm-start MSE plotted on a twin axis)
  (b) rate R^2 vs step
  (c) |omega| recovery Spearman vs step (rectified PC1)
  (d-e) final embedding scatter for each config
The point: identical trajectories. Warm-start the base MLP does not
help.
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


def trajectory(cfg_dir):
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


def final_emb(cfg_dir):
    snaps = sorted((cfg_dir / "snapshots").glob("step_*.pt"))
    for sp in snaps[::-1]:
        try:
            s = torch.load(sp, weights_only=False, map_location="cpu")
            return torch.as_tensor(s["summary"]["mu_e"]), \
                   torch.as_tensor(s["summary"]["omegas"]), s["step"]
        except Exception:
            continue
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("gcp_runs/exp-20260530-213415-warmstart-a100/results"))
    args = ap.parse_args()

    cfgs = [
        ("poisson_warm", "warm-start 1000 steps", "tab:green"),
        ("poisson_nowarm", "no warm-start (control)", "tab:gray"),
    ]

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(10.5, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_rho = fig.add_subplot(gs[0, 2])
    ax_emb_warm = fig.add_subplot(gs[1, 0])
    ax_emb_nowarm = fig.add_subplot(gs[1, 1])
    ax_warm = fig.add_subplot(gs[1, 2])

    for name, label, color in cfgs:
        cfg_dir = args.root / name
        traj = trajectory(cfg_dir)
        if not traj:
            continue
        steps = [t["step"] for t in traj]
        ax_loss.plot(steps, [t["loss"] for t in traj], color=color, lw=1.3, label=label)
        ax_r2.plot(steps, [t["r2"] for t in traj], color=color, lw=1.3)
        ax_rho.plot(steps, [t["abs_rho"] for t in traj], color=color, lw=1.3)

    ax_loss.set_yscale("log"); ax_loss.set_ylabel("loss (log)")
    ax_loss.set_xlabel("step"); ax_loss.set_title("(a) loss", fontsize=10, loc="left")
    ax_loss.legend(frameon=False, fontsize=8, loc="upper right")
    ax_r2.set_ylabel(r"R$^2$ (rate, median)"); ax_r2.set_xlabel("step")
    ax_r2.set_ylim(-1, 1.05); ax_r2.axhline(0, color="gray", lw=0.5)
    ax_r2.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax_r2.set_title("(b) rate R$^2$ (target 0.9)", fontsize=10, loc="left")
    ax_rho.set_ylabel(r"$|$Spearman$(|PC_1(\mu_e)|, |\omega|)|$")
    ax_rho.set_xlabel("step")
    ax_rho.set_ylim(-0.05, 1.05); ax_rho.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax_rho.set_title("(c) |omega| recovery", fontsize=10, loc="left")

    # bottom-left + bottom-mid: final embeddings
    for ax, (name, label, color) in zip([ax_emb_warm, ax_emb_nowarm], cfgs):
        mu_e, om, step = final_emb(args.root / name)
        if mu_e is None:
            ax.axis("off"); continue
        sc = ax.scatter(mu_e[:, 0], mu_e[:, 1], c=om, cmap="viridis", s=18,
                        edgecolor="black", lw=0.2)
        ax.set_xlabel(r"$\mu_e[0]$"); ax.set_ylabel(r"$\mu_e[1]$")
        ax.set_title(f"(d/{name}) step {step}", fontsize=9, loc="left")

    # bottom-right: warm-start training loss
    wpath = args.root / "poisson_warm" / "warm_start.json"
    if wpath.exists():
        ws = json.loads(wpath.read_text())
        ax_warm.plot([w["step"] for w in ws], [w["loss"] for w in ws],
                     color="tab:green", lw=1.2)
        ax_warm.set_yscale("log")
        ax_warm.set_xlabel("warm-start step")
        ax_warm.set_ylabel("MSE on true z trajectory")
        ax_warm.set_title("(e) warm-start: MlpDynamics learns z dynamics",
                          fontsize=9, loc="left")
    else:
        ax_warm.axis("off")

    fig.suptitle("Poisson observation: 10x trials + warm-start MlpDynamics "
                 "(both interventions, no effect on collapse)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.root / "warm_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
