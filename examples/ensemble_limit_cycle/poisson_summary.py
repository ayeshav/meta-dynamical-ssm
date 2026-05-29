"""One-figure summary of the Poisson SNR sweep.

Top row, three panels: convergence trajectory for each SNR config
(loss, R^2 of rate recovery, |omega| Spearman of rectified embedding).

Bottom row, three panels: 2-D inferred posterior-mean latent trajectory
(blue) overlaid on the true limit-cycle latent (gray) for omega = -5,
0, +5 at the high-SNR config. Shows the posterior collapse to a 1-D
line that is the failure mode in this sweep.
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


def load_trajectory(cfg_dir):
    """Per-step (loss, R^2 median, |omega| rho)."""
    metrics = json.loads((cfg_dir / "metrics.json").read_text()) if (cfg_dir / "metrics.json").exists() else None
    snap_dir = cfg_dir / "snapshots"
    snaps = sorted(snap_dir.glob("step_*.pt"))
    traj = []
    for sp in snaps:
        try:
            s = torch.load(sp, weights_only=False, map_location="cpu")
        except Exception:
            continue
        om = s["summary"]["omegas"]
        p1 = pc1(s["summary"]["mu_e"])
        traj.append({
            "step": s["step"],
            "loss": float(s["loss"]),
            "r2_median": s["diag"]["r2_median"],
            "abs_rho": spearman(p1.abs(), torch.tensor(om).abs()),
        })
    return traj, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--anchor", default="poisson_snr20")
    args = ap.parse_args()

    cfg_dirs = sorted(p for p in args.sweep_dir.iterdir()
                      if p.is_dir() and (p / "summary.json").exists())
    cfgs = []
    for d in cfg_dirs:
        summary = json.loads((d / "summary.json").read_text())
        traj, _ = load_trajectory(d)
        cfgs.append({
            "name": d.name,
            "snr": summary["config"]["snr_db"],
            "summary": summary,
            "traj": traj,
            "dir": d,
        })
    cfgs.sort(key=lambda c: -c["snr"])

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))
    color_map = {20.0: "#1f77b4", 15.0: "#d62728", 10.0: "#2ca02c"}

    # --- Top row: convergence per config ---
    for c in cfgs:
        col = color_map.get(c["snr"], "gray")
        steps = [t["step"] for t in c["traj"]]
        axes[0, 0].plot(steps, [t["loss"] for t in c["traj"]],
                        color=col, lw=1.3, label=f"SNR={c['snr']:.0f} dB")
        axes[0, 1].plot(steps, [t["r2_median"] for t in c["traj"]],
                        color=col, lw=1.3)
        axes[0, 2].plot(steps, [t["abs_rho"] for t in c["traj"]],
                        color=col, lw=1.3)

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("loss (log)")
    axes[0, 0].set_title("(a) loss", fontsize=10, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")

    axes[0, 1].set_ylabel(r"R$^2$ (rate, median)")
    axes[0, 1].set_ylim(-1, 1.05)
    axes[0, 1].axhline(0, color="gray", lw=0.5)
    axes[0, 1].axhline(0.9, color="gray", lw=0.5, ls=":")
    axes[0, 1].set_title("(b) rate R$^2$ (target = 0.9)", fontsize=10, loc="left")

    axes[0, 2].set_ylabel(r"$|$Spearman$(|PC_1(\mu_e)|, |\omega|)|$")
    axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].axhline(0.9, color="gray", lw=0.5, ls=":")
    axes[0, 2].set_title("(c) |omega| recovery", fontsize=10, loc="left")
    for ax in axes[0]:
        ax.set_xlabel("step")

    # --- Bottom row: dynamics collapse at the anchor config ---
    anchor = next((c for c in cfgs if c["name"] == args.anchor), cfgs[0])
    diag_pt = anchor["dir"] / "diagnostics.pt"
    if diag_pt.exists():
        diag = torch.load(diag_pt, weights_only=False, map_location="cpu")
        omegas = diag["omegas"]
        z_true = diag["z_true"]
        mu_q = diag["mu_q"]
    else:
        # Fall back to the last available snapshot (used when the rsync
        # poller missed the final snapshot save). We don't have mu_q here,
        # so just show the embedding scatter in the bottom row.
        diag = None

    if diag is not None:
        # Pick omega ~ -5, 0, +5.
        order = sorted(range(len(omegas)), key=lambda i: omegas[i])
        pick = [order[0], order[len(order) // 2], order[-1]]
        for ax, idx in zip(axes[1], pick):
            zt = z_true[idx]
            mq = mu_q[idx]
            n_show = min(6, zt.shape[0])
            for tr in range(n_show):
                ax.plot(zt[tr, :, 0], zt[tr, :, 1], color="gray", lw=0.7, alpha=0.7)
                ax.plot(mq[tr, :, 0], mq[tr, :, 1], color="C0", lw=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(f"omega={omegas[idx]:+.2f}", fontsize=9)
        fig.text(0.5, 0.46,
                 f"(d) inferred posterior mean (blue) vs true limit cycle (gray) at SNR={anchor['snr']:.0f} dB",
                 ha="center", fontsize=9)
    else:
        # Backup: embedding scatter.
        snap_dir = anchor["dir"] / "snapshots"
        snaps = sorted(snap_dir.glob("step_*.pt"))
        for sp in snaps[::-1]:
            try:
                s = torch.load(sp, weights_only=False, map_location="cpu")
                break
            except Exception:
                continue
        mu_e = torch.tensor(s["summary"]["mu_e"])
        om = torch.tensor(s["summary"]["omegas"])
        for j in range(3):
            axes[1, j].axis("off")
        ax = axes[1, 1]
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(mu_e[:, 0], mu_e[:, 1], c=om, cmap="viridis", s=20)
        ax.set_title(f"(d) embedding scatter at SNR={anchor['snr']:.0f} dB (step {s['step']})", fontsize=9, loc="left")

    fig.suptitle("Poisson-observation sweep summary "
                 "(N=100 datasets, log-linear log-rate, alpha=0.01, dim_emb=2)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = args.sweep_dir / "poisson_summary.png"
    fig.savefig(out)
    plt.close(fig)

    # Cross-config scalar table.
    table = []
    for c in cfgs:
        snap_traj = c["traj"]
        last = snap_traj[-1] if snap_traj else None
        table.append({
            "name": c["name"],
            "snr_target": c["snr"],
            "snr_realized_mean": _mean_realized(c["dir"]),
            "n_steps": c["summary"]["config"]["steps"],
            "loss_final": c["summary"]["final"]["loss"],
            "r2_median_final": c["summary"]["final"]["r2_median"],
            "abs_omega_rho_final": last["abs_rho"] if last else None,
            "emb_range_final": c["summary"]["final"]["emb_range"],
        })
    (args.sweep_dir / "poisson_summary.json").write_text(json.dumps(table, indent=2))
    for row in table:
        print(row)
    print(f"\nwrote {out}")


def _mean_realized(cfg_dir):
    log_path = cfg_dir.with_suffix(".log")
    if not log_path.exists():
        return None
    for line in log_path.read_text().splitlines():
        if "realized SNR mean=" in line:
            try:
                return float(line.split("realized SNR mean=")[1].split(" ")[0])
            except Exception:
                pass
    return None


if __name__ == "__main__":
    main()
