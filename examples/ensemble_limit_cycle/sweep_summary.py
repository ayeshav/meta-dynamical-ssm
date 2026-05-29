"""One-figure summary of the alpha x dim_embedding sweep.

The latent dynamics are rotation-invariant: a flip of the latent basis
(z -> -z) is absorbed by per-dataset readin/readout adapters. So
sign(omega) is not identifiable from observations alone, and the meta-
learning question is whether the model recovers |omega| (rotation
magnitude) as a 1-D family in embedding space.

Reads each config dir in <sweep-dir>, computes
  Spearman( |PC1(mu_e)|, |omega| )            -- 1-D recovery, identifiable
  mean ||delta|| across datasets and LoRA layers
and writes one figure: sweep_summary.png.

Top row (alpha across configs, color by dim_embedding):
  (a) |omega|-recovery Spearman
  (b) mean ||delta||  (log y)

Bottom row: canonical example (alpha=0, dim_emb=1)
  (c) mu_e vs omega   -- X-shape from sign ambiguity
  (d) |mu_e| vs |omega| -- rectified, ~linear
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def pc1(mu_e):
    M = torch.tensor(mu_e, dtype=torch.float)
    if M.shape[1] == 1:
        return M[:, 0]
    Mc = M - M.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(Mc, full_matrices=False)
    return Mc @ v[0]


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (torch.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def load_config(cfg_dir: Path) -> dict:
    summary = json.loads((cfg_dir / "summary.json").read_text())
    diag = torch.load(cfg_dir / "diagnostics.pt", weights_only=False)
    return {
        "name": cfg_dir.name,
        "alpha": summary["config"]["alpha"],
        "dim_emb": summary["config"]["dim_emb"],
        "final": summary["final"],
        "omegas": diag["omegas"],
        "mu_e": diag["mu_e"],
        "delta_norms": diag["delta_norms"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    args = ap.parse_args()

    cfg_dirs = sorted(p for p in args.sweep_dir.iterdir()
                      if p.is_dir() and (p / "summary.json").exists())
    configs = [load_config(d) for d in cfg_dirs]
    configs.sort(key=lambda c: (c["alpha"], c["dim_emb"]))

    # Augment with corrected metrics.
    for c in configs:
        omegas = torch.tensor(c["omegas"], dtype=torch.float)
        p1 = pc1(c["mu_e"])
        c["abs_rho"] = spearman(p1.abs(), omegas.abs())
        c["signed_rho"] = c["final"]["spearman"]
        c["delta_mean"] = float(torch.tensor(c["delta_norms"]).mean())

    # Persist corrected table.
    (args.sweep_dir / "sweep_summary.json").write_text(json.dumps(
        [{k: v for k, v in c.items()
          if k in ("name", "alpha", "dim_emb", "abs_rho", "signed_rho", "delta_mean")
          or k == "final"}
         for c in configs],
        indent=2, default=str,
    ))

    # Canonical example for the bottom row: prefer (alpha=0, dim_emb=1).
    canonical = next(
        (c for c in configs if c["alpha"] == 0.0 and c["dim_emb"] == 1),
        configs[0],
    )

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    (axa, axb), (axc, axd) = axes

    # Color by dim_emb (consistent across panels).
    color_map = {1: "#1f77b4", 2: "#d62728", 4: "#2ca02c"}
    marker_map = {1: "o", 2: "s", 4: "^"}

    # --- (a) |omega|-recovery Spearman ---
    for c in configs:
        axa.scatter(
            c["alpha"] if c["alpha"] > 0 else 1e-3,
            c["abs_rho"],
            s=90, color=color_map[c["dim_emb"]], marker=marker_map[c["dim_emb"]],
            edgecolor="black", lw=0.4, zorder=3,
        )
    axa.set_xscale("symlog", linthresh=1e-3)
    axa.set_xticks([0, 1e-2, 1e-1])
    axa.set_xticklabels(["0", "0.01", "0.1"])
    axa.set_xlim(-3e-4, 0.3)
    axa.set_ylim(0.96, 1.0)
    axa.set_xlabel(r"$\alpha$  (||delta|| penalty)")
    axa.set_ylabel(r"Spearman$(|PC_1(\mu_e)|, |\omega|)$")
    axa.set_title("(a) 1-D |omega| recovery", fontsize=10, loc="left")
    axa.axhline(0.99, color="gray", lw=0.5, ls=":")

    # --- (b) mean ||delta|| ---
    for c in configs:
        axb.scatter(
            c["alpha"] if c["alpha"] > 0 else 1e-3,
            c["delta_mean"],
            s=90, color=color_map[c["dim_emb"]], marker=marker_map[c["dim_emb"]],
            edgecolor="black", lw=0.4, zorder=3,
        )
    # reference: alpha=1 run from the previous experiment (collapsed)
    axb.scatter(1.0, 2.04e-4, s=90, color="gray", marker="x", lw=1.6,
                label=r"$\alpha=1$ ref (collapsed)")
    axb.set_xscale("symlog", linthresh=1e-3)
    axb.set_yscale("log")
    axb.set_xticks([0, 1e-2, 1e-1, 1])
    axb.set_xticklabels(["0", "0.01", "0.1", "1"])
    axb.set_xlim(-3e-4, 3)
    axb.set_xlabel(r"$\alpha$")
    axb.set_ylabel(r"mean $\|\Delta\|$ (across datasets, layers)")
    axb.set_title("(b) LoRA hypernet engagement", fontsize=10, loc="left")
    # legend: dim_emb markers + collapsed ref
    handles = [
        plt.Line2D([], [], marker=marker_map[d], color=color_map[d], lw=0,
                   markersize=8, markeredgecolor="black", markeredgewidth=0.4,
                   label=f"dim_emb={d}")
        for d in (1, 2, 4)
    ]
    handles.append(plt.Line2D([], [], marker="x", color="gray", lw=0,
                              markersize=8, markeredgewidth=1.6, label=r"$\alpha=1$ ref"))
    axa.legend(handles=handles[:3], frameon=False, fontsize=8, loc="lower left")
    axb.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")

    # --- (c) signed: mu_e vs omega (X-shape) ---
    om = torch.tensor(canonical["omegas"])
    p1 = pc1(canonical["mu_e"])
    axc.scatter(om, p1, c=om, cmap="viridis", s=22, edgecolor="black", lw=0.2)
    axc.set_xlabel(r"true $\omega$")
    axc.set_ylabel(r"$PC_1(\mu_e)$")
    axc.set_title(rf"(c) signed: rho={canonical['signed_rho']:+.2f}  "
                  rf"({canonical['name']})", fontsize=10, loc="left")

    # --- (d) folded: |mu_e| vs |omega| ---
    axd.scatter(om.abs(), p1.abs(), c=om.abs(), cmap="viridis", s=22,
                edgecolor="black", lw=0.2)
    axd.set_xlabel(r"$|\omega|$")
    axd.set_ylabel(r"$|PC_1(\mu_e)|$")
    axd.set_title(rf"(d) folded: rho={canonical['abs_rho']:+.3f}",
                  fontsize=10, loc="left")

    fig.suptitle(
        "Meta-dynamical SSM sweep: rotation-invariant 1-D recovery\n"
        "(N=100 limit-cycle datasets, 30 dB SNR, 2000 steps)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.sweep_dir / "sweep_summary.png"
    fig.savefig(out)
    plt.close(fig)

    # Also print table.
    print(f"{'config':>20} | {'alpha':>6} | {'dim':>3} | {'signed':>7} | {'|omega|':>7} | {'||d||_mean':>10}")
    for c in configs:
        print(f"{c['name']:>20} | {c['alpha']:>6.3f} | {c['dim_emb']:>3d} | "
              f"{c['signed_rho']:>+7.3f} | {c['abs_rho']:>+7.3f} | {c['delta_mean']:>10.3e}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
