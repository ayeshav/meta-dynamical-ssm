from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.motor_cortex.data import load_motor_cortex_data, sample_batch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    data = load_motor_cortex_data(
        co_path=cfg.data.co_path,
        maze_path=cfg.data.maze_path,
        decode_r2_threshold=cfg.data.decode_r2_threshold,
        n_folds=cfg.data.n_decode_folds,
        tasks=tuple(cfg.data.tasks),
        co_train_subjects=list(cfg.data.co_train_subjects),
        co_val_fraction=cfg.data.co_val_fraction,
        max_rate_drift=cfg.data.max_rate_drift,
        max_rate_cv=cfg.data.max_rate_cv,
        device=device,
    )

    n_total = len(data.decode_r2)
    n_train = len(data.train)
    n_val = len(data.val)
    print(f"Sessions: {n_train} train / {n_val} val (within-subject held-out) / "
          f"{n_total - n_train - n_val} excluded")
    for key in sorted(data.decode_r2):
        if key in data.train:
            mark = "train"
        elif key in data.val:
            mark = "val  "
        else:
            mark = "skip "
        stab = data.stability.get(key, {})
        drift_str = f"drift={stab.get('drift', float('nan')):.2f}"
        cv_str = f"cv={stab.get('cv', float('nan')):.2f}"
        print(f"  [{mark}] {key}  R²={data.decode_r2[key]:.3f}  {drift_str}  {cv_str}")

    adapters = instantiate(cfg.model.adapters, dim_observations=data.train_dims)
    model = instantiate(cfg.model, adapters=adapters).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(cfg.seed + 1)

    history = []
    for step in range(1, cfg.training.steps + 1):
        batch = sample_batch(
            data.train, cfg.training.batch_size,
            n_sessions=cfg.training.n_sessions_per_step,
            generator=batch_generator,
        )

        optimizer.zero_grad(set_to_none=True)
        out = model(batch, p_mask=cfg.training.p_mask)
        loss = out["loss"].mean()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append({"step": step, "loss": loss_value})
        if step == 1 or step % cfg.training.log_every == 0:
            print(f"step={step:04d}  loss={loss_value:.4f}")

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"Saved checkpoint → {output_dir / 'model.pt'}")
    print(f"Saved metrics    → {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
