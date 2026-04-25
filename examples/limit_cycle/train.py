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

from examples.limit_cycle.data import generate_limit_cycle_data, sample_batch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    data = generate_limit_cycle_data(
        num_trials=cfg.data.num_trials,
        num_timepoints=cfg.data.num_timepoints,
        observation_dims=cfg.data.observation_dims,
        angular_velocities=cfg.data.angular_velocities,
        latent_dim=cfg.data.latent_dim,
        radius_scale=cfg.data.radius_scale,
        dt=cfg.data.dt,
        process_noise=cfg.data.process_noise,
        observation_noise=cfg.data.observation_noise,
        stride=cfg.data.stride,
        seed=cfg.seed,
        device=device,
    )

    adapters = instantiate(cfg.model.adapters, dim_observations=data.observation_dims)
    model = instantiate(cfg.model, adapters=adapters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(cfg.seed + 1)

    history = []
    for step in range(1, cfg.training.steps + 1):
        batch = sample_batch(
            data.observations,
            cfg.training.batch_size,
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
            print(f"step={step:04d} loss={loss_value:.4f}")

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"saved checkpoint to {output_dir / 'model.pt'}")
    print(f"saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
