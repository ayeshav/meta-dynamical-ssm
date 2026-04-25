from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class LimitCycleData:
    latents: dict[str, torch.Tensor]
    observations: dict[str, torch.Tensor]
    observation_matrices: dict[str, torch.Tensor]

    @property
    def observation_dims(self) -> dict[str, int]:
        return {key: y.shape[-1] for key, y in self.observations.items()}


class BaseDynamics:
    def __init__(self, dt: float = 1e-2):
        self.dt = dt

    def step(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def trajectories(
        self,
        x0: torch.Tensor,
        num_steps: int,
        process_noise: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = torch.empty(
            x0.shape[0],
            num_steps,
            x0.shape[1],
            dtype=x0.dtype,
            device=x0.device,
        )
        x[:, 0] = x0

        for t in range(num_steps - 1):
            noise = torch.randn(
                x0.shape,
                generator=generator,
                dtype=x0.dtype,
                device=x0.device,
            )
            noise = self.dt * process_noise * noise
            x[:, t + 1] = self.step(x[:, t]) + noise

        return x


class LimitCycle(BaseDynamics):
    def __init__(self, radius_scale: float, angular_velocity: float, dt: float = 1e-2):
        super().__init__(dt=dt)
        self.radius_scale = radius_scale
        self.angular_velocity = angular_velocity

    def step(self, x: torch.Tensor) -> torch.Tensor:
        radius = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        theta = torch.atan2(x[:, 1], x[:, 0])

        radius = radius + radius * (self.radius_scale - radius**2) * self.dt
        theta = theta + self.angular_velocity * self.dt

        return torch.stack(
            [radius * torch.cos(theta), radius * torch.sin(theta)],
            dim=-1,
        )


def generate_limit_cycle_data(
    *,
    num_trials: int = 128,
    num_timepoints: int = 300,
    observation_dims: Sequence[int] = (20, 30, 40, 50),
    angular_velocities: Sequence[float] = (-5.0, -2.5, 2.5, 5.0),
    latent_dim: int = 2,
    radius_scale: float = 2.0,
    dt: float = 1e-2,
    process_noise: float = 0.5,
    observation_noise: float = 0.1,
    stride: int = 3,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> LimitCycleData:
    if latent_dim != 2:
        raise ValueError("The limit-cycle generator currently supports latent_dim=2.")
    if len(observation_dims) != len(angular_velocities):
        raise ValueError("observation_dims and angular_velocities must have the same length.")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    latents = {}
    observations = {}
    observation_matrices = {}

    for dataset_idx, (obs_dim, angular_velocity) in enumerate(
        zip(observation_dims, angular_velocities)
    ):
        key = str(dataset_idx)
        dynamics = LimitCycle(radius_scale, angular_velocity, dt=dt)
        x0 = torch.randn(num_trials, latent_dim, generator=generator, device=device)
        x = dynamics.trajectories(
            x0,
            num_timepoints,
            process_noise=process_noise,
            generator=generator,
        )
        x = x[:, ::stride].float()

        C = torch.randn(latent_dim, obs_dim, generator=generator, device=device)
        C = C / latent_dim**0.5
        y = x @ C
        y = y + observation_noise * torch.randn(
            y.shape,
            generator=generator,
            device=device,
        )

        latents[key] = x
        observations[key] = y.float()
        observation_matrices[key] = C.float()

    return LimitCycleData(
        latents=latents,
        observations=observations,
        observation_matrices=observation_matrices,
    )


def sample_batch(
    observations: dict[str, torch.Tensor],
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    batch = {}
    for key, y in observations.items():
        indices = torch.randint(
            y.shape[0],
            (batch_size,),
            generator=generator,
            device=y.device,
        )
        batch[key] = {"y": y[indices]}
    return batch
