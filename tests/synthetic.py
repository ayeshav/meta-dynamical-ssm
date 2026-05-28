"""Ensemble synthetic data generator with calibrated SNR.

Produces N_ensemble datasets spanned by a 1-D family of dynamical systems
(limit cycle, varying angular velocity). Each dataset has its own
observation dimensionality and additive Gaussian observation noise scaled
to a target signal-to-noise ratio.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.limit_cycle.data import LimitCycle, sample_batch  # noqa: E402,F401


@dataclass
class EnsembleData:
    observations: dict[str, torch.Tensor]
    latents: dict[str, torch.Tensor]
    observation_matrices: dict[str, torch.Tensor]
    omegas: dict[str, float]
    observation_dims: dict[str, int]
    snr_db: float


def generate_ensemble(
    *,
    num_ensemble: int = 100,
    num_trials: int = 64,
    num_timepoints: int = 100,
    omega_range: tuple[float, float] = (-5.0, 5.0),
    obs_dim_range: tuple[int, int] = (20, 50),
    latent_dim: int = 2,
    radius_scale: float = 2.0,
    dt: float = 1e-2,
    process_noise: float = 0.5,
    snr_db: float = 20.0,
    stride: int = 3,
    seed: int = 20260528,
    device: str | torch.device = "cpu",
) -> EnsembleData:
    if latent_dim != 2:
        raise ValueError("LimitCycle requires latent_dim=2.")

    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # 1-D family parameter, skipping exactly zero to avoid a fixed point at origin.
    omegas_t = torch.linspace(omega_range[0], omega_range[1], num_ensemble)
    omegas_t = torch.where(
        omegas_t.abs() < 1e-6,
        torch.full_like(omegas_t, (omega_range[1] - omega_range[0]) / (2 * num_ensemble)),
        omegas_t,
    )

    obs_dims_t = torch.randint(
        obs_dim_range[0],
        obs_dim_range[1] + 1,
        (num_ensemble,),
        generator=generator,
        device=device,
    )

    target_linear = 10.0 ** (snr_db / 10.0)

    observations: dict[str, torch.Tensor] = {}
    latents: dict[str, torch.Tensor] = {}
    observation_matrices: dict[str, torch.Tensor] = {}
    omegas: dict[str, float] = {}
    observation_dims: dict[str, int] = {}

    for i in range(num_ensemble):
        key = str(i)
        omega = float(omegas_t[i].item())
        obs_dim = int(obs_dims_t[i].item())

        dyn = LimitCycle(radius_scale=radius_scale, angular_velocity=omega, dt=dt)
        x0 = torch.randn(num_trials, latent_dim, generator=generator, device=device)
        x = dyn.trajectories(
            x0,
            num_timepoints,
            process_noise=process_noise,
            generator=generator,
        )
        x = x[:, ::stride].float()

        C = torch.randn(latent_dim, obs_dim, generator=generator, device=device)
        C = C / latent_dim**0.5

        y_clean = x @ C
        signal_power = float(torch.mean(y_clean**2).item())
        # SNR_linear = signal_power / sigma^2, so sigma = sqrt(signal_power / SNR_linear).
        sigma = (signal_power / target_linear) ** 0.5

        noise = torch.randn(y_clean.shape, generator=generator, device=device)
        y = y_clean + sigma * noise

        observations[key] = y.float()
        latents[key] = x.float()
        observation_matrices[key] = C.float()
        omegas[key] = omega
        observation_dims[key] = obs_dim

    return EnsembleData(
        observations=observations,
        latents=latents,
        observation_matrices=observation_matrices,
        omegas=omegas,
        observation_dims=observation_dims,
        snr_db=snr_db,
    )
