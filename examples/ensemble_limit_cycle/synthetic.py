"""Ensemble synthetic data generator with calibrated SNR.

Produces N_ensemble datasets spanned by a 1-D family of dynamical systems
(limit cycle, varying angular velocity). Each dataset has its own
observation dimensionality and either (a) additive Gaussian observation
noise scaled to a target SNR, or (b) log-linear Poisson spike counts with
loading matrix calibrated to a target SNR per the Fisher-information
definition from neurofisherSNR (arXiv:2408.08752).
"""
from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _load_neurofisher():
    """Import neurofisherSNR from the submodule at external/."""
    repo_root = Path(__file__).resolve().parents[2]
    lib_path = repo_root / "external" / "neurofisherSNR"
    if str(lib_path) not in sys.path:
        sys.path.insert(0, str(lib_path))
    import neurofisherSNR  # noqa: F401
    from neurofisherSNR.observation import gen_poisson_observations  # noqa: E402
    return gen_poisson_observations


def generate_ensemble_poisson(
    *,
    num_ensemble: int = 100,
    num_trials: int = 32,
    num_timepoints: int = 100,
    omega_range: tuple[float, float] = (-5.0, 5.0),
    n_neurons_range: tuple[int, int] = (500, 1500),
    latent_dim: int = 2,
    radius_scale: float = 2.0,
    dt: float = 1e-2,
    process_noise: float = 0.5,
    target_mean_rate: float = 0.05,
    target_max_rate: float = 0.5,
    target_snr_db: float = 20.0,
    priority: str = "max",
    p_coh: float = 0.95,
    p_sparse: float = 0.1,
    stride: int = 3,
    seed: int = 20260528,
    device: str | torch.device = "cpu",
) -> EnsembleData:
    """Per-dataset log-linear Poisson observations of a limit-cycle ensemble.

    Uses `neurofisherSNR.observation.gen_poisson_observations` (Jeon &
    Park, EUSIPCO 2024) for SNR-matched loading-matrix construction. The
    library expects 2-D numpy (T, d_latent) inputs and returns
    (T, n_neurons) observations and rates; we concatenate across trials
    per dataset, call the lib, then reshape back to (B, T, n_neurons).
    """
    if latent_dim != 2:
        raise ValueError("LimitCycle requires latent_dim=2.")

    import numpy as np  # neurofisherSNR is numpy-based
    gen_poisson_observations = _load_neurofisher()

    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    omegas_t = torch.linspace(omega_range[0], omega_range[1], num_ensemble)
    omegas_t = torch.where(
        omegas_t.abs() < 1e-6,
        torch.full_like(omegas_t, (omega_range[1] - omega_range[0]) / (2 * num_ensemble)),
        omegas_t,
    )

    n_neurons_t = torch.randint(
        n_neurons_range[0],
        n_neurons_range[1] + 1,
        (num_ensemble,),
        generator=generator,
        device=device,
    )

    observations: dict[str, torch.Tensor] = {}
    latents: dict[str, torch.Tensor] = {}
    observation_matrices: dict[str, torch.Tensor] = {}
    biases: dict[str, torch.Tensor] = {}
    rates_map: dict[str, torch.Tensor] = {}
    realized_snrs: dict[str, float] = {}
    omegas: dict[str, float] = {}
    observation_dims: dict[str, int] = {}

    # neurofisherSNR uses np.random under the hood; seed it deterministically.
    rng_np = np.random.RandomState(seed)
    np_seed_state = rng_np.randint(0, 2**31 - 1, size=num_ensemble)

    for i in range(num_ensemble):
        key = str(i)
        omega = float(omegas_t[i].item())
        n_neurons = int(n_neurons_t[i].item())

        dyn = LimitCycle(radius_scale=radius_scale, angular_velocity=omega, dt=dt)
        x0 = torch.randn(num_trials, latent_dim, generator=generator, device=device)
        x = dyn.trajectories(
            x0, num_timepoints, process_noise=process_noise, generator=generator,
        )
        x = x[:, ::stride].float()           # [B, T_eff, d_latent]
        B, T_eff, _ = x.shape

        # Flatten (B, T) -> rows for the library; per-dataset normalize to
        # zero mean / unit variance per latent dim before passing in.
        x_flat = x.reshape(-1, latent_dim).cpu().numpy()
        x_flat = x_flat - x_flat.mean(axis=0, keepdims=True)
        x_flat = x_flat / (x_flat.std(axis=0, keepdims=True) + 1e-8)

        # Subsample timepoints for SNR fitting -- the library iterates
        # an O(n_neurons^2) python loop over every input row. We only
        # need a representative sample of the latent distribution.
        n_calib = min(60, x_flat.shape[0])
        idx = np.linspace(0, x_flat.shape[0] - 1, n_calib).round().astype(int)
        x_calib = x_flat[idx]

        # Pre-seed a sparse Gaussian C and pass it to
        # gen_poisson_observations to skip the lib's initialize_C, which
        # runs a 15000-iter coherence optimization that does not converge
        # for d_latent=2 (in 2-D, max coherence over many vectors is 1).
        np.random.seed(int(np_seed_state[i]))
        C_init = np.random.randn(n_neurons, latent_dim)
        mask = np.random.rand(n_neurons, latent_dim) > p_sparse
        C_init = C_init * mask
        C_init = C_init / (np.linalg.norm(C_init, axis=1, keepdims=True) + 1e-8)

        with contextlib.redirect_stdout(io.StringIO()):
            _obs_cal, C_np, b_np, _rates_cal, snr_realized = gen_poisson_observations(
                x=x_calib,
                C=C_init,
                d_neurons=n_neurons,
                tgt_rate_per_bin=target_mean_rate,
                max_rate_per_bin=target_max_rate,
                priority=priority,
                p_coh=p_coh,
                p_sparse=p_sparse,
                tgt_snr=target_snr_db,
            )
        # Apply the calibrated (C, b) to ALL trial-time rows to get the
        # full per-bin observations.
        rates_full = np.exp(x_flat @ C_np.T + b_np)
        obs_full = np.random.poisson(rates_full).astype("float32")

        y = torch.from_numpy(obs_full).to(device).reshape(B, T_eff, n_neurons)
        rates = torch.from_numpy(rates_full.astype("float32")).to(device).reshape(B, T_eff, n_neurons)
        z_norm = torch.from_numpy(x_flat.astype("float32")).to(device).reshape(B, T_eff, latent_dim)
        C = torch.from_numpy(C_np.astype("float32")).to(device)        # (n_neurons, d_latent)
        b = torch.from_numpy(b_np.astype("float32")).to(device)        # (1, n_neurons)

        observations[key] = y
        latents[key] = z_norm
        observation_matrices[key] = C
        biases[key] = b
        rates_map[key] = rates
        realized_snrs[key] = float(snr_realized)
        omegas[key] = omega
        observation_dims[key] = n_neurons

    out = EnsembleData(
        observations=observations,
        latents=latents,
        observation_matrices=observation_matrices,
        omegas=omegas,
        observation_dims=observation_dims,
        snr_db=target_snr_db,
    )
    out.biases = biases  # type: ignore[attr-defined]
    out.rates = rates_map  # type: ignore[attr-defined]
    out.realized_snrs = realized_snrs  # type: ignore[attr-defined]
    return out
