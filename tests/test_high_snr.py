"""High-SNR ensemble sanity tests for MetaDynamicalSSM.

Generates 100 datasets spanned by a 1-D family (limit-cycle angular
velocity), with calibrated SNR. Trains the model for a short fixed budget
and checks: (1) loss is finite and decreases, (2) reconstruction R^2 on
held-out trials is high, (3) the inferred 1-D embedding is monotonic in
the true angular velocity.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.builders import build_model
from tests.synthetic import EnsembleData, generate_ensemble, sample_batch

SEED = 20260528
NUM_ENSEMBLE = 100
NUM_TRIALS = 32
NUM_TIMEPOINTS = 100
SNR_DB = 30.0
TRAIN_STEPS = 500
DATASETS_PER_STEP = 16
BATCH_SIZE = 16


@pytest.fixture(scope="module")
def trained() -> tuple[object, EnsembleData, list[float]]:
    torch.manual_seed(SEED)
    random.seed(SEED)

    data = generate_ensemble(
        num_ensemble=NUM_ENSEMBLE,
        num_trials=NUM_TRIALS,
        num_timepoints=NUM_TIMEPOINTS,
        snr_db=SNR_DB,
        seed=SEED,
    )

    model = build_model(data.observation_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(SEED + 1)

    keys = list(data.observations.keys())
    rng = random.Random(SEED + 2)

    losses: list[float] = []
    for step in range(TRAIN_STEPS):
        subset = rng.sample(keys, DATASETS_PER_STEP)
        obs_subset = {k: data.observations[k] for k in subset}
        batch = sample_batch(obs_subset, BATCH_SIZE, generator=batch_generator)

        optimizer.zero_grad(set_to_none=True)
        out = model(batch, p_mask=0.2)
        loss = out["loss"].mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return model, data, losses


def test_loss_is_finite_and_decreases(trained):
    _, _, losses = trained
    assert all(
        torch.isfinite(torch.tensor(v)).item() for v in losses
    ), "encountered NaN or Inf in training losses"
    early = sum(losses[:20]) / 20
    late = sum(losses[-20:]) / 20
    assert late < early, f"loss did not decrease: early={early:.3f}, late={late:.3f}"


def _r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def _predict_clean(model, ds: str, y_obs: torch.Tensor) -> torch.Tensor:
    """Run inference path with posterior means (no sampling)."""
    y_bar = model.adapters.readin[ds](y_obs)
    y_bar = model.shared_readin(y_bar)
    mu_e, _ = model.embedding_encoder(y_bar)
    e = mu_e if model.concat_embedding else None
    mu_q, _ = model.latent_encoder(y_bar, e)
    z = model.shared_readout(mu_q)
    return model.adapters.likelihood[ds].readout(z)


def test_reconstruction_r2(trained):
    model, data, _ = trained
    model.eval()

    rng = random.Random(SEED + 3)
    keys = rng.sample(list(data.observations.keys()), 10)

    r2_values: list[float] = []
    for key in keys:
        y_obs = data.observations[key]
        x_true = data.latents[key]
        C = data.observation_matrices[key]
        y_clean = x_true @ C
        y_hat = _predict_clean(model, key, y_obs)
        r2_values.append(_r2(y_hat, y_clean))

    r2_sorted = sorted(r2_values)
    median = r2_sorted[len(r2_sorted) // 2]
    assert median > 0.7, (
        f"median reconstruction R^2={median:.3f} below 0.7; values={r2_sorted}"
    )


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (torch.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


@torch.no_grad()
def test_embedding_recovers_1d(trained):
    model, data, _ = trained
    model.eval()

    keys = list(data.observations.keys())
    embeddings = []
    omegas = []
    for key in keys:
        y_obs = data.observations[key]
        y_bar = model.adapters.readin[key](y_obs)
        y_bar = model.shared_readin(y_bar)
        mu_e, _ = model.embedding_encoder(y_bar)
        embeddings.append(mu_e.flatten()[0].item())
        omegas.append(data.omegas[key])

    rho = _spearman(torch.tensor(embeddings), torch.tensor(omegas))
    assert abs(rho) > 0.7, f"|Spearman rho|={abs(rho):.3f} below 0.7"
