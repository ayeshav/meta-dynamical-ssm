"""Unit tests for the PoissonLikelihood PCA/lstsq warm-start (PR #7).

Run from the repo root:

    uv run --with torch --with pytest pytest tests/test_poisson_init.py
"""
import warnings
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from meta_ssm.nn.adapter import Adapters
from meta_ssm.nn.likelihood import (
    EPS,
    GaussianLikelihood,
    PoissonLikelihood,
    _pca_lstsq_loading,
)

SEED = 20260608  # date-based seed, per project convention
B, T, N, DZ = 8, 40, 12, 3


def _poisson_data(num_neurons=N, latent_dim=DZ, seed=SEED):
    """Log-linear Poisson counts, matching the generative model (rates = exp(Cz+b))."""
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(num_neurons, latent_dim, generator=g)
    b = torch.randn(num_neurons, generator=g) - 1.0
    z = torch.randn(B, T, latent_dim, generator=g)
    rates = torch.exp(z @ C.T + b)
    y = torch.poisson(rates, generator=g)
    return y, z, C, b


def test_pca_lstsq_loading_shapes():
    y, _, _, _ = _poisson_data()
    C, b = _pca_lstsq_loading(y, latent_dim=DZ)
    assert C.shape == (N, DZ)
    assert b.shape == (N,)
    assert torch.isfinite(C).all() and torch.isfinite(b).all()


def test_normalize_preserves_bias():
    """normalize rescales C by the latent std but leaves the fitted bias unchanged."""
    y, _, _, _ = _poisson_data()
    C_n, b_n = _pca_lstsq_loading(y, latent_dim=DZ, normalize=True)
    C_u, b_u = _pca_lstsq_loading(y, latent_dim=DZ, normalize=False)
    assert torch.allclose(b_n, b_u, atol=1e-4)


def test_init_from_data_sets_readout_and_flag():
    y, _, _, _ = _poisson_data()
    lik = PoissonLikelihood(num_latents=DZ, num_observations=N)
    assert lik._initialized_from_data is False
    lik.init_from_data(y)
    assert lik._initialized_from_data is True
    assert lik.readout.weight.shape == (N, DZ)
    assert lik.readout.bias.shape == (N,)


def test_default_link_is_exp_and_byte_identical():
    """Default link must reproduce the original exp formula exactly."""
    y, z, _, _ = _poisson_data()
    lik = PoissonLikelihood(num_latents=DZ, num_observations=N)
    assert lik.link == "exp"
    raw = lik.get_mean_output(z)
    expected = torch.exp(raw.clamp(max=lik.log_rate_clamp)) + EPS
    assert torch.equal(lik.mean_rate(raw), expected)


def test_softplus_link_mean_rate():
    y, z, _, _ = _poisson_data()
    lik = PoissonLikelihood(num_latents=DZ, num_observations=N, link="softplus")
    assert lik.link == "softplus"
    raw = lik.get_mean_output(z)
    assert torch.equal(lik.mean_rate(raw), F.softplus(raw) + EPS)


def test_invalid_link_raises():
    with pytest.raises(ValueError, match="link must be"):
        PoissonLikelihood(num_latents=DZ, num_observations=N, link="relu")


def test_forward_finite_for_both_links():
    y, z, _, _ = _poisson_data()
    for link in ("exp", "softplus"):
        lik = PoissonLikelihood(num_latents=DZ, num_observations=N, link=link)
        lik.init_from_data(y)
        out = lik(z, y)
        assert torch.isfinite(out).all(), link


def test_forward_after_init_no_warning_finite_loss():
    y, z, _, _ = _poisson_data()
    lik = PoissonLikelihood(num_latents=DZ, num_observations=N)
    lik.init_from_data(y)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = lik(z, y)
    pois = [x for x in w if "PoissonLikelihood" in str(x.message)]
    assert len(pois) == 0
    assert torch.isfinite(out).all()


def test_forward_without_init_warns_once():
    y, z, _, _ = _poisson_data()
    lik = PoissonLikelihood(num_latents=DZ, num_observations=N)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lik(z, y)
        lik(z, y)  # second call must NOT re-warn
    pois = [x for x in w if "PoissonLikelihood" in str(x.message)]
    assert len(pois) == 1


def test_nonlinear_readout_raises_not_implemented():
    y, _, _, _ = _poisson_data()
    lik = PoissonLikelihood(num_latents=DZ, num_observations=N, linear=False)
    with pytest.raises(NotImplementedError):
        lik.init_from_data(y)


def test_latent_dim_exceeds_rank_raises_value_error():
    """Graceful error when the readout asks for more latents than the data supports."""
    y, _, _, _ = _poisson_data(num_neurons=2)  # N=2 < DZ=3
    with pytest.raises(ValueError, match="exceeds the number of identifiable"):
        _pca_lstsq_loading(y, latent_dim=DZ)


def _adapters(keys_dims):
    return Adapters(
        keys_dims,
        readin_modules=partial(_make_readin, dim_shared=5),
        likelihood_modules=partial(PoissonLikelihood, num_latents=DZ),
    )


def _make_readin(num_observations, dim_shared):
    return torch.nn.Linear(num_observations, dim_shared)


def test_adapters_init_all_poisson():
    y0, _, _, _ = _poisson_data(num_neurons=N, seed=SEED)
    y1, _, _, _ = _poisson_data(num_neurons=N + 2, seed=SEED + 1)
    ad = _adapters({"0": N, "1": N + 2})
    n = ad.init_likelihoods_from_data({"0": y0, "1": y1})
    assert n == 2
    assert ad.likelihood["0"]._initialized_from_data
    assert ad.likelihood["1"]._initialized_from_data


def test_adapters_skips_non_poisson():
    y0, _, _, _ = _poisson_data(num_neurons=N, seed=SEED)
    y1, _, _, _ = _poisson_data(num_neurons=N + 2, seed=SEED + 1)
    ad = _adapters({"0": N, "1": N + 2})
    ad.likelihood["1"] = GaussianLikelihood(num_latents=DZ, num_observations=N + 2)
    n = ad.init_likelihoods_from_data({"0": y0, "1": y1})
    assert n == 1  # only the Poisson readout is warm-started


def test_adapters_missing_key_raises_key_error():
    y0, _, _, _ = _poisson_data(num_neurons=N, seed=SEED)
    ad = _adapters({"0": N, "1": N + 2})
    with pytest.raises(KeyError):
        ad.init_likelihoods_from_data({"0": y0})  # "1" missing
