"""Direct (no-Hydra) construction of MetaDynamicalSSM.

Mirrors examples/limit_cycle/configs/model/meta_ssm.yaml so tests do not
depend on Hydra or on the example config files.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_ssm import MetaDynamicalSSM  # noqa: E402
from meta_ssm.nn import (  # noqa: E402
    Adapters,
    EmbeddingEncoder,
    GaussianLikelihood,
    LatentDynamicsEncoderDKF,
    LoRAHypernet,
    MlpDynamics,
    PoissonLikelihood,
    ReadinNetwork,
    ReadinShared,
)


def build_model(
    observation_dims: dict[str, int],
    *,
    num_latents: int = 2,
    width: int = 64,
    hidden_layers: int = 2,
    dim_y_bar: int = 32,
    dim_shared: int = 16,
    dim_embedding: int = 1,
    rank: int = 1,
    adapt_layers: tuple[int, ...] = (0, 1),
    alpha: float = 1.0,
    concat_embedding: bool = True,
    common_init_condition: bool = True,
    linear_readin: bool = False,
    likelihood: str = "gaussian",
) -> MetaDynamicalSSM:
    adapt_set = set(adapt_layers)

    if likelihood == "gaussian":
        likelihood_modules = partial(GaussianLikelihood, num_latents=num_latents)
    elif likelihood == "poisson":
        likelihood_modules = partial(PoissonLikelihood, num_latents=num_latents)
    else:
        raise ValueError(f"unknown likelihood: {likelihood!r}")

    adapters = Adapters(
        dim_observations=observation_dims,
        readin_modules=partial(ReadinNetwork, dim_y_bar=dim_y_bar, linear=linear_readin),
        likelihood_modules=likelihood_modules,
    )

    return MetaDynamicalSSM(
        adapters=adapters,
        latent_encoder=LatentDynamicsEncoderDKF(
            num_latents=num_latents,
            dim_shared=dim_shared,
            dim_embedding=dim_embedding if concat_embedding else 0,
            dim_hidden=width,
        ),
        embedding_encoder=EmbeddingEncoder(
            dim_embedding=dim_embedding,
            dim_in=dim_shared,
            dim_hidden=width,
        ),
        hypernetwork=LoRAHypernet(
            dim_embedding=dim_embedding,
            num_latents=num_latents,
            width_dynamics=width,
            hidden_layers=hidden_layers,
            adapt_layers=adapt_set,
            rank=rank,
            width=width,
            common_init_condition=common_init_condition,
        ),
        dynamics=MlpDynamics(
            num_latents=num_latents,
            adapt_layers=adapt_set,
            hidden_layers=hidden_layers,
            width=width,
        ),
        shared_readin=ReadinShared(dim_y_bar=dim_y_bar, dim_shared=dim_shared),
        shared_readout=nn.Identity(),
        alpha=alpha,
        concat_embedding=concat_embedding,
        common_init_condition=common_init_condition,
    )
