from dataclasses import dataclass
from typing import List, Tuple, Set

import torch.nn as nn

from nn import (ReadinNetwork, 
                MlpDynamics, 
                LoRAHypernet, 
                GaussianLikelihood,
                LatentDynamicsEncoderDKF,
                EmbeddingEncoder)
from model import MetaDynamicalSSM


@dataclass
class ModelCfg:
    num_latents: int
    dim_observations: List[int]
    dim_shared: int
    dim_embedding: int
    rank: int
    layer_dims: List[Tuple[int,int]]
    hidden_layers: int = 2
    adapt_layers: Set[int] = None
    width: int = 128
    dropout: float = 0.0
    concat_embedding: bool = True
    linear_readin: bool = False


def build_meta_ssm(cfg: ModelCfg) -> MetaDynamicalSSM:
    adapt_layers = {0, 1} if cfg.adapt_layers is None else set(cfg.adapt_layers)

    readin_net = nn.ModuleList([
        ReadinNetwork(num_observations=d, dim_hidden=cfg.width, dim_out=cfg.dim_shared,
                      linear=cfg.linear_readin, dropout=cfg.dropout)
        for d in cfg.dim_observations
    ])

    dim_encoder_in = cfg.dim_shared + cfg.dim_embedding if cfg.concat_embedding else cfg.dim_shared

    latent_encoder = LatentDynamicsEncoderDKF(
        num_latents=cfg.num_latents,
        dim_in=dim_encoder_in,
        dim_hidden=cfg.width
    )

    embedding_encoder = EmbeddingEncoder(
        dim_embedding=cfg.dim_embedding,
        dim_in=cfg.dim_shared,
        dim_hidden=cfg.width
    )

    hypernetwork = LoRAHypernet(
        dim_embedding=cfg.dim_embedding,
        layer_dims=cfg.layer_dims,
        adapt_layers=adapt_layers,
        rank=cfg.rank,
        width=cfg.width
    )

    dynamics = MlpDynamics(
        num_latents=cfg.num_latents,
        hidden_layers=cfg.hidden_layers,
        dim=cfg.width,
        adapt_layers=adapt_layers
    )

    likelihood = nn.ModuleList([
        GaussianLikelihood(num_latents=cfg.num_latents, num_observations=d)
        for d in cfg.dim_observations
    ])

    return MetaDynamicalSSM(
        readin_net=readin_net,
        latent_encoder=latent_encoder,
        embedding_encoder=embedding_encoder,
        hypernetwork=hypernetwork,
        dynamics=dynamics,
        likelihood=likelihood,
    )
