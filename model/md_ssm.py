import torch
import torch.nn as nn

from nn import (ReadinNetwork,
                MlpDynamics,
                LoRAHypernet,
                GaussianLikelihood,
                LatentDynamicsEncoderDKF,
                EmbeddingEncoder)

from utils import *


class MetaDynamicalSSM(nn.Module):
    def __init__(
            self,
            num_latents: int,
            dim_observations: list[int],
            dim_shared: int,
            dim_embedding: int,
            rank: int,
            hidden_layers: int = 2,
            adapt_layers: set[int] = {0, 1},
            width: int = 128,
            dropout: float = 0.0,
            concat_embedding: bool = True,
            linear_readin: bool = False
    ):
        super().__init__()

        self.readin_net = nn.ModuleList(
            ReadinNetwork(num_observations=dim_observations, dim_hidden=width,
                          dim_out=dim_shared, linear=linear_readin, dropout=dropout)
        )
        
        dim_encoder_in = dim_shared + dim_embedding if concat_embedding else dim_shared
        
        self.latent_encoder = LatentDynamicsEncoderDKF(
            num_latents=num_latents, 
            dim_in=dim_encoder_in,
            dim_hidden=width
            )
        
        self.embedding_encoder = EmbeddingEncoder(
            dim_embedding=dim_embedding,
            dim_in=dim_shared,
            dim_hidden=width
        )

        self.hypernetwork = 

        self.dynamics = MlpDynamics(
            num_latents=num_latents,
            hidden_layers=hidden_layers,
            dim=width,
            adapt_layers=adapt_layers
        )



        