import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class ReadinNetwork(nn.Module):
    def __init__(
            self, 
            dim_observation: int, 
            width: int = 128,
            dim_shared: int = 64,
            linear: bool = False,
            dropout: float = 0.0
    ):
        super().__init__()

        if linear:
            self.net = nn.Linear(dim_observation, dim_shared)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim_observation, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_shared)
                )

    def forward(self, y):
        return self.net(y)


class LatentDynamicsEncoderDKF(nn.Module):
    def __init__(
            self,
            num_latents: int,
            dim_in: int, 
            dim_hidden: int = 128,

    ):
        super().__init__()

        self.num_latents = num_latents

        self.net = nn.GRU(input_size=dim_in,
                          hidden_size=dim_hidden,
                          bidirectional=True,
                          batch_first=True)
        
        self.out = nn.Linear(2 * dim_hidden, 2 * num_latents)

    def forward(self, y, e=None):

        y_cat = y
        if e is not None:
            e = e.expand(y.size(0), y.size(1), -1)
            y_cat = torch.cat([y, e], -1)

        h, _ = self.net(y_cat)
        out = self.out(h)

        mu, log_variance = out.chunk(2, dim=-1)
        variance = F.softplus(log_variance) + EPS

        return mu, variance


class LatentDynamicsEncoderDVBF(nn.Module):
    def __init__(
            self,
            num_latents: int,
            dim_in: int, 
            dim_hidden: int = 128,

    ):
        super().__init__()

        self.num_latents = num_latents

        self.net = nn.GRU(input_size=dim_in,
                          hidden_size=dim_hidden,
                          batch_first=True)
        
        self.out = nn.Linear(2 * dim_hidden, 2 * num_latents)

    def forward(self, y, e=None):

        y_cat = y
        if e is not None:
            e = e.expand(y.size(0), y.size(1), -1)
            y_cat = torch.cat([y, e], -1)

        h, _ = self.net(torch.flip(y_cat, dims=[1]))

        out = self.out(torch.flip(h, dims=[1]))

        mu, log_variance = out.chunk(2, dim=-1)
        variance = F.softplus(log_variance) + EPS

        return mu, variance


class EmbeddingEncoder(nn.Module):
    def __init__(
            self,
            dim_embedding: int,
            dim_in: int,
            t_max: int,
            randomize: bool = True,
            dim_hidden: int = 128,
            pool: bool = True,
    ):
        super().__init__()

        self.dim_embedding = dim_embedding
        self.pool = pool

        self.net = nn.GRU(input_size=dim_in, hidden_size=dim_hidden, batch_first=True)
        
        self.out = nn.Linear(dim_hidden, 2 * dim_embedding)

    def forward(self, y):
        
        h, _ = self.net(y)
        out = self.out(h[:, -1].unsqueeze(1))

        mu, log_variance = out.chunk(2, dim=-1)
        variance = F.softplus(log_variance) + EPS

        if self.pool:
            mu = mu.mean(0)
            variance = variance.mean(0)
        
        return mu, variance







