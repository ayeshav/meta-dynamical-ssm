import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


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

        mu, log_variance = torch.split(out, [self.num_latents, -1], -1)
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

        mu, log_variance = torch.split(out, [self.num_latents, -1], -1)
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
        
        self.out = nn.Linear(2 * dim_hidden, 2 * dim_embedding)

    def forward(self, y):
        
        h, _ = self.net(y)
        out = self.out(h)

        mu, log_variance = torch.split(out, [self.dim_embedding, -1], -1)
        variance = F.softplus(log_variance) + EPS

        if self.pool:
            mu = mu.mean(0)
            variance = variance.mean(0)
        
        return mu, variance







