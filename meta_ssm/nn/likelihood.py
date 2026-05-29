import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Poisson

EPS = 1e-6


class ReadoutShared(nn.Module):
    def __init__(self, dim_y_bar: int, dim_shared: int, width: int = 128, dropout: float = 0.0, affine_ln: bool = True):
        super().__init__()
        self.net = nn.Linear(dim_y_bar, dim_shared)
        self.ln = nn.LayerNorm(dim_shared, elementwise_affine=affine_ln)

        nn.init.orthogonal_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, h):     # h: [B,T,dim_readin]
        return self.net(h)


class Likelihood(nn.Module):
    def __init__(
            self,
            num_latents : int,
            num_observations: int,
            linear: bool = True,
            dim_hidden: int = 128
    ):
        super().__init__()

        if linear:
            self.readout = nn.Linear(num_latents, num_observations)
        else:
            self.readout = nn.Sequential(
                nn.Linear(num_latents, dim_hidden),
                nn.SiLU(),
                nn.Linear(dim_hidden, num_observations)
            )

    def get_mean_output(self, z):
        return self.readout(z)

    def forward(self, z, y):
        pass


class GaussianLikelihood(Likelihood):
    def __init__(
            self,
            num_latents : int,
            num_observations: int,
            linear: bool = True,
            dim_hidden: int = 128
    ):
        super().__init__(num_latents, num_observations, linear, dim_hidden)

        self.log_variance = nn.Parameter(torch.ones(1, num_observations))

    def forward(self, z, y):
        mu = self.get_mean_output(z)
        variance = F.softplus(self.log_variance) + EPS

        log_likelihood = Normal(mu, torch.sqrt(variance)).log_prob(y)

        return torch.sum(log_likelihood, (-1, -2))

    
class PoissonLikelihood(Likelihood):
    """Log-linear Poisson: lambda = exp(C z + b), matching the generative model.

    `self.readout` is `nn.Linear(num_latents, num_observations)`, which
    learns both the loading matrix C and the per-neuron bias b. The
    log-rate is clamped before exp for numerical stability.
    """

    def __init__(
            self,
            num_latents : int,
            num_observations: int,
            linear: bool = True,
            dim_hidden: int = 128,
            log_rate_clamp: float = 8.0,
    ):
        super().__init__(num_latents, num_observations, linear, dim_hidden)
        self.log_rate_clamp = log_rate_clamp

    def forward(self, z, y):
        log_rates = self.get_mean_output(z).clamp(max=self.log_rate_clamp)
        rates = torch.exp(log_rates) + EPS

        log_likelihood = Poisson(rates).log_prob(y)

        return torch.sum(log_likelihood, (-1, -2))




        

