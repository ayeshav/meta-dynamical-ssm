import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Poisson

EPS = 1e-6


def _gaussian_smooth_time(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Per-trial, per-neuron 1-D Gaussian smoothing along the time axis.

    `y` has shape `[B, T, N]`. Returns the same shape.
    """
    if sigma <= 0:
        return y
    radius = max(1, int(round(3 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=y.dtype, device=y.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    B, T, N = y.shape
    y_perm = y.permute(0, 2, 1).reshape(B * N, 1, T)
    y_sm = F.conv1d(y_perm, kernel.view(1, 1, -1), padding=radius)
    return y_sm.reshape(B, N, T).permute(0, 2, 1)


def _pca_lstsq_loading(
    y: torch.Tensor,
    latent_dim: int,
    sigma_smooth: float = 2.0,
    log_floor: float = 1e-3,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PFA / PLDS-style initialization of a log-linear Poisson loading + bias.

    Given spike counts `y` of shape `[B, T, N]`, returns `(C, b)` with
    shapes `[N, latent_dim]` and `[N]` such that

        log lambda(t) ~= C @ z(t) + b

    where `z(t)` is the first `latent_dim` principal components of
    `log(rate_hat)` and `rate_hat` is `y` smoothed in time. No oracle on
    the true latent.

    If `normalize=True`, rescale `(z, C)` so that the per-dim std of the
    warm-started latent is 1 (matches a zero-mean unit-variance
    convention; preserves the fitted log-rate exactly).

    Caveat: PCA components are sign-ambiguous. For dynamics with a
    sign-equivariance (e.g., omega <-> -omega on a limit cycle) this is
    harmless. For asymmetric dynamics, per-dataset sign flips would
    propagate to the shared dynamics and gradient descent cannot bridge
    the discontinuity. In that regime a sign-disambiguation step is
    required.

    References:
        Macke, J. H., Buesing, L., Cunningham, J. P., Yu, B. M.,
            Shenoy, K. V., & Sahani, M. (2011). Empirical models of
            spiking in neural populations. NeurIPS.
        Yu, B. M., Cunningham, J. P., Santhanam, G., Ryu, S. I.,
            Shenoy, K. V., & Sahani, M. (2009). Gaussian-process factor
            analysis for low-dimensional single-trial analysis of
            neural population activity. J. Neurophysiol.
    """
    y_smooth = _gaussian_smooth_time(y, sigma_smooth)
    log_rate = torch.log(y_smooth + log_floor)
    log_rate_flat = log_rate.reshape(-1, log_rate.shape[-1])
    L_centered = log_rate_flat - log_rate_flat.mean(0, keepdim=True)
    U, S, _Vh = torch.linalg.svd(L_centered, full_matrices=False)
    z_hat = U[:, :latent_dim] * S[:latent_dim]
    # Augment with a column of ones so lstsq picks up the bias.
    z_aug = torch.cat(
        [z_hat, torch.ones(z_hat.shape[0], 1, device=z_hat.device, dtype=z_hat.dtype)],
        dim=1,
    )
    sol = torch.linalg.lstsq(z_aug, log_rate_flat).solution
    C = sol[:latent_dim, :].T  # [N, latent_dim]
    b = sol[latent_dim, :]      # [N]
    if normalize:
        sigma = z_hat.std(dim=0, unbiased=False).clamp_min(1e-8)
        C = C * sigma
    return C, b


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

    Joint Poisson + latent-VAE optimization from a random readout init
    is prone to a "predict mean rate via bias alone" collapse. Calling
    `init_from_data(y)` once, before training, with the dataset's spike
    counts fixes this. See
    `examples/ensemble_limit_cycle/LESSONS_LEARNED.md` for the diagnosis
    journey that led to this default.
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
        # Tracks whether the user warm-started the readout from data.
        # Used only to emit a one-time UserWarning on the first forward
        # pass when this has not been done.
        self._initialized_from_data: bool = False
        self._warned_uninitialized: bool = False

    def init_from_data(
        self,
        y: torch.Tensor,
        *,
        sigma_smooth: float = 2.0,
        log_floor: float = 1e-3,
        normalize: bool = True,
    ) -> None:
        """Warm-start `(self.readout.weight, self.readout.bias)` from spike data.

        Implements the standard PFA / PLDS / GPFA-style initialization
        (Macke et al. 2011; Yu et al. 2009). See
        `_pca_lstsq_loading` for the math and references.

        `y` is the spike-count tensor for this dataset, shape `[B, T, N]`.
        Only supported when `self.readout` is a single `nn.Linear` (the
        `linear=True` mode used by every standard setup).
        """
        if not isinstance(self.readout, nn.Linear):
            raise NotImplementedError(
                "init_from_data only supports linear readouts "
                f"(got {type(self.readout).__name__}). "
                "For nonlinear readouts the analog would be to fit only "
                "the final linear layer; not yet implemented."
            )
        latent_dim = self.readout.weight.shape[1]
        C, b = _pca_lstsq_loading(
            y, latent_dim=latent_dim, sigma_smooth=sigma_smooth,
            log_floor=log_floor, normalize=normalize,
        )
        with torch.no_grad():
            self.readout.weight.copy_(C)
            self.readout.bias.copy_(b)
        self._initialized_from_data = True

    def forward(self, z, y):
        if not self._initialized_from_data and not self._warned_uninitialized:
            warnings.warn(
                "PoissonLikelihood is being used without a data-driven "
                "warm-start. Joint Poisson + latent-VAE optimization "
                "from random init often collapses to a 'predict mean "
                "rate via bias alone' solution. Call "
                "`PoissonLikelihood.init_from_data(y)` before training, "
                "or use `Adapters.init_likelihoods_from_data(observations)` "
                "to initialize all Poisson readouts at once. See "
                "examples/ensemble_limit_cycle/LESSONS_LEARNED.md.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_uninitialized = True

        log_rates = self.get_mean_output(z).clamp(max=self.log_rate_clamp)
        rates = torch.exp(log_rates) + EPS

        log_likelihood = Poisson(rates).log_prob(y)

        return torch.sum(log_likelihood, (-1, -2))




        

