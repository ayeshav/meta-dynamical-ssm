from __future__ import annotations
from typing import Callable, Dict

import torch
import torch.nn as nn

from .likelihood import PoissonLikelihood


class Adapters(nn.Module):
    """
    Dataset-dependent readin and likelihood modules.
    """

    def __init__(
        self,
        dim_observations: Dict[str, int],
        *,
        readin_modules: Callable[..., nn.Module],
        likelihood_modules: Callable[..., nn.Module],
    ):
        super().__init__()

        self.readin = nn.ModuleDict()
        self.likelihood = nn.ModuleDict()

        self.readin = nn.ModuleDict({
            k: readin_modules(num_observations=d)
            for k, d in dim_observations.items()
        })
        self.likelihood = nn.ModuleDict({
            k: likelihood_modules(num_observations=d)
            for k, d in dim_observations.items()
        })

    def keys(self):
        return list(self.readin.keys())

    def init_likelihoods_from_data(
        self,
        observations: Dict[str, torch.Tensor],
        **kwargs,
    ) -> int:
        """Warm-start every Poisson likelihood's readout from its dataset's spikes.

        Iterates `self.likelihood.items()` and, for each likelihood that
        is a `PoissonLikelihood`, calls
        `lik.init_from_data(observations[key], **kwargs)`. Likelihoods of
        other types (e.g. Gaussian) are skipped silently. Returns the
        number of likelihoods that were initialized.

        See `PoissonLikelihood.init_from_data` for the available
        keyword arguments (e.g. `sigma_smooth`, `normalize`).
        """
        n = 0
        for key, lik in self.likelihood.items():
            if isinstance(lik, PoissonLikelihood):
                if key not in observations:
                    raise KeyError(
                        f"observations missing data for dataset {key!r} "
                        "(needed to warm-start its PoissonLikelihood)"
                    )
                lik.init_from_data(observations[key], **kwargs)
                n += 1
        return n
