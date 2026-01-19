from __future__ import annotations
from typing import Callable, Dict

import torch.nn as nn


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
