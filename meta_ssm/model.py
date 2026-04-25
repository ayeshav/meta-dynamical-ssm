from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

EPS = 1e-6


class MetaDynamicalSSM(nn.Module):
    def __init__(self,
                 *, 
                 adapters,
                 latent_encoder,
                 embedding_encoder,
                 hypernetwork,
                 dynamics,
                 shared_readin,
                 shared_readout,
                 alpha: float = 0.1,
                 concat_embedding: bool = True, 
                 common_init_condition: bool = True):
        super().__init__()

        self.adapters = adapters

        self.latent_encoder = latent_encoder
        self.embedding_encoder = embedding_encoder
        self.hypernetwork = hypernetwork
        self.dynamics = dynamics

        self.shared_readin = shared_readin
        self.shared_readout = shared_readout

        self.alpha = alpha
        self.concat_embedding = concat_embedding
        self.common_init_condition = common_init_condition

        if common_init_condition:
            self.mu_0 = nn.Parameter(torch.ones(1, dynamics.num_latents))
            self.log_variance_0 = nn.Parameter(torch.ones(1, dynamics.num_latents))

    def _run_one_dataset(
        self,
        ds: int,
        y_ds: torch.Tensor,                  # [b,T,Dobs]
        *,
        u_ds: Optional[torch.Tensor] = None, # [b,T,U] or [b,T,?]
        n_samples: int = 1,
        p_mask: float = 0.0,
        return_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns (loss_ds, outputs_dict)
        """

        # 1) readin to shared dimension
        y_bar = self.adapters.readin[ds](y_ds)  # [b,T,dim_shared]

        y_bar = self.shared_readin(y_bar)

        # 2) task/dataset embedding
        mu_e, var_e = self.embedding_encoder(y_bar)
        e = reparametrize(mu_e, var_e)

        # 3) hypernet -> dynamics parameters (low-rank deltas etc.)
        deltas, (mu_0, var_0), delta_norm = self.hypernetwork(e)

        if self.common_init_condition:
            mu_0 = self.mu_0.expand(y_ds.size(0), -1)
            var_0 = F.softplus(self.log_variance_0).expand(y_ds.size(0), -1) + EPS

        # 4) posterior params from encoder
        mu_q, var_q = self.latent_encoder(y_bar, e if self.concat_embedding else None)

        # 5) posterior sampling (scheme-specific)
        z, t_mask = masked_posterior_sampler(
            mu_q, var_q,
            dynamics=self.dynamics,
            n_samples=n_samples,
            p_mask=p_mask,
            deltas=deltas,
        )  # z: [b,T,Dz]

        z = self.shared_readout(z) 

        # 6) likelihood loss
        ll = self.adapters.likelihood[ds](z, y_ds)  # scalar tensor

        # 7) dynamics KL: q(z_t) vs p(z_t|z_{t-1})
        mu_p, var_p_t = self.dynamics(z[..., :-1, :], deltas)

        kl_t = gaussian_kl(
            mu_q[..., 1:, :], var_q[..., 1:, :],
            mu_p, var_p_t
        )  # [b, T-1, Dz]

        # apply mask on time steps if sampler returns one (align to T-1)
        if t_mask is not None:
            kl_t = kl_t * t_mask[:, 1:, :]

        kl = torch.sum(kl_t, (-1, -2))
        kl_0 = gaussian_kl(mu_q[..., 0, :], var_q[..., 0, :], mu_0, var_0).sum(-1).mean()

        # Prior is N(0, I)
        mu_e_0 = torch.zeros_like(mu_e)
        var_e_0 = torch.ones_like(var_e)

        kl_e = gaussian_kl(mu_e, var_e, mu_e_0, var_e_0)     # shape [1, E] (or [B, E])
        kl_e = kl_e.sum(dim=-1).mean()

        elbo = torch.mean(ll - kl) - kl_0 - kl_e
        loss = - elbo + self.alpha * delta_norm 

        outs = {}
        if return_outputs:
            outs = {
                "z": z, "mu_q": mu_q, "var_q": var_q,
                "e": e, "deltas": deltas
            }
        return loss, outs

    
    def forward(self,
                batch: Dict[int, Dict[str, torch.Tensor]],
                n_samples: int = 1,
                p_mask: float = 0.0,
                return_outputs: bool = False):
        

        losses_by_ds = {}
        outputs_by_ds = {} if return_outputs else None

        total_loss = 0.0

        for ds, b in batch.items():
            y = b["y"]
            u = b.get("u", None)

            loss_ds, outs = self._run_one_dataset(
                ds, y, u_ds = u,
                n_samples=n_samples,
                p_mask=p_mask,
                return_outputs=return_outputs
            )
            losses_by_ds[ds] = loss_ds
            total_loss = total_loss + loss_ds
            if return_outputs:
                outputs_by_ds[ds] = outs

        out = {"loss": total_loss,
               "losses_by_dataset": losses_by_ds}
        
        if return_outputs:
            out["outputs_by_dataset"] = outputs_by_ds
        return out


        


    



        