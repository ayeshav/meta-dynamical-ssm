from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from utils import *

from nn import (ReadinNetwork,
                MlpDynamics,
                LoRAHypernet,
                GaussianLikelihood,
                LatentDynamicsEncoderDKF,
                EmbeddingEncoder)


class MetaDynamicalSSM(nn.Module):
    def __init__(self,
                 *, 
                 readin_net,
                 latent_encoder,
                 embedding_encoder,
                 hypernetwork,
                 dynamics,
                 likelihood,
                 alpha = 0.1):
        super().__init__()

        self.readin_net = readin_net
        self.latent_encoder = latent_encoder
        self.embedding_encoder = embedding_encoder
        self.hypernetwork = hypernetwork
        self.dynamics = dynamics
        self.likelihood = likelihood

        self.alpha = alpha

    
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
        y_bar = self.readin_net[ds](y_ds)  # [b,T,dim_shared]

        # 2) task/dataset embedding
        e = self.embedding_encoder(y_bar)  # [1, dim_embedding] if pool = True

        # 3) hypernet -> dynamics parameters (low-rank deltas etc.)
        deltas, (mu_0, var_0), delta_norm = self.hypernet(e) 

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

        # 6) likelihood loss
        recon = self.likelihood[ds](y_ds, z)  # scalar tensor

        # 7) dynamics KL: q(z_t) vs p(z_t|z_{t-1})
        _, mu_p, var_p_t = self.dynamics.sample_forward(z[..., :-1, :], deltas)

        kl_t = gaussian_kl(
            mu_q[..., 1:, :], var_q[..., 1:, :],
            mu_p, var_p_t
        )  # [b, T-1, Dz]

        # apply mask on time steps if sampler returns one (align to T-1)
        if t_mask is not None:
            kl_t = kl_t * t_mask[:, 1:, :]

        kl = torch.sum(kl, (-1, -2))
        kl_0 = gaussian_kl(mu_q[..., 0, :], var_q[..., 0, :], mu_0, var_0).sum(-1).mean()

        loss = recon + kl + self.alpha * delta_norm

        outs = {}
        if return_outputs:
            outs = {
                "z": z, "mu_q": mu_q, "var_q": var_q,
                "e": e, "deltas": deltas
            }
        return loss, outs

    
    def forward(self,
                batch: Dict[str, torch.Tensor],
                n_samples: int = 1,
                p_mask: float = 0.0,
                return_outputs: bool = False):
        
        y = batch['y']
        dataset_id = batch.get("y_id", None)

        losses_by_ds: Dict[int, torch.Tensor] = {}
        outputs_by_ds: Optional[Dict[int, Dict[str, torch.Tensor]]] = {} if return_outputs else None

        ds_ids = dataset_id
        total_loss = 0.0

        for ds in ds_ids.unique().tolist():
            idx = (ds_ids == ds).nonzero(as_tuple=True)[0]
            y_ds = y.index_select(0, idx)

            loss_ds, outs = self._run_one_dataset(
                int(ds), y_ds,
                n_samples=n_samples,
                p_mask=p_mask,
                return_outputs=return_outputs
            )
            losses_by_ds[int(ds)] = loss_ds
            total_loss = total_loss + loss_ds
            if return_outputs:
                outputs_by_ds[int(ds)] = outs

        out = {"loss": total_loss,
               "losses_by_dataset": losses_by_ds}
        
        if return_outputs:
            out["outputs_by_dataset"] = outputs_by_ds
        return out


        


    



        