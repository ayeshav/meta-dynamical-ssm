import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class LoRAHypernet(nn.Module):
    """
    Hypernetwork that outputs low-rank weight updates
    """
    def __init__(
            self,
            dim_embedding: int,
            layer_dims: list[tuple[int, int]],
            adapt_layers: set[int] | None = {0, 1},
            rank: int = 1,
            width: int = 256,
    ):
        super().__init__()

        self.rank = rank
        self.adapt_layers = set(range(len(layer_dims))) if adapt_layers is None else adapt_layers

        self.net = nn.Sequential(nn.Linear(dim_embedding, width),
                                 nn.Tanh(),
                                 nn.Linear(width, width),
                                 nn.Tanh())
        
        self.heads = nn.ModuleList()

        for dim_in, dim_out in layer_dims:
            self.heads.append(nn.Linear(width, rank * dim_in + rank * dim_out))
        
    def set_adapt_layers(self, adapt_layers: set[int] | None):
        """Change which layers are adapted at runtime"""
        self.adapt_layers = set(range(len(self.layer_dims))) if adapt_layers is None else set(adapt_layers)

    def forward(self, e) -> tuple[dict[str, torch.Tensor], float]:
        if e.dim() == 1:
            e = e.unsqueeze(0)
        B = e.shape[0]

        out = self.net(e)

        deltas = {}
        delta_norm = 0

        for i, (dim_in, dim_out) in enumerate(self.layer_dims):
            if i not in self.adapt_layers:
                continue
            
            param_vec = self.heads[i](out)

            u, vh = torch.split(param_vec, [dim_in * self.rank, dim_out * self.rank], -1)

            u = u.view(B, dim_in, self.rank)
            vh = vh.view(B, dim_out, self.rank)

            delta = torch.einsum('bij, bkj -> bik', [u, vh]).squeeze()

            deltas[f"linear_{i}"] = delta
            delta_norm += torch.norm(delta, (-1, -2))

        return deltas, delta_norm


class MlpDynamics(nn.Module):
    def __init__(
            self,
            num_latents: int,
            hidden_layers: int = 2,
            width: int = 128,
            adapt_layers: set[int] | None = {0, 1}
            ):
        super().__init__()

        dims = [num_latents] + hidden_layers * [width] + [num_latents]

        self.adapt_layers = adapt_layers

        self.fc_layers = nn.ModuleList(nn.Linear(dims[i], dims[i+1]) for i in range(len(dims) - 1))
        
        self.act = nn.Tanh()
        self.log_variance = nn.Parameter(torch.ones(1, num_latents))

    def forward(self, z_prev, deltas = None):

        z_next = z_prev
        for i, linear in enumerate(self.fc_layers):
            z_next = linear(z_next)

            if deltas is not None and i in self.adapt_layers:
                z_next = z_next + z_next @ deltas[f"linear_{i}"]

            if i < len(self.fc_layers) - 1:
                z_next = self.act(z_next)

        variance = F.softplus(self.log_variance) + EPS
        return z_next, variance



