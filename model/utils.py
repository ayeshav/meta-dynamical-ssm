import torch

def reparametrize(mu, var, n_samples=1):
    eps = torch.randn((n_samples,) + mu.shape)
    return (mu + torch.sqrt(var) * eps).squeeze(0)

def gaussian_kl(mu_q, var_q, mu_p, var_p) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussians. Returns [B] or [B,T] depending on inputs."""
    return 0.5 * (torch.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)

def masked_posterior_sampler(mu_q, var_q, dynamics, deltas = None, n_samples = 1, p_mask = 0.2):
    T = mu_q.size(1)

    t_mask = torch.bernoulli((1 - p_mask) * torch.ones((1, T, 1)))

    z0 = reparametrize(mu_q[:, :1], var_q[:, :1], n_samples=n_samples)
    z_s = [z0]
    for t in range(1, T):
        if t_mask[0, t] == 0:
            z_prev = z_s[t - 1]
            z_t = dynamics(z_prev, deltas=deltas)[0]
        else:
            z_t = reparametrize(mu_q[:, t:t+1], var_q[:, t:t+1], n_samples=n_samples)
        z_s.append(z_t)

    return torch.cat(z_s, dim=1), t_mask
