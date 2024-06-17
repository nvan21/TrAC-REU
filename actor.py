import torch
import torch.nn as nn
from torch.distributions import Normal


class StochasticActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        units: list,
        device: torch.device,
        activation_fn,
    ):
        super(StochasticActor, self).__init__()

        layers = [obs_dim] + units + [act_dim]
        modules = []

        for in_dim, out_dim in zip(layers, layers[1:]):
            modules.append(nn.Linear(in_dim, out_dim, dtype=torch.float32))
            modules.append(activation_fn)
            modules.append(nn.LayerNorm(out_dim, dtype=torch.float32))

        self.mu_network = nn.Sequential(*modules[:-2]).to(device)
        self.logstd_layer = nn.Parameter(torch.ones(act_dim, device=device) * -1).to(
            device
        )

    def forward(self, obs):
        # Get the predicted mean and standard deviation of the state
        mu = self.mu_network(obs)
        std = self.logstd_layer.exp()

        # Sample actions from the corresponding normal distribution, and then apply tanh squashing function
        dist = Normal(mu, std)
        actions = dist.rsample()
        actions = torch.tanh(actions)

        return actions
