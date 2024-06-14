import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        units: list,
        device: torch.device,
        activation_fn,
    ):
        super(Critic, self).__init__()

        layers = [obs_dim] + units + [1]
        modules = []

        for in_dim, out_dim in zip(layers, layers[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(activation_fn)
            modules.append(nn.LayerNorm(out_dim))

        # Get rid of the last two items in the list because the last layer only needs the linear connection
        self.critic = nn.Sequential(*modules[:-2]).to(device)
        print(f"critic {self.critic}")

    def forward(self, obs):
        return self.critic(obs)
