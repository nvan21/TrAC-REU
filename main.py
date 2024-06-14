from hyperparameters import HopperConfig
from actor import StochasticActor
from critic import Critic

import torch

obs_dim = 10
act_dim = 5

config = HopperConfig()
actor = StochasticActor(
    obs_dim=obs_dim,
    act_dim=act_dim,
    units=config.actor_units,
    activation_fn=config.actor_activation,
    device=config.device,
)
critic = Critic(
    obs_dim=obs_dim,
    units=config.critic_units,
    activation_fn=config.critic_activation,
    device=config.device,
)

obs = torch.rand(10).unsqueeze(0).to(config.device)
actions = actor(obs)
value = critic(obs)
print(actions, value)
