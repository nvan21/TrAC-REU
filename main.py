from hyperparameters import HopperConfig
from actor import StochasticActor
from critic import Critic
from shac import SHAC

import torch

import gymnasium as gym

config = HopperConfig()

envs = gym.make_vec("Hopper-v4", num_envs=config.num_envs)
obs_dim = envs.single_observation_space.shape[0]
act_dim = envs.single_action_space.shape[0]

envs = gym.wrappers.NormalizeObservation(envs)
envs = gym.wrappers.NormalizeReward(envs)

shac = SHAC(config=config)
shac.create_models(act_dim=act_dim, obs_dim=obs_dim)

obs, _ = envs.reset()
print(obs.shape)

actions = shac.get_action(obs)
value = shac.get_value(obs)
print(actions.shape, value.shape)
