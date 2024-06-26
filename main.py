from hyperparameters import HopperConfig, PendulumConfig
from shac import SHAC
import utils
from envs import PendulumEnv

import torch
import wandb
import gymnasium as gym


if __name__ == "__main__":
    # Get config
    config = PendulumConfig()

    # Initialize wandb logger
    if config.do_wandb_logging:
        wandb.init(
            project="shac-hopper",
            config={
                "actor_learning_rate": config.actor_learning_rate,
                "critic_learning_rate": config.critic_learning_rate,
                "epochs": config.max_epochs,
            },
        )

    envs = PendulumEnv(num_envs=config.num_envs, device=config.device)
    # envs = gym.make_vec("Pendulum-v1", num_envs=config.num_envs)
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]

    # envs = gym.wrappers.NormalizeObservation(envs)

    shac = SHAC(config=config, envs=envs)
    shac.create_models(act_dim=act_dim, obs_dim=obs_dim)

    shac.train()
