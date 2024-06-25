from hyperparameters import HopperConfig
from shac import SHAC
import utils

import torch
import wandb
import gymnasium as gym


if __name__ == "__main__":
    # Get config
    config = HopperConfig()

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

    envs = gym.make_vec("Pendulum-v1", num_envs=config.num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    envs = gym.wrappers.NormalizeObservation(envs)

    shac = SHAC(config=config, envs=envs)
    shac.create_models(act_dim=act_dim, obs_dim=obs_dim)

    shac.train()
