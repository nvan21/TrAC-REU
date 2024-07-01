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
                "actor_units": config.actor_units,
                "critic_units": config.critic_units,
                "critic_minibatches": config.critic_minibatches,
                "critic_iterations": config.critic_iterations,
                "actor_activation": config.actor_activation,
                "critic_activation": config.critic_activation,
                "num_steps": config.num_steps,
                "num_envs": config.num_envs,
                "epochs": config.max_epochs,
                "polyak_averaging": config.tau,
            },
        )

    envs = PendulumEnv(num_envs=config.num_envs, device=config.device)
    # envs = gym.make_vec("Pendulum-v1", num_envs=config.num_envs)
    # envs = gym.make("Pendulum-v1", render_mode="rgb_array")
    # envs = gym.wrappers.RecordVideo(envs, ".")

    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]

    # envs = gym.wrappers.NormalizeObservation(envs)

    shac = SHAC(config=config, envs=envs)
    shac.create_models(act_dim=act_dim, obs_dim=obs_dim)

    # shac.load(
    #     filename="/Users/nvan/Documents/Code/shac/weights/2024-06-27_16-11-31/best_policy.pt"
    # )
    # shac.evaluate_policy()

    shac.train()
