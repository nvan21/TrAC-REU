from hyperparameters import HopperConfig
from shac import SHAC

import gymnasium as gym

if __name__ == "__main__":
    config = HopperConfig()

    envs = gym.make_vec("Hopper-v4", num_envs=config.num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.NormalizeReward(envs)

    shac = SHAC(config=config, envs=envs)
    shac.create_models(act_dim=act_dim, obs_dim=obs_dim)

    shac.train()
