import os
import gymnasium as gym
import torch
import numpy as np
import random
from stable_baselines3.common.monitor import Monitor


# Set Seed Function
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Create Seeded Environment Function
def create_seeded_env(env_id, seed, log_dir, rank=0):
    env = gym.make(env_id)
    env = SeededEnvWrapper(env, seed=seed)
    env = Monitor(
        env,
        filename=os.path.join(log_dir, f"monitor_{seed}_{rank}.csv"),
        allow_early_resets=True,
    )
    return env


# Function to make environments
def make_env(env_id, seed, rank, log_dir):
    def _init():
        env = create_seeded_env(env_id, seed, log_dir, rank)
        return env

    return _init


# Custom Environment Wrapper
class SeededEnvWrapper(gym.Wrapper):
    def __init__(self, env, seed=None):
        super(SeededEnvWrapper, self).__init__(env)
        self.seed_value = seed

    def reset(self, **kwargs):
        if self.seed_value is not None:
            self.env.action_space.seed(self.seed_value)
            self.env.observation_space.seed(self.seed_value)
            set_seed(self.seed_value)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
