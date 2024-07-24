import os
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv
import torch
import numpy as np
import random
import pickle
from hyperparameters import SACPendulumParams

from stable_baselines3 import SAC


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


# Main Training Script
if __name__ == "__main__":
    env_id = "Pendulum-v1"
    params = SACPendulumParams()
    seeds = params.seeds  # List of seeds
    num_envs = params.num_envs  # Number of parallel environments

    for seed in seeds:
        # Create log directory if it doesn't exist already
        log_dir = f"experiments/length_parameter_curves/SAC/seed_{seed}"
        os.makedirs(log_dir, exist_ok=True)

        # Create environment
        env = PendulumEnv()

        # Load the SAC model
        model = SAC.load(f"experiments/training_curves/SAC/seed_{seed}/best_policy.zip")
        model.set_random_seed(seed=seed)
        model.policy.eval()

        # Set the PRNG seed
        set_seed(seed=seed)
        rewards = []
        lengths = []

        for length in params.lengths:
            env.m = length
            state, _ = env.reset(seed=seed)
            done, truncated = False, False
            total_reward = 0

            for i in range(200):
                action, _ = model.predict(state)
                state, reward, done, truncated, _ = env.step(action)

                total_reward += reward

            print(f"Final reward for pendulum length of {length}: {total_reward}")
            rewards.append(total_reward)
            lengths.append(length)

        with open(f"{log_dir}/data.pkl", "wb") as f:
            data = {"rewards": rewards, "lengths": lengths}
            pickle.dump(data, f)
