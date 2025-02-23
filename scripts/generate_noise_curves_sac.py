import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import pickle
import numpy as np
from stable_baselines3 import SAC

from utils.hyperparameters import SACPendulumParams, Experiments
from utils.utils import set_seed


# Main Training Script
if __name__ == "__main__":
    env_id = "Pendulum-v1"
    params = SACPendulumParams()
    seeds = Experiments.seeds  # List of seeds
    num_envs = params.num_envs  # Number of parallel environments

    for seed in seeds:
        # Create log directory if it doesn't exist already
        log_dir = f"../experiments/noise_curves/SAC/seed_{seed}"
        os.makedirs(log_dir, exist_ok=True)

        # Create environment
        env = gym.make("Pendulum-v1")

        # Load the SAC model
        model = SAC.load(
            f"../experiments/training_curves/SAC/seed_{seed}/best_policy.zip"
        )
        model.set_random_seed(seed=seed)
        model.policy.eval()

        # Set the PRNG seed
        set_seed(seed=seed)
        rewards = []
        std = []

        for noise in Experiments.noises:
            state, _ = env.reset(seed=seed)
            done, truncated = False, False
            total_reward = 0

            while not done and not truncated:
                state = state + np.random.normal(
                    loc=0.0, scale=noise, size=env.observation_space.shape[0]
                )
                action, _ = model.predict(state)
                state, reward, done, truncated, _ = env.step(action)

                total_reward += reward

            print(f"Final reward for standard deviation of {noise}: {total_reward}")
            rewards.append(total_reward)
            std.append(noise)

        with open(f"{log_dir}/data.pkl", "wb") as f:
            data = {"rewards": rewards, "noises": std}
            pickle.dump(data, f)
