import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.hyperparameters import SHACPendulumParams, Experiments
from utils.utils import set_seed
from algos.shac import SHAC

import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv
import os
import pickle


if __name__ == "__main__":
    # Get config
    params = SHACPendulumParams()

    # Initialize environments and SHAC instance
    envs = PendulumEnv(num_envs=params.num_envs, device=params.device)
    eval_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    eval_env.action_space.seed(0)
    eval_env.observation_space.seed(0)

    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    for seed in Experiments.seeds:
        run_dir = f"../experiments/noise_curves/SHAC/seed_{seed}"
        os.makedirs(run_dir, exist_ok=True)
        shac = SHAC(params=params, envs=envs, eval_env=eval_env, run_dir=run_dir)
        shac.set_seed(seed=seed)
        shac.create_models(act_dim=act_dim, obs_dim=obs_dim)
        shac.load(f"../experiments/training_curves/SHAC/seed_{seed}/best_policy.pt")
        shac.actor.eval()

        set_seed(seed=seed)
        rewards = []
        std = []

        with torch.no_grad():
            for noise in Experiments.noises:
                state, _ = eval_env.reset(seed=seed)
                done, truncated = False, False
                total_reward = 0

                while not done and not truncated:
                    state = state + np.random.normal(loc=0.0, scale=noise, size=obs_dim)
                    if isinstance(state, np.ndarray):
                        state = torch.tensor(
                            state, dtype=torch.float32, device=params.device
                        )

                    action = shac.actor(state).cpu().detach().numpy()
                    state, reward, done, truncated, _ = eval_env.step(action)

                    total_reward += reward

                print(f"Final reward for standard deviation of {noise}: {total_reward}")
                rewards.append(total_reward)
                std.append(noise)

        with open(f"{run_dir}/data.pkl", "wb") as f:
            data = {"rewards": rewards, "noises": std}
            pickle.dump(data, f)
