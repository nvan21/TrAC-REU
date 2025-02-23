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

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = PendulumEnv()
    env.m = 2.0
    env.action_space.seed(0)
    env.observation_space.seed(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    for seed in Experiments.seeds:
        run_dir = f"../experiments/mass_parameter_curves/SHAC/seed_{seed}"
        os.makedirs(run_dir, exist_ok=True)
        shac = SHAC(params=params, envs=env, eval_env=env, run_dir=run_dir)
        shac.set_seed(seed=seed)
        shac.create_models(act_dim=act_dim, obs_dim=obs_dim)
        shac.load(f"../experiments/training_curves/SHAC/seed_{seed}/best_policy.pt")
        shac.actor.eval()

        set_seed(seed=seed)
        rewards = []
        masses = []

        with torch.no_grad():
            for mass in Experiments.masses:
                env.m = mass
                state, _ = env.reset(seed=seed)
                done, truncated = False, False
                total_reward = 0

                for i in range(200):
                    if isinstance(state, np.ndarray):
                        state = torch.tensor(
                            state, dtype=torch.float32, device=params.device
                        )

                    action = shac.actor(state).cpu().detach().numpy()
                    state, reward, done, truncated, _ = env.step(action)

                    total_reward += reward

                print(f"Final reward for pendulum mass of {mass}: {total_reward}")
                rewards.append(total_reward)
                masses.append(mass)

        with open(f"{run_dir}/data.pkl", "wb") as f:
            data = {"rewards": rewards, "masses": masses}
            pickle.dump(data, f)
