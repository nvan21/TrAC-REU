import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.hyperparameters import SACPendulumParams, Experiments
from utils.utils import set_seed, make_env


# Main Training Script
if __name__ == "__main__":
    env_id = "Pendulum-v1"
    params = SACPendulumParams()
    seeds = Experiments.seeds  # List of seeds
    num_envs = params.num_envs  # Number of parallel environments

    for seed in seeds:
        set_seed(seed)
        log_dir = f"../experiments/training_curves/SAC/seed_{seed}"
        os.makedirs(log_dir, exist_ok=True)
        # Create vectorized environment
        envs = [make_env(env_id, seed, i, log_dir) for i in range(num_envs)]
        vec_env = DummyVecEnv(envs)  # Use DummyVecEnv for simplicity

        # Create and train the SAC model
        model = SAC("MlpPolicy", vec_env, verbose=1, seed=seed)
        model.learn(total_timesteps=params.max_timesteps)

        # Save the model
        model.save(f"{log_dir}/best_policy")
