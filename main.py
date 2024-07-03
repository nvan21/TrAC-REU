from hyperparameters import PendulumParams
from shac import SHAC
from envs import PendulumEnv

import wandb
import gymnasium as gym

TRAIN_MODE = True

if __name__ == "__main__":
    # Get config
    params = PendulumParams()

    # Create the dictionary of wandb logged parameters
    log_params = {
        k: v
        for k, v in PendulumParams.__dict__.items()
        if not k.startswith("__")
        and not callable(v)
        and not isinstance(v, staticmethod)
        and not k == "project_name"
    }

    # Initialize wandb logger
    if params.do_wandb_logging:
        wandb.init(project=params.project_name, config=log_params)

    # Initialize environments and SHAC instance
    envs = PendulumEnv(num_envs=params.num_envs, device=params.device)
    # envs = gym.make_vec("Pendulum-v1", num_envs=params.num_envs)
    # envs = gym.make("Pendulum-v1", render_mode="rgb_array")
    # envs = gym.wrappers.RecordVideo(envs, ".")

    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]

    shac = SHAC(params=params, envs=envs)

    if TRAIN_MODE:
        shac.create_models(act_dim=act_dim, obs_dim=obs_dim)
        shac.train()
    else:
        shac.load(filename="./weights/2024-06-27_16-11-31/best_policy.pt")
        shac.evaluate_policy()
