from hyperparameters import SHACPendulumParams
from shac import SHAC
from envs import PendulumEnv

import wandb
import gymnasium as gym


if __name__ == "__main__":
    # Get config
    params = SHACPendulumParams()

    # Create the dictionary of wandb logged parameters
    log_params = {
        k: v
        for k, v in SHACPendulumParams.__dict__.items()
        if not k.startswith("__")
        and not callable(v)
        and not isinstance(v, staticmethod)
        and not k == "project_name"
    }

    # Initialize environments and SHAC instance
    envs = PendulumEnv(num_envs=params.num_envs, device=params.device)
    eval_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    eval_env.action_space.seed(0)
    eval_env.observation_space.seed(0)

    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    for seed in params.seeds:
        # Initialize wandb logger
        if params.do_wandb_logging:
            wandb.init(project=params.project_name, config=log_params)
            wandb.define_metric("epoch")
            wandb.define_metric("step")
            wandb.define_metric("step/*", step_metric="step")
            wandb.define_metric("epoch/*", step_metric="epoch")

        run_dir = f"experiments/training_curves/SHAC/seed_{seed}"
        shac = SHAC(params=params, envs=envs, eval_env=eval_env, run_dir=run_dir)
        shac.set_seed(seed=seed)
        shac.create_models(act_dim=act_dim, obs_dim=obs_dim)
        shac.train()

        if params.do_wandb_logging:
            wandb.finish
