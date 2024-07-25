import pickle

# import torch
# import math
# from shac import SHAC
# from hyperparameters import PendulumParams
# from envs import PendulumEnv

# config = PendulumParams()

# envs = PendulumEnv(num_envs=config.num_envs, device=config.device)
# obs_dim = envs.observation_space.shape[0]
# act_dim = envs.action_space.shape[0]

# shac = SHAC(params=config, envs=envs, eval_env="HI")
# shac.create_models(act_dim=act_dim, obs_dim=obs_dim)
# envs = PendulumEnv(num_envs=config.num_envs, device=config.device)

# shac.load("/work/flemingc/nvan21/projects/shac/runs/2024-07-23_13-15-47/best_policy.pt")

# actor = shac.actor
# critic = shac.critic
# target_critic = shac.target_critic


# # Stuck at the top
# good_angle = 0  # rads
# good_ang_vel = 0  # rads/s
# good_state = torch.tensor(
#     [math.cos(good_angle), math.sin(good_angle), good_ang_vel]
# ).to(config.device)

# # Stuck at the bottom
# bad_angle = math.pi / 2  # rads
# bad_ang_vel = 8  # rads/s
# bad_state = torch.tensor([math.cos(bad_angle), math.sin(bad_angle), bad_ang_vel]).to(
#     config.device
# )

# good_value = critic(good_state)
# bad_value = critic(bad_state)

# print(f"good value: {good_value}")
# print(f"bad value: {bad_value}")

# idxs = torch.randperm(16).view(4, -1)
# print(idxs[0], idxs[1], idxs[2], idxs[3])

# from hyperparameters import PendulumParams

# test = PendulumParams()

# params = {
#     k: v
#     for k, v in test.__dict__.items()
#     if not k.startswith("__")
#     and not callable(v)
#     and not isinstance(v, staticmethod)
#     and not k == "project_name"
# }

# print(params)

with open("experiments/training_curves/SHAC/seed_0/data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["rewards"])
