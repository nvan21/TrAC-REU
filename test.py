# import pickle
# import torch
# import math
# from shac import SHAC
# from hyperparameters import PendulumConfig
# from envs import PendulumEnv

# config = PendulumConfig()

# envs = PendulumEnv(num_envs=config.num_envs, device=config.device)
# obs_dim = envs.observation_space.shape[0]
# act_dim = envs.action_space.shape[0]

# shac = SHAC(config=config, envs=envs)
# shac.create_models(act_dim=act_dim, obs_dim=obs_dim)
# envs = PendulumEnv(num_envs=config.num_envs, device=config.device)

# shac.actor.load_state_dict(
#     torch.load(
#         "/work/flemingc/nvan21/projects/shac/weights/2024-06-27_15-02-59/actor.pt"
#     )
# )
# shac.critic.load_state_dict(
#     torch.load(
#         "/work/flemingc/nvan21/projects/shac/weights/2024-06-27_15-02-59/critic.pt"
#     )
# )
# shac.target_critic.load_state_dict(
#     torch.load(
#         "/work/flemingc/nvan21/projects/shac/weights/2024-06-27_15-02-59/target_critic.pt"
#     )
# )

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

import torch
import numpy as np

I = torch.tensor(
    [[1, 2, 3, 4], [0, 1, 2, 3], [-1, -2, -3, -4], [10, 20, 30, 40]],
    dtype=torch.float32,
)

Inp = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
state = torch.ones((3, 4))
statenp = np.ones((3, 4))

print((I @ state.T).T)
meas_state = (I.unsqueeze(0).repeat(3, 1, 1) @ state.unsqueeze(-1)).squeeze(-1)
print(meas_state, meas_state.shape)

x_meas = np.array((1, 200, 300, 100))
noise = np.array((1, 1, 1, 1))

x_meas = torch.tensor(x_meas)
# x_meas = torch.zeros(4)
noise = torch.randn_like(state)

print(meas_state + noise * x_meas)
