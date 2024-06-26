# import torch
# import torch.distributions
# from envs import Pendulum

# device = torch.device("cpu")
# num_envs = 4

# env = Pendulum(num_envs, device)
# states, _ = env.reset()

# done = False
# truncated = False

# loss = torch.tensor(0, dtype=torch.float32).to(device)
# while not done and not truncated:
#     action = torch.randn((4, 1)).to(device)
#     state, reward, done, truncated, info = env.step(action.unsqueeze(-1))

#     loss += reward.sum()
# loss /= 200 * num_envs
# print(loss)
# loss.backward()

import torch
import torch.nn as nn


class DifferentiablePendulumEnv:
    def __init__(self):
        self.g = 9.81
        self.l = 1.0
        self.dt = 0.1
        self.state = None

    def reset(self):
        self.state = torch.tensor([0.5, 0.0], requires_grad=True)
        return self.state

    def step(self, action):
        theta, theta_dot = self.state
        theta_dot_new = (
            theta_dot
            + (-3 * self.g / (2 * self.l) * torch.sin(theta + action)) * self.dt
        )
        theta_new = theta + theta_dot_new * self.dt
        self.state = torch.tensor([theta_new, theta_dot_new], requires_grad=True)
        reward = -(theta_new**2)  # Simple reward: penalize angle
        done = False  # Pendulum never really 'ends'
        return self.state, reward, done


# Define a function to run the simulation and update the loss after each step
def run_simulation(env, actions):
    state = env.reset()
    trajectory = [state]
    loss = torch.tensor(0.0, requires_grad=True)

    for action in actions:
        state, reward, done = env.step(action)
        trajectory.append(state)
        loss = loss + reward**2  # Update loss incrementally

    return trajectory, loss


# Initialize environment
env = DifferentiablePendulumEnv()

# Define actions
actions = [torch.tensor([0.1], requires_grad=True) for _ in range(16)]

# Run simulation
trajectory, loss = run_simulation(env, actions)

# Perform backward pass
loss.backward()

# Print gradients
print("Gradients for initial state:", trajectory[0].grad)
for i, action in enumerate(actions):
    print(f"Gradients for action {i}:", action.grad)

# Reset environment for another episode
env.reset()


# import pickle
# import torch
# import math

# checkpoint = torch.load("best_policy.pt")
# actor = checkpoint[0]
# critic = checkpoint[1]
# target_critic = checkpoint[2]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Stuck at the top
# good_angle = 0  # rads
# good_ang_vel = 0  # rads/s
# good_state = torch.tensor(
#     [math.cos(good_angle), math.sin(good_angle), good_ang_vel]
# ).to(device)

# # Stuck at the bottom
# bad_angle = math.pi  # rads
# bad_ang_vel = 0  # rads/s
# bad_state = torch.tensor([math.cos(bad_angle), math.sin(bad_angle), bad_ang_vel]).to(
#     device
# )

# good_value = critic(good_state)
# bad_value = critic(bad_state)

# print(f"good value: {good_value}")
# print(f"bad value: {bad_value}")
