import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the policy network and value network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


# Example environment setup
input_size = 4  # Example state size
output_size = 2  # Example action size

policy_net = PolicyNetwork(input_size, output_size)
value_net = ValueNetwork(input_size)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.01)
optimizer_value = optim.Adam(value_net.parameters(), lr=0.01)

# Simulate a trajectory
trajectory = {
    "states": [np.random.rand(input_size) for _ in range(16)],
    "actions": [np.random.randint(output_size) for _ in range(16)],
    "rewards": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],  # Example constant reward
}

# Convert trajectory to PyTorch tensors
states = torch.tensor(trajectory["states"], dtype=torch.float32)
actions = torch.tensor(trajectory["actions"], dtype=torch.int64)
rewards = torch.tensor(trajectory["rewards"], dtype=torch.float32)

# Calculate discounted returns
gamma = 0.99
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)
returns = torch.tensor(returns, dtype=torch.float32)

# Policy gradient update
optimizer_policy.zero_grad()
optimizer_value.zero_grad()

log_probs = torch.log(policy_net(states))
selected_log_probs = log_probs[range(len(actions)), actions]

# Calculate the value of the last state
last_state = states[-1]
last_value = value_net(last_state).squeeze()

# Update returns with the value of the last state
returns[-1] += gamma * last_value.item()

# Calculate the policy loss
policy_loss = -torch.sum(selected_log_probs * returns)

# Calculate the value loss for the last state
value_loss = nn.functional.mse_loss(last_value, returns[-1].unsqueeze(0))

# Backpropagation
(policy_loss + value_loss).backward()
optimizer_policy.step()
optimizer_value.step()

print("Policy loss:", policy_loss.item())
print("Value loss:", value_loss.item())


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
