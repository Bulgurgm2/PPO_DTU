import sys
import torch
import torch.nn as nn
from utils import make_env, Storage, orthogonal_init
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

print("start")

total_steps = 15e6
num_envs = 64  # unchangeable
num_levels = 10000
num_steps = 256
num_epochs = 3
batch_size = 16
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
learning_rate = 1e-4
optimizer_eps = 1e-5
feature_dim = 16
in_channels = 3  # unchangeable

title = ""
name_val = ""

for arg in sys.argv[1:]:
    option, val = arg.split("=")

    if "title" == option:
        title = val
    elif "name" == option:
        name_val = val
    else:
        val = float(val)
        name_val += arg
        if "total_steps" == option:
            total_steps = val
        elif "num_envs" == option:
            num_envs = val
        elif "num_levels" == option:
            num_levels = val
        elif "num_steps" == option:
            num_steps = val
        elif "num_epochs" == option:
            num_epochs = val
        elif "batch_size" == option:
            batch_size = val
        elif "eps" == option:
            eps = val
        elif "grad_eps" == option:
            grad_eps = val
        elif "value_coef" == option:
            value_coef = val
        elif "entropy_coef" == option:
            entropy_coef = val
        elif "learning_rate" == option:
            learning_rate = val
        elif "feature_dim" == option:
            feature_dim = val
        else:
            raise TypeError(f"unknown arg: {arg}")
    name_val.replace("_", "")


#  start PPO

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)


class Policy(nn.Module):
    def __init__(self, encoder, feature_dim, num_actions):
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

    def act(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            dist, value = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value


# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

eval_env = make_env(num_envs, num_levels=num_levels, seed=2)
eval_obs = eval_env.reset()

# Define network
num_actions = env.action_space.n
encoder = Encoder(in_channels, feature_dim)
policy = Policy(encoder, feature_dim, num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=optimizer_eps)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Run training
obs = env.reset()
step = 0

data_steps = []
data_val = []
data_train = []

while step < total_steps:

    # Use policy to collect data for num_steps steps
    policy.eval()
    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = policy.act(obs)

        # Take step in environment
        next_obs, reward, done, info = env.step(action)

        # print(reward.mean())
        # Store data
        storage.store(obs, action, reward, done, info, log_prob, value)

        # Update current observation
        obs = next_obs

    # Add the last observation to collected data
    _, _, value = policy.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()

    # Optimize policy
    policy.train()
    for epoch in range(num_epochs):

        # Iterate over batches of transitions
        generator = storage.get_generator(batch_size)
        for batch in generator:
            b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

            # Get current policy outputs
            new_dist, new_value = policy(b_obs)
            new_log_prob = new_dist.log_prob(b_action)

            # Clipped policy objective
            ratio = torch.exp(new_log_prob - b_log_prob)
            clipped_ratio = ratio.clamp(min=1.0 - eps, max=1.0 + eps)
            policy_reward = torch.min(ratio * b_advantage, clipped_ratio * b_advantage)
            pi_loss = policy_reward.mean()

            # Clipped value function objective
            clipped_value = b_value + (new_value - b_value).clamp(min=-eps, max=eps)
            vf_loss = torch.max((new_value - b_returns) ** 2, (clipped_value - b_returns) ** 2)
            value_loss = 0.5 * vf_loss.mean()

            # Entropy loss
            entropy_loss = new_dist.entropy().mean()

            # Backpropagate losses
            c_1 = value_coef
            c_2 = entropy_coef
            loss = -(pi_loss - c_1 * value_loss + c_2 * entropy_loss)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

            # Update policy
            optimizer.step()
            optimizer.zero_grad()

    # Update stats
    step += num_envs * num_steps

    # Evaluate policy
    policy.eval()

    t_reward = []
    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = policy.act(eval_obs)
        # print(action)
        # Take step in environment
        eval_obs, reward, done, info = eval_env.step(action)

        t_reward.append(reward)

    data_val.append(np.mean(t_reward))
    data_train.append(storage.get_reward() / num_envs)
    data_steps.append(step)
    print(f'Step: {step}\nMean train reward: {data_train[-1]}\nMean val reward: {data_val[-1]}')

print('Completed training!')
try:
    torch.save(policy.state_dict, f"{title}/{name_val}_checkpoint.pt")
except:
    pass
plt.semilogx(data_steps, data_train, label='train', color="red")
plt.semilogx(data_steps, data_val, label='validation', color="blue")

plt.xlabel("Steps")
plt.ylabel("Reward")

plt.legend(loc='upper right')
plt.grid()

for i in range(2000):
    name = f"{title}/graph{name_val}.png"
    if os.path.isfile(os.path.join(os.getcwd(), name)):
        continue
    else:
        plt.savefig(name)
        print("Graph ", i, " saved")
        plt.clf()
        break

with open(f"{title}/sample_{name_val}.csv", "w", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["step", "train", "val"])
    for (s, t, v) in zip(data_train, data_steps, data_val):
        writer.writerow([t, s, v])
print('Completed training!')
