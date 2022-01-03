import imageio
from torch import nn

from utils import make_env, Storage, orthogonal_init
import torch


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
      dist, value, _ = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)

    return action.cpu(), log_prob.cpu(), value.cpu()

  def act_greedy(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value, logits = self.forward(x)
      action = torch.argmax(logits, dim=1)
      log_prob = dist.log_prob(action)

    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value, logits

num_envs = 1
num_levels = 2000

seed = 10



p = input("path:")
# Evaluate policy
in_channels = 3
feature_dim = 512
#feature_dim = 256
num_actions = 15

encoder = Encoder(in_channels, feature_dim)
policy = Policy(encoder, feature_dim, num_actions)
policy.load_state_dict(torch.load(p))
policy.cuda()
policy.eval()

frames = []
total_reward = []
# Make evaluation environment
eval_env = make_env(64, num_levels=num_levels,seed=seed)
obs = eval_env.reset()

for _ in range(1600):

# Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = eval_env.step(action)

    # Render environment and store
    frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
    frames.append(frame)



# Save frames as video
frames = torch.stack(frames)
imageio.mimsave(f'{p}_seed{seed}.mp4', frames, fps=25)