import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_action_repeats, hidden_dim):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_action_repeats)

        self.apply(weights_init_)
        self.linear3.weight.data.fill_(0)
        self.linear3.bias.data.fill_(0)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, action_space):
        super(ActorNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x)) * self.action_scale + self.action_bias
        return x

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(ActorNetwork, self).to(device)


class TD3Network(nn.Module):
    def __init__(self, num_inputs, num_actions, action_repeats, hidden_dim, action_space):
        super(TD3Network, self).__init__()
        self.action_repeats = action_repeats
        self.actor = ActorNetwork(num_inputs, num_actions, hidden_dim, action_space)
        self.critic_1 = QNetwork(num_inputs, num_actions, len(action_repeats), hidden_dim)
        self.critic_2 = QNetwork(num_inputs, num_actions, len(action_repeats), hidden_dim)
