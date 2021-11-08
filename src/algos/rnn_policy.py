import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Policy Optimization
# Trust Region Policy Optimization (use a library)
from gym_bandits.bandit import BanditEnv

from src.envs.init import ENVS_DICT, load_env


# TODO: add generally
# TODO: fix to(device) and to(dtype)
def init():
    global dtype, device
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.double


def Trajectory():
    def __init__(self):
        self.data = defaultdict(list)

    def record(self, s, a, r, d, logits):
        self.data['s'].append(s)
        self.data['a'].append(a)
        self.data['r'].append(r)
        self.data['d'].append(d)
        self.data['logits'].append(logits)

    def add(self, type, data):
        self.data[type].append(data)

    def get(self, type):
        return self.data[type]


class RNNPolicy(nn.Module):
    def __init__(self, env, hidden_dim, n_layers):
        super(RNNPolicy, self).__init__()
        self.env = env
        self.state_emb_size = self.env.observation_space.n
        self.action_emb_size = self.env.action_space.n
        self.input_dim = self.state_emb_size + self.action_emb_size + 2  # + reward + done
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, n_layers, bias=True).double()
        self.fc = nn.Linear(self.hidden_dim, self.action_emb_size).double()

        self.batch_size = 1

        self.init_hidden()

    # we let PyTorch handle weight initializations https://stackoverflow.com/a/56773737
    #   weight normalization without data-dependent initialization to all weight matrixes
    #   hidden to hidden w: orthogonal init
    #   all other w: Xavier init
    #   bias vector: 0 init
    def init_hidden(self):
        # TODO: change for GRU (e.g. no internal cell state)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim, dtype=dtype)
        self.cell = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim, dtype=dtype)
        # TODO: to.(dtype).to.device(xxx) ?
        # ReLU as hidden activation

    # bandits: constant 0 embedding for states, one-hot encode actions
    # tabular MDP: one-hot encode states and actions
    # visual: TODO
    def encode_input(self, state, action, reward, done):
        if isinstance(self.env, BanditEnv):
            state = torch.tensor([0]).to(dtype).unsqueeze(0)
        # TODO elif: tabular MDP

        action = F.one_hot(action.long(), num_classes=self.action_emb_size)
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        # TODO: to.(dtype).to.device(xxx) ?
        return torch.cat((state, action, reward, done), dim=1)
        # TODO: to.(dtype).to.device(xxx) ?

    # returns actions probabilities
    # input (state, action, reward, done)
    # embedded with function for input of GRU
    # fully connected
    # softmax dist over actions
    def forward(self, state, action, reward, done):
        input = self.encode_input(state, action, reward, done)
        input = input.unsqueeze(0)  # needs to be of size (seq_len, batch, input_size)

        out, (self.hidden, self.cell) = self.rnn(input, (self.hidden, self.cell))
        # TODO: call ReLU on hidden too? theoretically not needed as LSTM already has non-linearity, for GRU?
        out = self.fc(F.relu(out))

        # TODO: to.(dtype).to.device(xxx) ?
        self.hidden.detach_()

        prob = F.softmax(out, dim=2).view(-1)
        return prob


def get_init_input(env):
    s = env.reset()
    s = torch.tensor(s, device=device)
    s = F.one_hot(s, num_classes=env.observation_space.n).to(dtype)
    s = s.unsqueeze(0)
    a = torch.tensor([0], device=device, dtype=dtype)
    r = torch.tensor([0], device=device, dtype=dtype)
    d = torch.tensor([0], device=device, dtype=dtype)
    return s, a, r, d


def unit_test():
    env = load_env('bandits')
    policy = RNNPolicy(env, 32, 2)
    probs = policy(*get_init_input(env))
    print('action probabilities:', probs)
    print('sampled action:', torch.distributions.Categorical(probs).sample())


def train(env, N_HIDDEN=32, N_LAYERS=2):
    # TODO: do not forget to set to eval() mode during testing
    # policy.train()
    policy = RNNPolicy(env, N_HIDDEN, N_LAYERS)


if __name__ == '__main__':
    init()
    unit_test()
