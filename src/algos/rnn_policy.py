import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Policy Optimization
# Trust Region Policy Optimization (use a library)
from gym_bandits.bandit import BanditEnv
from torch import optim

from src.envs.init import ENVS_DICT, load_env


LEARNING_RATE = 0.01    # TODO try 0.003
MOMENTUM = 0.9
TRIALS = 100
EPISODES = 100
TRAJECTORY_LEN = 100
GAMMA = 0.99

# TODO: add generally
# TODO: fix to(device) and to(dtype)
def init():
    global dtype, device
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.double


class Trajectory():
    def __init__(self, s=None, a=None, r=None, d=None, prob=None):
        self.data = defaultdict(list)
        self.existing_keys = {'s', 'a', 'r', 'd', 'prob'}
        if s is not None:
            self.record(s, a, r, d, prob)

    def record(self, s, a, r, d, prob):
        self.data['s'].append(s)
        self.data['a'].append(a)
        self.data['r'].append(r)
        self.data['d'].append(d)
        self.data['prob'].append(prob)

    def add(self, type, data):
        self.data[type].append(data)

    def __getattr__(self, item):
        if item not in self.existing_keys:
            super().__getattribute__(item)
            #raise KeyError(f'{item} must be in {self.existing_keys}')
        return self.data[item]


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
        self.cell.detach_()

        prob = F.softmax(out, dim=2).view(-1)
        return prob


def make_tensor(s, a, r, d, env):
    s = torch.tensor(s, device=device)
    s = F.one_hot(s, num_classes=env.observation_space.n).to(dtype)
    s = s.unsqueeze(0)
    a = torch.tensor([a], device=device, dtype=dtype)
    r = torch.tensor([r], device=device, dtype=dtype)
    d = torch.tensor([d], device=device, dtype=dtype)
    return s, a, r, d


def unit_test():
    env = load_env('bandits')
    policy = RNNPolicy(env, 32, 2)
    s = env.reset()
    probs = policy(*make_tensor(s, 0, 0, 0, env))
    print('action probabilities:', probs)
    print('sampled action:', torch.distributions.Categorical(probs).sample())


# TODO: add TRPO/PPO
def optim_step(policy_optim, trajectory, method='REINFORCE'):
    if method == 'REINFORCE':
        rewards = trajectory.r
        prob = trajectory.prob

        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + GAMMA * R
            discounted_rewards.append(R)

        discounted_rewards = discounted_rewards[::-1]

        policy_loss = []
        # use log to scale loss https://towardsdatascience.com/policy-gradient-methods-104c783251e0
        # TODO: can also try out averaging probs
        for p, r in zip(prob, discounted_rewards):
            policy_loss.append(- torch.log(p) * r)

        policy_optim.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()#retain_graph=True)     # TODO: retain_graph only needed when we have 2 losses/ two outputs
        policy_optim.step()


def train(env, N_HIDDEN=32, N_LAYERS=2):
    # TODO: do not forget to set to eval() mode during testing
    policy = RNNPolicy(env, N_HIDDEN, N_LAYERS)
    policy.train()
    policy_optim = optim.SGD(policy.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # maximize the expected total discounted reward accumulated during a single trial rather than a single episode
    # TODO: plot rewards over trials
    for trial in range(TRIALS):
        # reset hidden state
        policy.init_hidden()
        # for each trial, separate MDP is drawn
        # TODO: reinit new Banditproblem?

        # TODO: they also have for e in epochs here (only needed for PPO?)

        for episode in range(EPISODES):
            start_time = time.time()
            # for each episode, new initial state
            s, a, r, d = s = env.reset(), 0, 0, 0

            trajectory = Trajectory(s, a, r, d, torch.tensor([]))

            for t in range(TRAJECTORY_LEN):
                a_probs = policy(*make_tensor(s, a, r, d, env))
                a = torch.distributions.Categorical(a_probs).sample()
                s, r, d, _ = env.step(a.item())

                if isinstance(env, BanditEnv):
                    d = (t == TRAJECTORY_LEN - 1)
                print(f'Episode: {episode} | step: {t}', s, a, a_probs.tolist(), r, d)

                trajectory.record(s, a, r, d, a_probs)

                if d:
                    break

            # after trajectory, update (or after episodes acc to PPO?)
            optim_step(policy_optim, trajectory)

            # record time, avg/var of action and reward

            # print rewards


if __name__ == '__main__':
    init()
    unit_test()
    train(load_env('bandits'), 32, 2)
