import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Policy Optimization
# Trust Region Policy Optimization (use a library)
import wandb as logger
from gym_bandits.bandit import BanditEnv
from torch import optim

from src.envs.init import ENVS_DICT, load_env

DEBUG = True

POLICY_LEARNING_RATE = 0.001  # 0.01  # TODO try 0.003 to see act_std change for the better
VALUE_FN_LEARNING_RATE = 0.005  # 0.01  # TODO try 0.003 to see act_std change for the better
# MOMENTUM = 0.9
GAMMA = 0.99

TRIALS = 10
EPOCHS = 100  # 500
EPISODES = 100  # 50
TRAJECTORY_LEN = 100  # 1000
PRINT_EVERY = 10

if DEBUG:
    TRIALS = 2
    EPOCHS = 10  # 500
    EPISODES = 50  # 50
    TRAJECTORY_LEN = 50  # 1000
    PRINT_EVERY = 1

N_HIDDEN_POLICY = 32
N_HIDDEN_VALUE_FN = 32
N_LAYERS = 2

N_POLICY_UPDATES = 8  # 16
N_VALUE_UPDATES = 8  # 16
CLIP_EPSILON = 0.2


# TODO: add generally
# TODO: fix to(device) and to(dtype)
def init():
    global dtype, device
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.double


class DataRecorder():
    def __init__(self, existing_keys):
        self.data = defaultdict(list)
        self.existing_keys = existing_keys

    def record(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.existing_keys:
                raise KeyError(f'{k} must be in {self.existing_keys}')
            self.data[k].append(v)

    def __getattr__(self, item):
        if item not in self.existing_keys:
            super().__getattribute__(item)
        return self.data[item]


class Trajectory(DataRecorder):
    def __init__(self):
        super(Trajectory, self).__init__(existing_keys={'s', 'a', 'r', 'd', 'a_logits'})

    def record(self, s, a, r, d, a_logits):
        super().record(s=s, a=a, r=r, d=d, a_logits=a_logits)


class SummaryStatistics(DataRecorder):
    def __init__(self, trajectory_len, episodes):
        super(SummaryStatistics, self).__init__(
            existing_keys={
                # per episode
                'time', 'a_avg', 'a_std', 'r_avg', 'r_std', 'r_sum', 'trajectories',
                # per epoch
                'episode_r_avg'})
        self.trajectory_len = trajectory_len
        self.episodes = episodes

    def record(self, time, trajectory):
        a_avg = np.mean(trajectory.a)
        a_std = np.std(trajectory.a)
        r_avg = np.mean(trajectory.r)
        r_std = np.std(trajectory.r)
        r_sum = np.sum(trajectory.r)
        super().record(time=time, a_avg=a_avg, a_std=a_std, r_avg=r_avg, r_std=r_std, r_sum=r_sum,
                       trajectories=trajectory)

    # for readability
    def add_epochal(self, **oneval):
        super().record(**oneval)


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

    def encode_state(self, state, batch=False):
        if batch:
            return F.one_hot(state, num_classes=self.state_emb_size).to(dtype)
        else:
            return F.one_hot(state.long(), num_classes=self.state_emb_size).to(dtype).unsqueeze(1)

    # bandits: constant 0 embedding for states, one-hot encode actions
    # tabular MDP: one-hot encode states and actions
    # visual: TODO
    def encode_input(self, state, action, reward, done, batch):
        state = self.encode_state(state, batch)
        # if isinstance(self.env, BanditEnv):
        #    state = torch.tensor([0]).to(dtype).unsqueeze(0)
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
    def forward(self, state, action, reward, done, batch=False):
        input = self.encode_input(state, action, reward, done, batch)
        batch_size = input.shape[0]
        input = input.unsqueeze(0)  # needs to be of size (seq_len, batch, input_size)

        if batch:
            h, c = self.hidden.clone(), self.cell.clone()
            h, c = h.repeat(1, batch_size, 1), c.repeat(1, batch_size, 1)
            # h, c = torch.cat(batch_size * [h]), torch.cat(batch_size * [c])
            h, c = h.detach(), c.detach()
            # 2 20 32

            out, (h, c) = self.rnn(input, (h, c))
            h.detach_(), c.detach_()

            # TODO: call ReLU on hidden too? theoretically not needed as LSTM already has non-linearity, for GRU?
            out = self.fc(F.relu(out))
            return out.squeeze(0)

        out, (self.hidden, self.cell) = self.rnn(input, (self.hidden, self.cell))
        # TODO: call ReLU on hidden too? theoretically not needed as LSTM already has non-linearity, for GRU?
        out = self.fc(F.relu(out))

        # TODO: to.(dtype).to.device(xxx) ?
        self.hidden.detach_()
        self.cell.detach_()

        return out.view(-1)  # logit

        # prob = F.softmax(out, dim=2).view(-1)
        # if any(torch.isnan(p) for p in prob):
        #    print('err')
        # return prob


# torch.autograd.detect_anomaly()


def make_tensor(s, a, r, d, env):
    s = torch.tensor(s, device=device, dtype=dtype)
    # s = F.one_hot(s, num_classes=env.observation_space.n).to(dtype)
    # s = s.unsqueeze(0)
    a = torch.tensor([a], device=device, dtype=dtype)
    r = torch.tensor([r], device=device, dtype=dtype)
    d = torch.tensor([d], device=device, dtype=dtype)
    return s, a, r, d


def unit_test():
    env = load_env('bandits')
    policy = RNNPolicy(env, 32, 2)
    s = env.reset()
    logits = policy(*make_tensor(s, 0, 0, 0, env))
    print('action probabilities:', logits)
    print('sampled action:', torch.distributions.Categorical(logits=logits).sample())


def discount_rewards(rewards, normalize=True):
    discounted_rewards = []
    R = np.zeros(len(rewards))
    for r in np.array(rewards).T[::-1]:
        R = r + GAMMA * R
        discounted_rewards.append(R)
    discounted_rewards = np.array(discounted_rewards[::-1]).T
    # TODO check if correct
    if normalize:
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards, axis=1, keepdims=True)) / (
                np.std(discounted_rewards, axis=1, keepdims=True) + np.finfo('float32').eps)
    return discounted_rewards


# TODO: add TRPO/PPO
def optim_step(policy_optim, trajectory, method='REINFORCE'):
    if method == 'REINFORCE':
        rewards = trajectory.r
        prob = trajectory.prob

        discounted_rewards = discount_rewards(rewards, normalize=True)

        policy_loss = []
        # use log to scale loss https://towardsdatascience.com/policy-gradient-methods-104c783251e0
        # TODO: can also try out averaging probs
        for p, r in zip(prob, discounted_rewards):
            policy_loss.append(- torch.log(p) * r)
            # policy_loss.append(  p * r)

        policy_optim.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()  # retain_graph=True)     # TODO: retain_graph only needed when we have 2 losses/ two outputs
        policy_optim.step()


class PPO():
    def __init__(self, policy, value_fn):
        self.policy = policy
        self.value_fn = value_fn

    def train(self, env, epochs):
        start_time = time.time()

        self.policy.train()
        self.value_fn.train()
        policy_optim = optim.RMSprop(self.policy.parameters(), lr=POLICY_LEARNING_RATE)  # , momentum=MOMENTUM)
        value_fn_optim = optim.RMSprop(self.value_fn.parameters(), lr=VALUE_FN_LEARNING_RATE)  # , momentum=MOMENTUM)

        # TODO: don't save everything, only average rewards for plotting and batch of current epoch
        statistics = SummaryStatistics(TRAJECTORY_LEN, EPISODES)

        # maximize the expected total discounted reward accumulated during a single trial rather than a single episode
        try:
            for epoch in range(epochs):

                for episode in range(EPISODES):
                    # for each episode, new initial state
                    s, a, r, d = env.reset(), 0, 0, 0

                    trajectory = Trajectory()

                    for t in range(TRAJECTORY_LEN):
                        a_logits = self.policy(*make_tensor(s, a, r, d, env))
                        a = torch.distributions.Categorical(logits=a_logits).sample().item()
                        new_s, r, d, _ = env.step(a)

                        if episode < 0:
                            print(a, a_logits.tolist(), r)
                        if isinstance(env, BanditEnv):
                            d = (t == TRAJECTORY_LEN - 1)
                        # print(f'Episode: {episode} | step: {t}', s, a, a_logits.tolist(), r, d)

                        trajectory.record(s, a, r, d, a_logits)
                        s = new_s
                        if d:
                            break

                    # record time, avg/var of action and reward for one trajectory
                    statistics.record(time.time() - start_time, trajectory)

                # avg reward per episode in last epoch
                avg_reward_per_episode = np.mean(statistics.r_sum[-EPISODES])
                statistics.add_epochal(episode_r_avg=avg_reward_per_episode)

                # after one epoch (episodes * trajectories), update nets
                policy_loss, value_fn_loss = self.optim_step(policy_optim, value_fn_optim, statistics, env)

                if epoch % PRINT_EVERY == 0:
                    # TODO: fix seconds printing
                    print(f'Epoch: {epoch} | '
                          f'Time: {np.sum(statistics.time):.0f}s, {PRINT_EVERY // np.sum(statistics.time[-PRINT_EVERY:])}ep/s | '
                          f'Reward avg: {np.mean(statistics.r_avg[-PRINT_EVERY:]):.2f} | '
                          f'Action std: {statistics.a_std[-1]:.4f}'
                          )
                # logger.log({"policy_loss": policy_loss, "value_loss": value_fn_loss, "reward": avg_reward_per_episode,
                #            "epoch": epoch, })
        except KeyboardInterrupt as e:
            print(e)
        finally:
            print(f'Finished training, took {time.time() - start_time} s')
            plt.plot(statistics.episode_r_avg)
            plt.show()
            # TODO: save plot and statistics data pkl

    def optim_step(self, policy_optim, value_fn_optim, statistics, env):
        batch_len = statistics.episodes
        states = [traj.s for traj in statistics.trajectories[-batch_len:]]
        actions = [torch.tensor(traj.a) for traj in statistics.trajectories[-batch_len:]]
        a_logits = [torch.stack(traj.a_logits) for traj in statistics.trajectories[-batch_len:]]
        rewards = [traj.r for traj in statistics.trajectories[-batch_len:]]
        dones = [traj.d for traj in statistics.trajectories[-batch_len:]]
        discounted_r = discount_rewards(rewards, True)
        with torch.no_grad():
            values = [self.value_fn(self.policy.encode_state(torch.tensor(s), batch=True)) for s in states]
            advantage = [torch.tensor(d) - v for d, v in zip(discounted_r, values)]

            # minimizing cross entropy is same as maximizing likelihood
            old_log_probs = [- F.cross_entropy(old_logits, action) for old_logits, action in
                                 zip(a_logits, actions)]

        for it in range(N_POLICY_UPDATES):
            policy_loss = torch.zeros(1, requires_grad=True)
            for i, traj in enumerate(statistics.trajectories[-batch_len:]):
                # feed in whole trajectory as batch
                # return of batch must be
                curr_logits = self.policy(*(torch.tensor(x) for x in [traj.s, traj.a, traj.r, traj.d]),
                                          batch=True)  # , batch=True)
                curr_log_prob = - F.cross_entropy(curr_logits, torch.tensor(traj.a))
                ratio = np.exp(curr_log_prob.detach() - old_log_probs[i])
                clipped_ratio = np.clip(ratio, (1 - CLIP_EPSILON) * ratio, (1 + CLIP_EPSILON) * ratio)
                policy_loss = policy_loss + torch.min(ratio * advantage[i], clipped_ratio * advantage[i]).mean()

            policy_loss = policy_loss / batch_len
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

        for it in range(N_VALUE_UPDATES):
            value_loss = torch.zeros(1, requires_grad=True)
            for i in range(batch_len):
                v = self.value_fn(self.policy.encode_state(torch.tensor(states[i]))).view(-1)
                value_loss = value_loss + F.mse_loss(v, torch.tensor(discounted_r[i]))

            value_loss = value_loss / batch_len
            value_fn_optim.zero_grad()
            value_loss.backward()
            value_fn_optim.step()

        return policy_loss.item(), value_loss.item()


if __name__ == '__main__':
    init()
    unit_test()
    env = load_env('bandits')
    policy = RNNPolicy(env, N_HIDDEN_POLICY, N_LAYERS).to(device).to(dtype)
    value_fn = nn.Sequential(
        nn.Linear(policy.state_emb_size, N_HIDDEN_VALUE_FN),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(N_HIDDEN_VALUE_FN, 1),
    ).to(device).to(dtype)
    model = PPO(policy, value_fn)
    model.train(env, EPOCHS)
