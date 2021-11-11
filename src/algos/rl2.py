import time

from src.algos.rnn_policy import RNNPolicy, PPO
import torch.nn as nn

from src.envs.init import load_env


class RL2():
    def __init__(self, env_name, N_HIDDEN_POLICY, N_HIDDEN_VALUE_FN, N_LAYERS):
        self.env_name = env_name
        env = load_env(env_name)
        self.policy = RNNPolicy(env, N_HIDDEN_POLICY, N_LAYERS).to(device).to(dtype)
        self.value_fn = nn.Sequential(
            nn.Linear(self.policy.state_emb_size, N_HIDDEN_VALUE_FN),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_VALUE_FN, 1),
        ).to(device).to(dtype)

    def meta_train(self, n_trials):
        model = PPO(self.policy, self.value_fn)
        for t in range(n_trials):
            # for each trial, separate MDP is drawn
            env = load_env(self.env_name)         # TODO add params for specific env
            # reset hidden state
            model.policy.init_hidden()
            model.train()

    # TODO
    def meta_test(self):
        model = PPO(self.policy, self.value_fn)
        # load model (model.load())

