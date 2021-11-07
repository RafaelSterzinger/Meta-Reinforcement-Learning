import torch
import torch.nn as nn
import torch.nn.functional as F

# Policy Optimization
# Trust Region Policy Optimization (use a library)
from gym_bandits.bandit import BanditEnv

class RNNPolicy(nn.Module):
    def __init__(self, env, hidden_dim, n_layers):
        super(RNNPolicy, self).__init__()
        self.env = env
        self.state_emb_size = self.env.observation_space.n
        self.action_emb_size = self.env.action_space.n
        self.input_dim = self.state_emb_size + self.action_emb_size + 2  # + reward + done
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, n_layers, bias=True)
        self.fc = nn.Linear(self.hidden_dim, self.action_emb_size)

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
        self.hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        self.cell = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        # TODO: to.(dtype).to.device(xxx) ?
        # ReLU as hidden activation

    # bandits: constant 0 embedding for states, one-hot encode actions
    # tabular MDP: one-hot encode states and actions
    # visual: TODO
    def encode_input(self, state, action, reward, done):
        if isinstance(self.env, BanditEnv):
            state = 0
        # TODO elif: tabular MDP

        action = F.one_hot(action, num_classes=self.action_emb_size)
        # TODO: to.(dtype).to.device(xxx) ?
        return torch.cat((state, action, reward, done), dim=1)
        # TODO: to.(dtype).to.device(xxx) ?

    # input (state, action, reward, done)
    # embedded with function for input of GRU
    # fully connected
    # softmax dist over actions
    def forward(self, state, action, reward, done):
        input = self.encode_input(state, action, reward, done)

        out, (self.hidden, self.cell) = self.rnn(input, (self.hidden, self.cell))
        # TODO: call ReLU on hidden too?
        out = self.fc(F.relu(out))

        # TODO: to.(dtype).to.device(xxx) ?
        self.hidden.detach_()

        prob = F.softmax(out) # dim=2 ? view(-1)
        return prob


    # TODO: do not forget to set to eval() mode during testing
