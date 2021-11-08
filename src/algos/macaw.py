import torch
from torch import nn


class MACAW(nn.Module):
    def __init__(self, action_space: int, observation_space: int):
        super(MACAW, self).__init__()
        self.alpha1 = 0.9
        self.alpha2 = 0.9
        self.eta1 = 0.9
        self.eta2 = 0.9
        self.body = nn.Sequential(
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.action_head = nn.Sequential(
            # action as input
            nn.Linear(128 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, o, a):
        temp = self.body(o)
        act = self.action_head(temp)
        adv = self.advantage_head(torch.cat((temp, torch.Tensor(a))))
        return (act, adv)
