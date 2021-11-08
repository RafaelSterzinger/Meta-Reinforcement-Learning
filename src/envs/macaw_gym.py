import numpy as np
from gym import Env
from src.envs.env_init import ENVS_DICT
import pickle as pkl
from sklearn.model_selection import train_test_split


class GymMacaw(Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    # add callbacks for training/testing
    def __init__(self, ENV: str, train: bool):
        super(GymMacaw, self).__init__()  # Define action and observation space
        env = ENVS_DICT[ENV]()
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        f = open(f'../data/{ENV}/{"train" if train else "test"}.pkl', 'rb')
        self._data = pkl.load(f)
        self.size = len(self._data)
        f.close()
        self.cur_sets = [(train_test_split(np.random.choice(task, size=400, replace=False), train_size=0.6)) for task in
                         self._data]
        self.train_iter = iter([self.cur_sets[i][0]] for i in range(self.size))
        self.test_iter = iter([self.cur_sets[i][1]] for i in range(self.size))

    def step(self, action):
        return

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass
