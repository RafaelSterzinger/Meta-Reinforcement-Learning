# Load environment with corresponding params
import gym
import numpy as np
from gym_bandits.bandit import BanditEnv

RNG = np.random.default_rng(seed=69)
ENVS_DICT = {'bandits': lambda: BanditEnv(np.random.uniform(size=2), [1, 1])}

def load_env(name: str):
    np.random.seed(RNG.integers(0, 100, 1)[0])
    env = ENVS_DICT[name]()
    env.reset()
    return env
