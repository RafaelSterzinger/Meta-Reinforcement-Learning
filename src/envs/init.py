# Load environment with corresponding params
import gym
import gym_bandits

ENVS_DICT = {'bandits': 'BanditTenArmedRandomFixed-v0'}


def load_env(name: str):
    env = gym.make(ENVS_DICT[name])
    env.reset()
    return env
