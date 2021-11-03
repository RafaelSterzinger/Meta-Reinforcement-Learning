# Load environment with corresponding params
import gym

def load_env(name: str):
    env = gym.make(name)
    env.reset()
    return NotImplemented
