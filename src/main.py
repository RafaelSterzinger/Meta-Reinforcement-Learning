import argparse

from src.algos.algo_init import load_algo
from src.envs.env_init import load_env
from src.envs.macaw_gym import GymMacaw
from util.data_generator import generate_data
from os.path import exists


def main(ENV: str, ALGO: str):
    if ALGO == 'macaw':
        # check if training data has been generated
        if not exists(f'../data/{ENV}.pkl'):
            generate_data(ENV)

    train(ENV, ALGO)
    # test(ENV, ALGO)


def train(ENV: str, ALGO: str):
    env = GymMacaw(ENV, train=True) if ALGO == 'macaw' else load_env(ENV)
    model = load_algo(ALGO, env.action_space, env.observation_space)

    for _ in range(20000):
        reward = 0
        for _ in range(100):
            env.render()
            o, r, _, _ = env.step(env.action_space.sample())
            reward += r
        print(reward)
        env.close()


def test(ENV: str, ALGO: str):
    for _ in range(300):
        env = load_env(ENV)
        reward = 0
        for _ in range(100):
            env.render()
            o, r, _, _ = env.step(env.action_space.sample())
            reward += r
        print(reward)
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose an environment and an algorithm.')
    parser.add_argument('--env', type=str, choices=['bandits'], required=True)
    parser.add_argument('--alg', type=str, choices=['r', 'r2', 'maml', 'taml', 'macaw'], required=True)
    args = parser.parse_args()
    main(args.env, args.alg)
