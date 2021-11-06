import argparse
from envs.init import load_env
from util.data_generator import generate_data
from os.path import exists


def main(ENV: str, ALGO: str):
    if ALGO == 'macaw':
        # check if training data has been generated
        if exists(f'data/{ENV}.pkl'):
            pass
        else:
            generate_data(ENV)
    return

    train(ENV, ALGO)
    test(ENV, ALGO)


def train(ENV: str, ALGO: str):
    for _ in range(20000):
        env = load_env(ENV)
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
