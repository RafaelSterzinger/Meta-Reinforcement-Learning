import argparse
from envs.init import load_env


def main(env: str, algo: str):
    env = load_env(env)

    reward = 0
    for _ in range(1000):
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
