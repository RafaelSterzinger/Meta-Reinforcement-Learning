from envs.init import load_env


def main():
    env = load_env('CartPole-v0')
    env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

    env.close()

if __name__ == '__main__':
    # read args for env and algo
    pass