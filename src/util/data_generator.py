# Generate training and test data with TRPO algorithm
# https://stable-baselines.readthedocs.io/en/master/modules/trpo.html
from stable_baselines3 import PPO

from src.envs.init import load_env


def generate_data(ENV: str):
    env = load_env(ENV)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=25000)
    reward = 0

    o = env.reset()
    for _ in range(100):
        action, _states = model.predict(o, deterministic=True)
        o, r, done, info = env.step(action)
        env.render()
        reward += r
        if done:
            o = env.reset()
    print(reward)
    env.close()
