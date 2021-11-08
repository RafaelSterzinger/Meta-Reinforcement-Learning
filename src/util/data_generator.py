# Generate training and test data with TRPO algorithm
# https://stable-baselines.readthedocs.io/en/master/modules/trpo.html
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import pickle as pkl
from src.envs.env_init import load_env
from src.envs.env import ENV_CONFIGS


def generate_data(ENV: str):
    config = ENV_CONFIGS[ENV]['generation']
    samples_per_iter = config['samples_per_iter']
    train_tasks = config['num_of_train_tasks']
    test_tasks = config['num_of_test_tasks']
    tasks = generate_trajectories(ENV, samples_per_iter, train_tasks)
    f = open(f'../data/{ENV}/train.pkl', 'wb')
    pkl.dump(tasks, f)
    f.close()
    tasks = generate_trajectories(ENV, samples_per_iter, test_tasks)
    f = open(f'../data/{ENV}/test.pkl', 'wb')
    pkl.dump(tasks, f)
    f.close()


def generate_trajectories(ENV, samples_per_iter, num_of_tasks):
    tasks = []
    for _ in tqdm(range(num_of_tasks)):
        env = load_env(ENV)
        model = PPO('MlpPolicy', env, verbose=0, n_steps=samples_per_iter)
        callback = RolloutCallback()
        model.learn(samples_per_iter * num_of_tasks, callback=callback)
        tasks.append(callback.get_rollouts())
        env.close()
    return tasks


class Transition():
    def __init__(self, state0, action, reward, state1):
        self.state0 = state0
        self.action = action
        self.reward = reward
        self.state1 = state1


class RolloutCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RolloutCallback, self).__init__(verbose)
        self.rollouts = []

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts.append(self.model.rollout_buffer)

    def _on_training_end(self) -> None:
        pass

    def get_rollouts(self):
        states0 = []
        actions = []
        rewards = []
        for rollout in self.rollouts:
            states0 = np.concatenate((states0, rollout.observations.flatten()))
            actions = np.concatenate((actions, rollout.actions.flatten()))
            rewards = np.concatenate((rewards, rollout.rewards.flatten()))
        states1 = states0[1:]
        states0 = states0[:-1]
        actions = actions[:-1]
        rewards = rewards[:-1]
        return [[s0, a, r, s1] for s0, a, r, s1 in zip(states0, actions, rewards, states1)]
