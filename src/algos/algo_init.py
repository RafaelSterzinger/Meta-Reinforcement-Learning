# Load algorithm with corresponding params
from src.algos.macaw import MACAW

ALGOS_DICT = {'macaw': lambda action_space, observation_space: MACAW(action_space, observation_space)}


def load_algo(name: str, action_space: int, observation_space: int):
    return ALGOS_DICT[name](action_space, observation_space)
