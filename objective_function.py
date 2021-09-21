import numpy as np

def objective_function(reward: np.array):
    custom_reward = reward[0]
    env_reward = reward[1]
    reward = 0.9 * custom_reward + 0.1 * env_reward
    return reward