from ann import NeuralNetwork
import numpy as np

def get_action(network: NeuralNetwork, inputs: np.array):
    inputs = normalize_inputs(inputs)
    action = network.forward_prop(inputs, weights=network.weights, biases=network.biases)[0]
    if action > 0:
        action = 1
    else:
        action = 0
    return action

def normalize_inputs(inputs):
    nr_inputs = inputs.shape[0]
    cart_pos = np.copy(inputs[0])   # [-4.8, 4.8]
    a = -4.8
    b = 4.8
    c = 0
    d = 1
    cart_pos = shift_interval(a, b, c, d, cart_pos)

    cart_vel = np.copy(inputs[1])   # [-inf, inf]
    a = -1
    b = 1
    cart_vel = shift_interval(a, b, c, d, cart_vel)

    pole_ang = np.copy(inputs[2])   # [-0.418 rad (-24 deg), 0.418 rad (24 deg)]
    a = -0.418
    b = 0.418
    pole_ang = shift_interval(a, b, c, d, pole_ang)

    pole_vel = np.copy(inputs[3])   # [-inf, inf]
    a = -10
    b = 10
    pole_vel = shift_interval(a, b, c, d, pole_vel)
    inputs = np.array([cart_pos, cart_vel, pole_ang, pole_vel])
    inputs = inputs.reshape((nr_inputs, 1))
    return inputs

def shift_interval(a, b, c, d, val):
    return c + ((d - c)/(b - a))*(val - a)

def calculate_reward(state: np.array):
    cart_pos = np.copy(state[0])
    cart_vel = np.copy(state[1])
    pole_ang = np.copy(state[2])
    pole_vel = np.copy(state[3])

    terminal_state_reached = np.abs(pole_vel) > 0.7 or np.abs(cart_vel) > 2.4 or \
                                np.abs(cart_pos) > 2.4 or np.abs(pole_ang) > 0.209

    reward = -0.1

    if terminal_state_reached:
        reward = -10
    elif np.abs(pole_vel) < 0.25 and np.abs(cart_vel) < 1:
        reward = 0

    return reward, terminal_state_reached

