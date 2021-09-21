import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import gym
import os.path

from pso import Swarm
import help_functions

env_name = 'CartPole-v1'
env = gym.make(env_name)

# PSO parameters
swarm_size = 100
var_range = 1
max_speed_pso = 10
c1 = 0.5
c2 = 0.3
inertia_weight = 0.20
inertia_decay = 0.99
inertia_lower = 0.02

# Network parameters
nr_inputs = 4
nr_neurons = np.array([7, 13, 7, 3], dtype=np.intc)
# nr_neurons = np.array([7, 13], dtype=np.intc)
nr_outputs = 1

# Other parameters
nr_generations = 500

if nr_neurons.size != 0:
    n_vars = nr_inputs*nr_neurons[0]
    for i in range(1, nr_neurons.size):
        n_vars += nr_neurons[i] * nr_neurons[i-1]
    n_vars += nr_neurons[-1] * nr_outputs + np.sum(nr_neurons) + nr_outputs
else:
    n_vars = (1 + nr_inputs) * nr_outputs

# Init
test = True
best_network = None
last_best_network = None
best_fitness = -np.inf
swarm = Swarm(swarm_size, var_range, n_vars, max_speed=max_speed_pso, c1=c1, c2=c2,
              inertia_weight=inertia_weight, inertia_decay=inertia_decay, inertia_lower=inertia_lower)
if test == False:
    for iGeneration in range(nr_generations):
        # Get networks and initialize positions
        networks = swarm.decode_population(nr_inputs, nr_neurons, nr_outputs)
        objective_values = np.ones(swarm_size) * np.inf

        for iNetwork in range(swarm_size):
            # message = "Generation {}: {}/{}".format(iGeneration, iNetwork, swarm_size)
            done = False
            state = env.reset()
            state = np.array([0,0,0,0])
            network = networks[iNetwork]
            sum_rewards = np.array([0, 0])
            while not done:
                action = help_functions.get_action(network=network, inputs=state)
                state, env_reward, done, info = env.step(action)
                reward, terminal_state_reached = help_functions.calculate_reward(state)
                if terminal_state_reached:
                    done = True
                sum_rewards[0] += reward
                sum_rewards[1] += env_reward
            particle_reward = swarm.evaluate_particle(sum_rewards)
            objective_values[iNetwork] = particle_reward

        gen_best_fitness = np.max(objective_values)
        if iGeneration % (nr_generations / 100) == 0:
            print(f"Generation {iGeneration} best fitness: {gen_best_fitness} abs best: {best_fitness}")

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_network = networks[np.argmax(objective_values)]
        
        if iGeneration == nr_generations:
            last_best_network = networks[np.argmax(objective_values)]

        # Update swarm
        swarm.update_best_positions(objective_values)
        swarm.update_swarm_best(objective_values)
        swarm.update_velocities()
        swarm.update_positions()

    np.save('best_network', swarm.encode_particle(best_network, n_vars))
    state = np.array([0, -0.1, 0.1, 0])
    state = env.reset()
    env.state = state
    done = False
    steps = 0
    while not done:
        action = help_functions.get_action(network=best_network, inputs=state)
        state, reward, done, info = env.step(action)
        steps += 1
        env.render()
    print(f"Steps taken: {steps}")
else:
    best_network = np.load("best_working_network.npy")
    best_network = swarm.decode_particle(best_network, nr_of_inputs=nr_inputs, nr_of_neurons=nr_neurons, 
                        nr_of_outputs=nr_outputs)
    state = np.array([0, -0.1, 0.1, 0])
    env.reset()
    env.state = state
    done = False
    steps = 0
    while not done:
        action = help_functions.get_action(network=best_network, inputs=state)
        state, reward, done, info = env.step(action)
        steps += 1
        env.render()
    print(f"Steps taken: {steps}")