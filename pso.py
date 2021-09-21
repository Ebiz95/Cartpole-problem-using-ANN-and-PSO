from typing import Any, List
import numpy as np
from objective_function import objective_function
from ann import NeuralNetwork


class Swarm:
    def __init__(self, swarm_size, var_range,
                 n_variables, c1=2, c2=2,
                 inertia_weight=1.4, inertia_decay=0.99,
                 max_speed=10, inertia_lower=0.3):

        self.swarm_size = swarm_size
        self.positions = np.random.uniform(low=-var_range, high=var_range, size=(swarm_size, n_variables))
        self.velocities = np.random.uniform(low=-var_range, high=var_range, size=(swarm_size, n_variables))
        self.particle_best = np.copy(self.positions)
        self.swarm_best = self.positions[0]     # Placeholder
        self.swarm_best_value = np.inf
        self.var_range = var_range
        self.n_variables = n_variables
        self.c1 = c1
        self.c2 = c2
        self.inertia_weight = inertia_weight
        self.inertia_decay = inertia_decay
        self.max_speed = max_speed
        self.inertia_lower = inertia_lower
        self.objective_values = np.empty(swarm_size)     # Placeholder

    @staticmethod
    def evaluate_particle(reward: Any) -> float:
        """
        Returns the fitness of the current particle based on the reward.
        """
        return objective_function(reward)

    def update_best_positions(self, new_objective_values: np.array) -> None:
        """
        Updated the best position of each particle based on the new objective values.
        """
        is_larger = np.nonzero(new_objective_values > self.objective_values)[0]
        self.particle_best[is_larger] = self.positions[is_larger]
    
    def update_swarm_best(self, objective_values: np.array) -> None:
        """
        Updates the best particle in the swarm
        """
        for i in range(np.size(objective_values)):
            if objective_values[i] > self.swarm_best_value:
                self.swarm_best_value = objective_values[i]
                self.swarm_best = np.copy(self.positions[i])

    def restrict_velocities(self) -> None:
        """
        Restriction of the speed of all individual particles.
        """
        velocities = self.velocities
        max_speed = self.max_speed
        speeds = np.linalg.norm(velocities, ord=2, axis=1, keepdims=True)
        for i in (np.nonzero(speeds > max_speed)[0]):
            self.velocities[i] *= max_speed / speeds[i]

    def update_velocities(self) -> None:
        """
        Update of all velocities - call this before updating positions.
        """
        swarm_size = self.swarm_size
        positions = self.positions
        q = np.random.uniform(0, 1, (swarm_size, 1))
        r = np.random.uniform(0, 1, (swarm_size, 1))

        self_terms = self.c1 * np.multiply(self.particle_best - positions, q)
        swarm_term = self.c2 * np.multiply(self.swarm_best - positions, r)

        self.velocities = self.inertia_weight * self.velocities + self_terms + swarm_term
        self.restrict_velocities()
        self.update_inertia()

    def update_positions(self) -> None:
        """
        Updates the positions of the particles in the swarm.
        """
        self.positions = self.positions + self.velocities

    def update_inertia(self) -> None:
        """
        Updates the inertia weight.
        """
        if self.inertia_weight > self.inertia_lower:
            self.inertia_weight *= self.inertia_decay

    def encode_population(self, neural_networks: List[NeuralNetwork]) -> None:
        """
        Encodes the neural networks as particles in the swarm.
        """
        swarm = np.copy(self.positions)
        for i, network in enumerate(neural_networks):
            particle = Swarm.encode_particle(network, self.n_variables)
            swarm[i,:] = particle
        self.positions = swarm

    def decode_population(self, nr_of_inputs: int, nr_of_neurons: np.array, nr_of_outputs: int) -> List[NeuralNetwork]:
        """
        Decodes the particles in the swarm to neural networks. nr_of_neurons is a list that represents the number of 
        neurons in each hidden layer.
        """
        networks = []
        for particle in self.positions:
            temp_network = Swarm.decode_particle(particle, nr_of_inputs, nr_of_neurons, nr_of_outputs)
            networks.append(temp_network)
        return networks

    @staticmethod
    def encode_particle(neural_network: NeuralNetwork, n_variables: int) -> np.array:
        """
        Given a neural network, return an encoded version of the network in the shape of a numpy array.
        """
        particle = np.empty(n_variables)
        start_index = 0
        for layer in neural_network.weights:
            for weight in layer:
                particle[start_index : start_index + weight.size] = weight #.flatten()
                start_index += weight.size
        for layer in neural_network.biases:
            for threshold in layer:
                particle[start_index : start_index + threshold.size] = threshold
                start_index += threshold.size
        return particle

    # nr_of_neurons is a list that represents nr of neurons in each hidden layer
    @staticmethod
    def decode_particle(particle: np.array, nr_of_inputs: int, nr_of_neurons: np.array, nr_of_outputs: int) \
                             -> NeuralNetwork:
        """
        Given a particle (Numpy array), returns a decoded version in the shape of a NeuralNetwork.
        """
        network = NeuralNetwork()
        network.add_input_layer(nr_of_inputs)
        nr_of_layers = nr_of_neurons.size
        if nr_of_layers != 0:
            weight_range = 0
            if nr_of_layers != 1:
                for i in range(nr_of_layers - 1):
                    weight_range += nr_of_neurons[i]*nr_of_neurons[i+1]
            weight_range += nr_of_inputs*nr_of_neurons[0] + nr_of_neurons[-1]*nr_of_outputs
            bias_range = weight_range + sum(nr_of_neurons) + nr_of_outputs
            weights = particle[:weight_range]
            biases = particle[weight_range:bias_range]

            start_index = 0
            for layer in range(nr_of_layers):
                if layer == 0:
                    temp_weights = weights[:nr_of_inputs*nr_of_neurons[0]]
                    temp_weights = temp_weights.reshape(nr_of_neurons[layer], nr_of_inputs)
                    temp_bias = biases[:nr_of_neurons[0]]
                    temp_bias = temp_bias.reshape(temp_bias.size, 1)
                    start_index += temp_weights.size
                else:
                    temp_weights = weights[start_index : start_index + nr_of_neurons[layer]*nr_of_neurons[layer - 1]]
                    temp_weights = temp_weights.reshape(nr_of_neurons[layer], nr_of_neurons[layer - 1])
                    temp_bias = biases[np.sum(nr_of_neurons[:layer]) : np.sum(nr_of_neurons[:layer + 1])]
                    temp_bias = temp_bias.reshape(temp_bias.size, 1)
                    start_index += temp_weights.size
                network.add_preinitialized_dense_layer(temp_weights, temp_bias)
            temp_weights = weights[-nr_of_neurons[-1]*nr_of_outputs:]
            temp_weights = temp_weights.reshape(nr_of_outputs, nr_of_neurons[-1])
            temp_bias = biases[-nr_of_outputs:]
            temp_bias = temp_bias.reshape(nr_of_outputs, 1)
            network.add_preinitialized_output_layer(temp_weights, temp_bias)
        else:
            weight_range = nr_of_inputs*nr_of_outputs
            weights = particle[:weight_range].reshape(nr_of_inputs, nr_of_outputs)
            biases = particle[weight_range:].reshape(nr_of_outputs, 1)
            network.add_preinitialized_output_layer(weights, biases)
        return network
