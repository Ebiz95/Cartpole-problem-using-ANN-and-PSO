from numpy import tanh, cosh, power, dot, zeros, \
    array, genfromtxt, delete, sign, multiply, matmul, linspace, copy
from numba.typed import List
from numba import njit, jit


class NeuralNetwork:
    def __init__(self):
        self.weights = List()
        self.biases = List()
        self.history = None
        self.inputs = None

    def add_input_layer(self, nr_of_inputs):
        self.inputs = nr_of_inputs

    def add_preinitialized_dense_layer(self, weights, thresholds):
        self.biases.append(thresholds)
        self.weights.append(weights)

    def add_preinitialized_output_layer(self, weights, thresholds):
        self.biases.append(thresholds)
        self.weights.append(weights)

    # "inputs" can either be x from the input layer or V from a previous layer.
    # The function returns a column vector.
    def weighted_sum(self, inputs, layer):
        return matmul(self.weights[layer], inputs) + self.biases[layer]

    # Activation is tanh, with derivative 1/cosh^2.
    @staticmethod
    def activation_function(x):
        return tanh(x)

    @staticmethod
    @njit
    def forward_prop(network_input, weights, biases):
        nr_of_layers = len(biases)
        network_output = copy(network_input)
        for i in range(nr_of_layers):
            network_output = dot(weights[i], network_output) + biases[i]
            if i != nr_of_layers-1:
                network_output = tanh(network_output)
        return network_output
