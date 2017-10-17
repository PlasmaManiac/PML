# Python Machine Learning Module
# A project made for the purpose practising Machine Learning Concepts
# Built in reference to techniques shown in Andrew Ng's Deep Learning Course
# Project by: Luke Martin

import numpy as np
# import matplotlib.pyplot as plt


class Network:
    """

    """
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

class Optimizer:
    """

    """
    def __init__(self, net):
        self.placeholder = None
        self.network = net

    def forward_propagate(self, input_data):
        A = input_data
        L = len(self.network.layers)
        for l in range(L - 1):
            self.layers[l].linear_forward(A)
            A = self.layers[l].sigmoid_forward()
        cost = self.layers[L - 1]
        return cost


    def backward_propogate(self):
        for layer in


    def compute_cost(self, Y, AL): ##TODO: Update this to be more versitile
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y)))
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return cost


class Layer:
    def __init__(self):
        self.n_x = None
        self.n_h = None
        self.n_h = None

        self.layer_type = None

        self.W = None
        self.b = None

        self.A = None
        self.Z = None

        self.dZ = None
        self.dW = None
        self.db = None

        self.activation_function = None

    def initialize(self, n_x, n_h, n_y, layer_type = "fc"):
        """
        :param n_x: size of input layer
        :param n_h: size of hidden layer
        :param n_y: size of ouput layer
        :param layer_type: the type of the layer
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_h = n_y

        self.layer_type = layer_type

        self.W = np.random.randn(n_h, n_x) * 0.01
        self.b = np.zeros((n_h,1))

        assert (self.W.shape == (n_h, n_x))
        assert (self.b.shape == (n_h, 1))

    def linear_forward(self, input_vector):
        self.A = input_vector
        self.Z = np.dot(self.W.T, input_vector) + self.b
        return self.Z

    def relu_forward(self):
        return

    def sigmoid_forward(self):
        self.A = 1/(1 + np.exp(self.Z))
        return self.A

    def relu_backward(self):
        return

    def sigmoid_backward(self, m, A, dZ):
        """

        :param m:
        :param A: The previous layer, ie: this layer's input
        :return: The gradients of the layer
        """
        # Need W_prev, dZ_prev
        # dZ = A - Y for output layer
        # dZ = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))

        self.dW = (1 / m) * np.dot(self.dZ, A.T)
        self.db = (1 / m) * np.sum(self.dZ, axis=1, keepdims=True)

        return self.dW, self.db