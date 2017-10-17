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
        cost = 0
        A = input_data

        for layer in self.network.layers:
            layer.sigmoid_forward()

        return cost

    def backward_propogate(self):
        return

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

    def linear_forward(self):
        self.Z = np.dot(self.W.T, self.A) + self.b
        return self.Z

    def relu_forward(self):
        return

    def sigmoid_forward(self):
        self.A = 1/(1 + np.exp(self.Z))
        return self.A

    def relu_backward(self):
        return

    def sigmoid_backward(self, m, A):
        """

        :param m:
        :param A:
        :return: The gradients of the layer
        """
        self.dW = (1 / m) * np.dot(self.dZ, A.T)
        self.db = (1 / m) * np.sum(self.dZ, axis=1, keepdims=True)

        return self.dW, self.db