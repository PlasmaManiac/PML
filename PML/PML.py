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
        self.network = net

    def forward_propagate(self, input_data, Y):
        A = input_data
        L = len(self.network.layers)
        for l in range(L - 1):
            self.layers[l].linear_forward(A)
            A = self.layers[l].sigmoid_forward()
        cost = self.compute_cost(Y,A)
        return cost


    def backward_propogate(self):
        L = len(self.network.layers)
        for l in reversed(range(L - 1)):
            self.layers[l].linear_backward()
            self.layers[l].sigmoid_backwards()

    def compute_cost(self, Y, AL):
        # TODO: Update this to be more versatile
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
        self.A_prev = None

        self.dZ = None
        self.dW = None
        self.db = None

        self.dA = None

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

    def linear_forward(self, input_A):
        self.A_prev = input_A
        self.Z = np.dot(self.W.T, input_A) + self.b
        return self.Z

    def relu_forward(self):
        return

    def sigmoid_forward(self):
        self.A = 1/(1 + np.exp(-self.Z))
        return self.A

    def sigmoid_backward(self, dA):
        sig_deriv = np.exp(self.Z) / np.power((np.exp(self.Z) + 1), 2)
        self.dZ = dA * sig_deriv

        return self.dZ

    def relu_backward(self):
        return

    def linear_backward(self, dZ):
        """

        :param dZ:
        :return: The gradients of the layer
        """
        # Need W_prev, dZ_prev
        # dZ = A - Y for output layer
        # dZ = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        m = self.A_prev.shape[1]

        # print("dZ Shape:", dZ.shape)
        # print("dA_prev shape:", self.A_prev.shape)
        # print("W shape:", self.W.shape)

        self.dW = (1 / m) * np.dot(dZ, self.A_prev.T)
        self.db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        return self.dW, self.db, dA_prev
