# Python Machine Learning Module
# A project made for the purpose practising Machine Learning Concepts
# Built in reference to techniques shown in Andrew Ng's Deep Learning Course
# Inspired by Andrej Karpathy's JS library
# Project by: Luke Martin

import numpy as np
import matplotlib.pyplot as plt


class Network:
    """

    """
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagate(self, input_data):
        A = input_data
        L = len(self.layers)
        for l in range(L):
            # print("Layer: ", l)
            A = self.layers[l].forward_propagate(A)
            # print("Z" + str(l + 1) + "= ", Z)
            # print("A" + str(l + 1) + "= ", A)

        return A



    def back_propagate(self, labels, AL):
        L = len(self.layers)
        dA = -1 * (np.divide(labels, AL) - np.divide(1 - labels, 1 - AL))
        dZ = AL - labels
        dA = self.layers[L - 1].linear_backward(dZ)
        for l in reversed(range(L - 1)):
            self.layers[l].back_propagate(dA)
            # dZ = self.layers[l].sigmoid_backward(dA) # Requires dA
            # _,_, dA = self.layers[l].linear_backward(dZ)  # Require dZ

    def predict(self, input_data):
        AL = self.forward_propagate(input_data)
        prediction = 1*(AL > 0.5)
        return prediction

    def plot_decision_boundary(self, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        print("x min/max = ", x_min, x_max)
        print("y min/max = ", y_min, y_max)
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        print("xx shape = ", xx.shape)
        print("yy shape = ", yy.shape)

        mesh_points = np.array([xx.ravel(), yy.ravel()])

        # Predict the function value for the whole gid
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.show()


class Optimizer:
    """

    """
    def __init__(self, net):
        self.network = net
        self.learning_rate = 1.2

    def train(self, input_data, labels, iterations=10000):
        for i in range(iterations):

            AL = self.network.forward_propagate(input_data)
            cost = self.compute_cost(labels, AL)
            self.network.back_propagate(labels, AL)
            self.update_parameters()
            if i % 1000 is 0:
                print("iteration: ",i," cost = ", cost)

    def compute_cost(self, Y, AL):
        """

        :param Y:
        :param AL:
        :return:
        """
        # TODO: Update this to be more versatile
        m = Y.shape[1]
        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))
        cost = -1 * (1 / m) * np.sum(logprobs)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return cost

    def update_parameters(self):
        L = len(self.network.layers)
        for l in range(L):
            self.network.layers[l].W -= self.network.layers[l].dW * self.learning_rate
            self.network.layers[l].b -= self.network.layers[l].db * self.learning_rate


class Layer:
    def __init__(self, act_func):
        self.n_x = None
        self.n_h = None

        self.act_func = act_func

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

    def initialize(self, n_x, n_h, layer_type="fc"):
        """
        :param n_x: size of input layer
        :param n_h: size of hidden layer
        :param n_y: size of ouput layer
        :param layer_type: the type of the layer
        """
        self.n_x = n_x
        self.n_h = n_h

        self.layer_type = layer_type

        self.W = np.random.randn(n_h, n_x) * 0.01
        self.b = np.zeros((n_h,1))

        assert (self.W.shape == (n_h, n_x))
        assert (self.b.shape == (n_h, 1))

        return self.W, self.b

    def linear_forward(self, input_A):
        self.A_prev = input_A
        # print("W shape = ", self.W.shape)
        # print("Input shape = ", input_A.shape)
        self.Z = np.dot(self.W, input_A) + self.b
        # print("Output shape =  ", self.Z.shape)
        return self.Z

    def relu_forward(self):
        return

    def relu_backward(self):
        return

    def tanh_forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def tanh_backward(self, dA):
        self.dZ = dA * (1 - np.power(self.A, 2))
        return self.dZ

    def sigmoid_forward(self, Z):
        self.A = 1/(1 + np.exp(-Z))
        return self.A

    def sigmoid_backward(self, dA):
        sig_deriv = np.exp(self.Z) / np.power((np.exp(self.Z) + 1), 2)
        self.dZ = dA * sig_deriv
        return self.dZ

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

        # print("dW = ", self.dW)
        # print("db = ", self.db)

        dA_prev = np.dot(self.W.T, dZ)

        return dA_prev, self.dW, self.db

    def back_propagate(self, dA):
        dZ = None
        if self.act_func is "sigmoid":
            dZ = self.sigmoid_backward(dA)
        elif self.act_func is "relu":
            dZ = self.relu_backward()
        elif self.act_func is "tanh":
            dZ = self.tanh_backward(dA)

        dA_prev, dW, db = self.linear_backward(dZ)
        return dA_prev

    def forward_propagate(self, prev_A):
        Z = self.linear_forward(prev_A)
        A = None
        if self.act_func is "tanh":
            A = self.tanh_forward(Z)
        elif self.act_func is "sigmoid":
            A = self.sigmoid_forward(Z)
        elif self.act_func is "relu":
            A = self.relu_forward()

        return A