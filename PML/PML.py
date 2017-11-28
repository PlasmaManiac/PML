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
    def __init__(self, layers):
        self.layers = []
        self.L = 0

        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer):
        self.layers.append(layer)
        self.L += 1

    def forward_propagate(self, input_data):
        A = input_data
        L = self.L
        for l in range(L):
            A = self.layers[l].forward_propagate(A)

        return A

    def back_propagate(self, labels, AL):
        L = self.L
        # dA = -1 * (np.divide(labels, AL) - np.divide(1 - labels, 1 - AL))

        for l in reversed(range(L)):
            if l is L - 1:
                dZ = AL - labels
                # dZ = -(np.divide(labels, AL) - np.divide(1 - labels, 1 - AL))
                self.layers[l].dZ = dZ
            else:
                dZ = self.layers[l].back_propagate(self.layers[l+1].dZ, self.layers[l+1].W)

            self.layers[l].linear_backward(dZ)

        # for l in reversed(range(L)):
         # self.layers[l].back_propagate(dA)
            # dZ = self.layers[l].sigmoid_backward(dA) # Requires dA
            # _,_, dA = self.layers[l].linear_backward(dZ)  # Require dZ

    def predict(self, input_data):
        AL = self.forward_propagate(input_data)
        if(self.layers[self.L - 1].act_func is "sigmoid"):
            prediction = 1*(AL > 0.5)
        elif(self.layers[self.L - 1].act_func is "softmax"):
            prediction = AL
            # TODO: fix the softmax predictor

        # TODO: Change this based on the type of output layer. (softmax, Logistic)
        return prediction

    def plot_decision_boundary(self, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

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
    def __init__(self, net, learning_rate=0.1):
        self.network = net
        self.learning_rate = learning_rate
        self.costs = []

    def train(self, input_data, labels, iterations=10000):
        for i in range(iterations):

            AL = self.network.forward_propagate(input_data)
            cost = self.compute_cost(labels, AL)
            self.network.back_propagate(labels, AL)
            self.update_parameters()
            if i % 1000 is 0:
                print("iteration: ",i," cost = ", cost)
                self.costs.append(cost)

    def compute_cost(self, Y, AL):
        """

        :param Y:
        :param AL:
        :return:
        """
        # TODO: Update this to be more versatile
        m = Y.shape[1]
        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
        cost = -1 *np.sum(logprobs) * (1 / m)
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        return cost

    def update_parameters(self):
        L = len(self.network.layers)
        for l in range(L):
            self.network.layers[l].W = self.network.layers[l].W - self.network.layers[l].dW * self.learning_rate
            self.network.layers[l].b = self.network.layers[l].b - self.network.layers[l].db * self.learning_rate

    def plot_cost(self):
        plt.plot(np.array(self.costs))
        # plt.axis(0, 5, 0, 10000)
        plt.show()

    def compute_accuracy(self, X, labels):
        accuracy = 0.0

        AL = self.network.predict(X)

        predictions = AL.argmax(axis=0)

        print("Predictions = ", predictions)
        print("True labels = ", labels)
        print("Correct = ", predictions == labels)

        accuracy = sum(1*(predictions == labels)) / labels.shape[0]

        return accuracy

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

    def initialize(self, n_x, n_h):
        """
        :param n_x: size of input layer
        :param n_h: size of hidden layer
        :param n_y: size of ouput layer
        :param layer_type: the type of the layer
        """
        self.n_x = n_x
        self.n_h = n_h

        self.W = np.random.randn(n_h, n_x) * 0.1
        self.b = np.zeros((n_h, 1))

        assert (self.W.shape == (n_h, n_x))
        assert (self.b.shape == (n_h, 1))

        return self.W, self.b

    def linear_forward(self, input_A):
        self.A_prev = input_A
        self.Z = np.dot(self.W, input_A) + self.b

        return self.Z

    def relu_forward(self, Z):
        self.A = np.maximum(Z, 0)
        return self.A

    def relu_backward(self, dZ_prev, W_prev):
        dA = np.dot(W_prev.T, dZ_prev)
        self.dZ = 1*(dA > 1)
        return self.dZ

    def tanh_forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def tanh_backward(self, dZ_prev, W_prev):
        dA = np.dot(W_prev.T, dZ_prev)
        self.dZ = dA * (1 - np.power(self.A, 2))
        return self.dZ

    def sigmoid_forward(self, Z):
        self.A = 1/(1 + np.exp(-Z))
        return self.A

    def sigmoid_backward(self, dZ_prev, W_prev):

        dA = np.dot(W_prev.T, dZ_prev)
        sig_deriv = np.exp(self.Z) / np.power((np.exp(self.Z) + 1), 2)
        self.dZ = dA * sig_deriv
        return self.dZ

    def softmax_forward(self, Z):
        Z_e = np.exp(Z)
        self.A = Z_e / np.sum(Z_e, axis=0)
        return self.A

    def softmax_backward(self):
        return None

    def linear_backward(self, dZ):
        """

        :param dZ:
        :return: The gradients of the layer
        """
        # Need W_prev, dZ_prev
        # dZ = A - Y for output layer
        # dZ = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        m = self.A_prev.shape[1]

        self.dW = (1 / m) * np.dot(dZ, self.A_prev.T)
        self.db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        return self.dW, self.db

    def back_propagate(self, dZ_prev, W_prev):
        dZ = None
        if self.act_func is "sigmoid":
            dZ = self.sigmoid_backward(dZ_prev, W_prev)
        elif self.act_func is "relu":
            dZ = self.relu_backward(dZ_prev, W_prev)
        elif self.act_func is "tanh":
            dZ = self.tanh_backward(dZ_prev, W_prev)
        return dZ

    def forward_propagate(self, prev_A):
        Z = self.linear_forward(prev_A)
        A = None
        if self.act_func is "tanh":
            A = self.tanh_forward(Z)
        elif self.act_func is "sigmoid":
            A = self.sigmoid_forward(Z)
        elif self.act_func is "relu":
            A = self.relu_forward(Z)
        elif self.act_func is  "softmax":
            A = self.softmax_forward(Z)

        return A