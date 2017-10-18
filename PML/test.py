import os.path
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import PML as ml

def linear_forward_test1():
    test_layer = ml.Layer()

    test_layer.A = np.array([[1.62434536, -0.61175641],
                             [-0.52817175, -1.07296862],
                             [0.86540763, -2.3015387]])
    test_layer.W = np.array([1.74481176, -0.7612069, 0.3190391])
    test_layer.b = np.array([-0.24937038])

    print(test_layer.linear_forward())

    print(test_layer.sigmoid_forward())


def linear_backward_test1():
    test_layer = ml.Layer()

    test_layer.W = np.array([0.3190391, -0.24937038, 1.46210794], ndmin=2)
    test_layer.b = np.array([-2.06014071])
    test_layer.A_prev = np.array([[-0.52817175, -1.07296862],
                                  [ 0.86540763, -2.3015387 ],
                                  [ 1.74481176, -0.7612069 ]])

    dW, db, dA_prev = test_layer.linear_backward(np.array([1.62434536, -0.61175641], ndmin=2))

    #TODO: Copy above code and make a proper test_layer constructor

    print("dA_prev = ", str(dA_prev))
    print("dW = ", str(dW))
    print("db = ", str(db))

    return dW, db, dA_prev

def sigmoid_backward_test1():
    Z = np.array([0.04153939, -1.11792545], ndmin=2)
    dA = np.array([-0.41675785, -0.05626683], ndmin=2)
    test_layer = ml.Layer()
    test_layer.Z = Z
    dZ = test_layer.sigmoid_backward(dA)
    print("[ML]    dZ  = ", str(dZ))
    print("[Truth] dZ  =  [[-0.10414453 -0.01044791]]")


def main():

    #linear_forward_test1()
    #linear_backward_test1()
    sigmoid_backward_test1()
    #TODO: write a compute cost test


if __name__ == "__main__": main()