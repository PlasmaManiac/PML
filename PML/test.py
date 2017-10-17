import os.path
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import PML as ml


def main():
    test_layer = ml.Layer()

    test_layer.A = np.array([[ 1.62434536, -0.61175641],
                             [-0.52817175, -1.07296862],
                             [ 0.86540763, -2.3015387 ]])
    test_layer.W = np.array([ 1.74481176, -0.7612069, 0.3190391 ])
    test_layer.b = np.array([-0.24937038])

    print(test_layer.linear_forward())

    print(test_layer.sigmoid_forward())


if __name__ == "__main__": main()