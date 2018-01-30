import os.path
import sys
import sklearn
import sklearn.datasets
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import PML as ml


def linear_forward_test1():
    test_layer = ml.Layer()

    input_A = np.array([[1.62434536, -0.61175641],
                             [-0.52817175, -1.07296862],
                             [0.86540763, -2.3015387]])

    test_layer.W = np.array([1.74481176, -0.7612069, 0.3190391])
    test_layer.b = np.array([-0.24937038])

    print(test_layer.linear_forward(input_A))


def sigmoid_forward_test1():
    test_layer = ml.Layer()

    input_A = np.array([[-0.41675785, -0.05626683],
                        [-2.1361961,   1.64027081],
                        [-1.79343559, -0.84174737]])

    test_layer.W = np.array([ 0.50288142, -1.24528809, -1.05795222])
    test_layer.b = np.array([-0.90900761])

    Z = test_layer.linear_forward(input_A)

    A = test_layer.sigmoid_forward(Z)
    print("A = ", A)


def linear_backward_test1():
    test_layer = ml.Layer()

    test_layer.W = np.array([0.3190391, -0.24937038, 1.46210794], ndmin=2)
    test_layer.b = np.array([-2.06014071])
    test_layer.A_prev = np.array([[-0.52817175, -1.07296862],
                                  [ 0.86540763, -2.3015387 ],
                                  [ 1.74481176, -0.7612069 ]])

    dZ = np.array([1.62434536, -0.61175641], ndmin=2)
    dW, db, dA_prev = test_layer.linear_backward(dZ)

    # TODO: Copy above code and make a proper test_layer constructor

    print("dA_prev = ", dA_prev)
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

def layer_initialize_test1():
    np.random.seed(1)
    layer = ml.Layer()

    parameters = layer.initialize(3, 2, 1, layer_type="fc")

    print(parameters)


def forward_prop_test1():
    layer1 = ml.Layer()
    layer2 = ml.Layer()
    layer3 = ml.Layer()

    layer1.W = np.array(
        [[0.35480861,  1.81259031, -1.3564758, -0.46363197,  0.82465384],
         [-1.17643148,  1.56448966,  0.71270509, -0.1810066,   0.53419953],
         [-0.58661296, -1.48185327,  0.85724762,   0.94309899,  0.11444143],
         [-0.02195668, -2.12714455, -0.83440747, - 0.46550831,  0.23371059]]
    )

    layer1.b = np.array([[ 1.38503523],
                         [-0.51962709],
                         [-0.78015214],
                         [ 0.95560959]])

    layer2.W = np.array(
        [[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
         [-0.56147088, -1.0335199,   0.35877096,  1.07368134],
         [-0.37550472,  0.39636757, -0.47144628,  2.33660781]])

    layer2.b = np.array([[ 1.50278553],
                        [-0.59545972],
                        [ 0.52834106]])

    layer3.W = np.array([[ 0.9398248, 0.42628539, -0.75815703]])
    layer3.b = np.array([-0.16236698])

    net = ml.Network()

    net.add_layer(layer1)
    net.add_layer(layer2)
    net.add_layer(layer3)

    X = np.array([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
                  [-2.48678065,  0.91325152,  1.12706373, -1.51409323],
                  [ 1.63929108, -0.4298936,   2.63128056,  0.60182225],
                  [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
                  [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])


    test = np.dot(layer1.W, X) + layer1.b
    print("test = ", test)
    AL = net.forward_propagate(X)


def cost_function_test1():
    print("test")
    net = ml.Network()
    opt = ml.Optimizer(net)


    Y = np.array([1, 1, 1], ndmin=2)
    AL = np.array([ 0.8,  0.9, 0.4])

    cost = opt.compute_cost(Y, AL)

    print(cost)


def full_backprop_test1():
    print("Full backprop test1")


    labels = np.array([ True, False, True] , ndmin=2)
    X = np.array([[ 1.62434536, -0.61175641, -0.52817175],
                  [-1.07296862,  0.86540763, -2.3015387 ]])

    net = ml.Network()
    layer1 = ml.Layer("tanh")
    layer2 = ml.Layer("sigmoid")

    layer1.W = np.array([[-0.00416758, -0.00056267],
                         [-0.02136196,  0.01640271],
                         [-0.01793436, -0.00841747],
                         [ 0.00502881, -0.01245288]])
    layer1.b = np.zeros((4,1))
    layer1.A = np.array( [[-0.00616578,  0.0020626  , 0.00349619],
                          [-0.05225116  ,0.02725659 ,-0.02646251],
                          [-0.02009721  ,0.0036869  ,0.02883756],
                          [ 0.02152675 ,-0.01385234 ,0.02599885]])

    layer2.W = np.array([-0.01057952, -0.00909008,  0.00551454,  0.02292208], ndmin=2)
    layer2.b = np.zeros((1, 1))
    layer2.A = np.array([0.5002307, 0.49985831, 0.50023963], ndmin=2)
    layer2.A_prev = layer1.A
    layer1.A_prev = X

    net = ml.Network()
    net.add_layer(layer1)
    net.add_layer(layer2)

    AL = layer2.A
    net.back_propagate(labels, AL)

    print("[Output] dW1 = ", layer1.dW)
    print("[Output] db1 = ", layer1.db)

    print("[Output] dW2 = ", layer2.dW)
    print("[Output] db1 = ", layer2.db)

    print("[Truth] dW1 = [[ 0.00301023 -0.00747267] \n [ 0.00257968 -0.00641288]\n [-0.00156892  0.003893  ] \n [-0.00652037  0.01618243]]")
    print("[Truth] db1 = \n [[ 0.00176201] \n [ 0.00150995] \n  [-0.00091736] \n [-0.00381422]]")

    print("[Truth] dW2 = [[ 0.00078841  0.01765429 -0.00084166 -0.01022527]]")
    print("[Truth] db2 = [[-0.16655712]]")


def soft_max_test1():
    Z = np.array([1, 2, 3])

    layer1 = ml.Layer("softmax")

def model_test1():

    X, y = sklearn.datasets.make_moons(200, noise=0.2)


    layer1 = ml.Layer("tanh")
    layer1.initialize(2, 5)

    layer2 = ml.Layer("tanh")
    layer2.initialize(5, 5)

    layer3 = ml.Layer("tanh")
    layer3.initialize(5, 5)

    layer4 = ml.Layer("sigmoid")
    layer4.initialize(5, 1)

    layers = [layer1, layer2, layer3, layer4]

    net = ml.Network(layers)

    # print("L1 W = ", str(layer1.W))
    # print("L1 b = ", str(layer1.b))

    # net.add_layer(layer1)
    # net.add_layer(layer2)
    # net.add_layer(layer3)
    # net.add_layer(layer4)

    #  plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    X_n = np.array(X).T
    y_n = np.array(y, ndmin=2)
    # print("y shape = ", y.shape)
    # print("X shape = ", X.shape)
    # plt.show()
    # net.plot_decision_boundary(X, y)

    opt = ml.Optimizer(net, learning_rate=0.1)
    opt.train(X_n, y_n, iterations=15000)

    net.plot_decision_boundary(X, y)
    opt.plot_cost()


    prediction = net.predict(X_n)

    sum(prediction - y)


def model_test2():
    digits = load_digits()
    print(digits.data.shape)
    # plt.gray()
    # plt.matshow(digits.images[1])
    # plt.show()
    # np.random.seed(2)

    num_samples = digits.data.shape[0]
    print("number of samples: ", num_samples)
    labels = np.zeros((num_samples, 10))

    for i, value in enumerate(digits.target):
        labels[i, value] = 1

    layer1 = ml.Layer("relu")
    layer1.initialize(64, 10)

    layer2 = ml.Layer("softmax")
    layer2.initialize(10, 10)

    layer3 = ml.Layer("relu")
    layer3.initialize(10, 10)

    layer4 = ml.Layer("softmax")
    layer4.initialize(10, 10)

    layers = [layer1, layer2]

    net = ml.Network(layers)

    opt = ml.Optimizer(net, learning_rate=0.001)
    X = digits.data.T
    opt.train(X, labels.T, iterations=10000)
    opt.plot_cost()
    accuracy = opt.compute_accuracy(X, digits.target)

    print("Training Set accuracy = ", str(accuracy*100) + "%")


def zeropad_test():
    print("Zero padding test")
    layer1 = ml.Layer(parameters={
        "n_h" : 1,
        "n_x" : 1,
        "act_func" : "tanh",
    })
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = layer1.zero_pad(x, 2)

    print("x.shape =", x.shape)
    print("x_pad.shape =", x_pad.shape)
    print("x[1,1] =", x[1, 1])
    print("x_pad[1,1] =", x_pad[1, 1])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])
    plt.show()


def conv_single_step_test():
    print("Conv single step test")
    np.random.seed(1)
    layer1 = ml.Layer(parameters={
        "n_h": 1,
        "n_x": 1,
        "act_func": "tanh",
    })
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)

    Z = layer1.conv_single_step(a_slice_prev, W, b)
    print("Z =", Z)

def conv_test():
    layer1 = ml.Layer(
        type ="conv",
        parameters={"C":  8, "sx": 2, "act_func": "tanh"
        })

    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    layer1.W = np.random.randn(2, 2, 3, 8)
    layer1.b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 2}

    layer1.pad = hparameters["pad"]
    layer1.stride = hparameters["stride"]

    Z = layer1.conv_forward(A_prev)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3, 2, 1])
    # print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

def pool_for_test():
    layer1 = ml.Layer(parameters={
        "n_h": 1,
        "n_x": 1,
        "act_func": "tanh",
    })

    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride": 2, "f": 3}

    layer1.f = hparameters["f"]
    layer1.stride = hparameters["stride"]
    layer1.mode = "max"

    A = layer1.pool_forward(A_prev)
    print("mode = max")
    print("A =", A)
    print()

    layer1.mode = "average"
    A = layer1.pool_forward(A_prev)
    print("mode = average")
    print("A =", A)

def conv_back_test():

    layer1 = ml.Layer(parameters={
        "n_h": 1,
        "n_x": 1,
        "act_func": "tanh",
    })

    np.random.seed(1)

    layer1.A_prev = np.random.randn(10, 4, 4, 3)
    layer1.W = np.random.randn(2, 2, 3, 8)
    layer1.b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 2}

    layer1.pad = hparameters["pad"]
    layer1.stride = hparameters["stride"]

    Z = layer1.conv_forward(layer1.A_prev)

    np.random.seed(1)
    dA, dW, db = layer1.conv_backward(Z)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))

def mask_test():
    layer1 = ml.Layer(parameters={
        "n_h": 1,
        "n_x": 1,
        "act_func": "tanh",
    })

    np.random.seed(1)
    x = np.random.randn(2, 3)

    mask = layer1.create_mask_from_window(x)
    print('x = ', x)
    print("mask = ", mask)

def dist_test():
    layer1 = ml.Layer(parameters={
        "n_h": 1,
        "n_x": 1,
        "act_func": "tanh",
    })
    a = layer1.distribute_value(2, (2, 2))
    print('distributed value =', a)

def pool_back_test():
    layer1 = ml.Layer(parameters={
        "n_h": 1,
        "n_x": 1,
        "act_func": "tanh",
    })
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride": 1, "f": 2}

    layer1.A_prev = A_prev
    layer1.stride = hparameters["stride"]
    layer1.f = hparameters["f"]


    layer1.mode = "max"
    A = layer1.pool_forward(A_prev)
    layer1.A = A
    dA = np.random.randn(5, 4, 2, 2)
    layer1.dA = dA

    layer1.mode = "max"
    dA_prev = layer1.pool_backward(dA)
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print()

    layer1.mode = "average"
    dA_prev = layer1.pool_backward(dA)
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])


def main():
    print("Running test  case:")
    # linear_forward_test1()
    # sigmoid_forward_test1()
    # linear_backward_test1()
    # sigmoid_backward_test1()
    # model_test1()
    # model_test2()
    # layer_initialize_test1()
    # forward_prop_test1()
    # cost_function_test1()
    # full_backprop_test1() Not working
    # zeropad_test()
    # conv_single_step_test()
    conv_test()
    # pool_for_test()
    # conv_back_test()
    # mask_test()
    # dist_test()
    # pool_back_test()

    #TODO: write a compute cost test




if __name__ == "__main__": main()