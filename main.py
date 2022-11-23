from NeuralNet import NeuralNet
import numpy as np



NN = NeuralNet(3, [3, 5, 3])
XTrain = np.array([
    [1, 2, 3],
    [1, 3, 2],
    [4, 2, 3],
    [5, 1, 1],
    [-5, -3, -2],
    [-2, -2, -3],
    [-3, -1, -2]
])
YTrain = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
])
input = np.array([[100, 2, 3]]).T

NN.addData(XTrain, YTrain)
print(NN.C())
