import numpy as np

class NeuralNet:
    def __init__(self, L, Nn) -> None:
        """
            INPUTS
            Nl: Number of Layers
            Nn: List of neurons per layer
            PROPERTIES
            W[l][j, k]: Weights in node k of layer l-1 and Node j of layer l
        """
        self.L = L
        self.Nn = Nn
        self.W = [np.ones((Nn[i+1], Nn[i])) for i in range(L-1)]
        self.XTrain = None
        self.YTrain = None
        self.D = None

    def addData(self, XTrain, YTrain):
        """XTrain[i, j]: Dimension j of training sample i"""
        assert(XTrain[0].shape == self.Nn[0], "Training data input dimension does not match input layer")
        assert(YTrain[0].shape == self.Nn[-1], "Training data output dimension does not match output layer")
        self.XTrain = XTrain
        self.YTrain = YTrain
        self.D = self.XTrain.shape[1]

    def relu(x):
        return np.maximum(np.zeros(x.shape), x)

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    def _y(self, i, input):
        """Output y of layer i"""
        assert(i >= 0 and i < self.L-1, "Layer {i} is not valid")
        if i == 0:
            return input
        return self.relu( self.W @ self._y(i-1, input) )
    
    def predict(self, input):
        return self.softmax( self.W @ self._y(self.L-2, input) )
        
    def input_cost(self, x, y):
        np.linalg.norm(self.predict(x) - y, ord=2)

    def C(self):
        s = 0
        for i in range(self.D):
            s += self.input_cost(self.XTrain[i], self.YTrain[i])
        return s


