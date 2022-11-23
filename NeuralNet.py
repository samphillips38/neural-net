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
        assert XTrain.shape[1] == self.Nn[0], "Training data input dimension does not match input layer"
        assert YTrain.shape[1] == self.Nn[-1], "Training data output dimension does not match output layer"
        self.XTrain = XTrain
        self.YTrain = YTrain
        self.D = self.XTrain.shape[1]

    def relu(self, x):
        return np.maximum(np.zeros(x.shape), x)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def z(self, l, x):
        """z for layer l"""
        assert l >= 0 and l < self.L, f"Layer {i} is not valid"
        if l == 0:
            return x
        return self.W @ self.y(l-1, x)

    def y(self, l, x):
        """Output y of layer i"""
        assert l >= 0 and l < self.L, f"Layer {l} is not valid"
        if l == 0:
            return x
        if l == self.L-1:
            return self.softmax( self.z(l) )
        return self.relu( self.z(l) )
    
    def predict(self, x):
        return self.y(self.L-1, x)
        
    def input_cost(self, x, y):
        return np.linalg.norm(self.predict(x) - y, ord=2)

    def C(self):
        s = 0
        for i in range(self.D):
            s += self.input_cost(self.XTrain[i], self.YTrain[i])
        return s
    
    def drelu(self, x):
        pass

    def dsoftmax(self, x):
        return self.softmax(x) * (1 - self.softmax(x))

    def dydw(self, li, lj, x):
        """Derivative of y at layer li by w at layer lj"""
        assert lj <= li, "Cannot find forward derivative"
        if li == self.L-1:
            return self.dsoftmax( self.z(li-1, x) ) @ self.dzdw(li, lj, x)
        return self.drelu( self.z(li-1, x) ) @ self.dzdw(li, lj, x)

    def dzdw(self, li, lj, x):
        """Derivative of z at layer li by w at layer lj"""
        assert lj < li, "Cannot find forward derivative"
        if li-1 == lj:
            return self.y(lj, x)
        return self.W[li-1] @ self.dydw(li-1, lj, x)

    def dCdw(self, x, y, l):
        """Derivative wrt w at layer l"""
        return 2 * (self.predict(x) - y) @ self.dydw(self.L-1, l)