from layer.Layer import Layer
import numpy as np

from training.SGDFlavor import SGDFlavor


class FullyConnectedLayer(Layer):

    def __init__(self, in_features: int, out_features: int, t_weights: np.array = None, t_bias: np.array = None):

        self.in_features = in_features
        self.out_features = out_features

        if t_weights is not None:
            self.weights = t_weights
        else:
            self.weights = np.random.uniform(-1, 1, (in_features, out_features))

        if t_bias is not None:
            self.bias = t_bias
        else:
            self.bias = np.ones(out_features)

        self.weights_delta = np.ones((in_features, out_features))
        self.bias_delta = np.ones(out_features)

    def forward(self, inTensor: np.array) -> np.array:
        return np.dot(inTensor, self.weights) + self.bias

    def backward(self, outGradients: np.array, inTensor: np.array) -> np.array:
        self.weights_delta = inTensor.T * outGradients
        self.bias_delta = outGradients
        return np.matmul(outGradients, self.weights.T)

    def updateParameters(self, sgd: SGDFlavor):
        self.weights = sgd.calculate(self.weights, self.weights_delta)
        self.bias = sgd.calculate(self.bias, self.bias_delta)
