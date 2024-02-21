from layer.Layer import Layer
import numpy as np


class SigmoidLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, inTensor: np.array) -> np.array:
        return 1 / (1 + np.exp(-inTensor))

    def backward(self, outGradients: np.array, inTensor: np.array) -> np.array:
        sigmoid_derivation = self.forward(inTensor) * (1.0 - self.forward(inTensor))
        return sigmoid_derivation * outGradients
