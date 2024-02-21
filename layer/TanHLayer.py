
from layer.Layer import Layer
import numpy as np


class TanHLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, inTensor: np.array) -> np.array:
        return np.tanh(inTensor)

    def backward(self, outGradients: np.array, inTensor: np.array) -> np.array:
        tanh_derivation = 1-inTensor**2
        return tanh_derivation * outGradients
