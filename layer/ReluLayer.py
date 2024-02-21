
from layer.Layer import Layer
import numpy as np


class ReluLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, inTensor: np.array) -> np.array:
        return inTensor * (inTensor > 0)

    def backward(self, outGradients: np.array, inTensor: np.array) -> np.array:
        relu_derivation = (inTensor > 0) * 1
        return np.matmul(relu_derivation, outGradients)
