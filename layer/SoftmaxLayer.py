from layer.Layer import Layer
import numpy as np


class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, inTensor: np.array) -> np.array:
        softmax = (np.exp(inTensor) / np.exp(inTensor).sum())
        return softmax

    def backward(self, outGradients: np.array, inTensor: np.array) -> np.array:
        x = self.forward(inTensor).T
        matrix = np.zeros([len(x), len(x)])
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i == j:
                    matrix[i][j] = x[i] * (1 - x[i])
                else:
                    matrix[i][j] = -x[i] * x[j]
        return np.matmul(outGradients, matrix)
