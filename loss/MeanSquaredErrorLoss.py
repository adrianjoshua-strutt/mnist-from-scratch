import numpy as np
from loss.Loss import Loss


class MeanSquaredErrorLoss(Loss):

    def forward(self, output: np.array, label: np.array) -> np.array:
        return np.square(np.subtract(output, label)).mean()

    def backward(self, output: np.array, label: np.array) -> np.array:
        return output - label
