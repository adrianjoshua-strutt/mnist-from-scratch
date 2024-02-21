import numpy as np
from loss.Loss import Loss


class CrossEntropyLoss(Loss):

    def forward(self, output: np.array, label: np.array) -> np.array:
        loss = -np.sum(label*np.log(output))
        return loss / output.shape[0]

    def backward(self, output: np.array, label: np.array) -> np.array:
        return -(label / output)
