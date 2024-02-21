import numpy as np


class Loss(object):

    def forward(self, output: np.array, label: np.array) -> np.array:
        pass

    def backward(self, output: np.array, label: np.array) -> np.array:
        pass