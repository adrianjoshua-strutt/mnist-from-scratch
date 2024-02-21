import numpy as np
from training.SGDFlavor import SGDFlavor


class Layer(object):

    def forward(self, inTensors: np.array) -> np.array:
        pass

    def backward(self, outGradients: np.array, inTensor: np.array) -> np.array:
        pass

    def updateParameters(self, sgd: SGDFlavor):
        pass
