import numpy as np
from training.SGDFlavor import SGDFlavor


class SGDFlavorVanilla(SGDFlavor):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    # Parameter: Parameter to update
    # Gradient: Gradients to update
    def calculate(self, parameter: np.array, parameter_delta: np.array) -> np.array:
        return parameter - self.learning_rate * parameter_delta
