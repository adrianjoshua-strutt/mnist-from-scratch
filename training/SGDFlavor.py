import numpy as np


class SGDFlavor(object):

    # Parameter: Parameter to update
    # Gradient: Gradients to update
    def calculate(self, parameter: np.array, parameter_delta: np.array) -> np.array:
        pass