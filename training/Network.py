
from typing import List
import numpy as np
from layer.Layer import Layer
from loss.Loss import Loss
from training.SGDFlavor import SGDFlavor

class Network(SGDFlavor):

    def __init__(self, layers: List[Layer], loss: Loss, sgd: SGDFlavor):
        self.layers = layers
        self.loss = loss
        self.sgd = sgd

        self.cache_forward = [None] * (len(self.layers) + 1)
        self.cache_backward = [None] * (len(self.layers) + 1)

        self.loss_accumulated = []

        self.right = 0
        self.accumulated_loss = 0

    def print_info(self):
        print("input: " + str(self.cache_forward[0]))
        for i in range(len(self.layers)):
            print(str(self.layers[i]) + ": "+ str(self.cache_forward[i+1]))

    def training_step(self, input: np.array, label: np.array):

        self.cache_forward[0] = input
        for i in range(len(self.layers)):
            self.cache_forward[i+1] = self.layers[i].forward(self.cache_forward[i])

        forward_loss = self.loss.forward(self.cache_forward[-1], label)

        self.accumulated_loss += forward_loss

        if np.argmax(self.cache_forward[-1]) == np.argmax(label):
            self.right = self.right + 1

        backward_loss = self.loss.backward(self.cache_forward[-1], label)

        self.cache_backward[-1] = backward_loss
        for i in range(len(self.layers) - 1, -1, -1):
            self.cache_backward[i] = self.layers[i].backward(self.cache_backward[i + 1], self.cache_forward[i])

        for layer in self.layers:
            layer.updateParameters(self.sgd)

    def inference_step(self, input: np.array, label: np.array):

        self.cache_forward[0] = input
        for i in range(len(self.layers)):
            self.cache_forward[i + 1] = self.layers[i].forward(self.cache_forward[i])

        if np.argmax(self.cache_forward[-1]) == np.argmax(label):
            self.right = self.right + 1

        print(str(np.argmax(self.cache_forward[-1])) + " - " + str(np.argmax(label)))

        #self.print_info()
