from layer.FullyConnectedLayer import FullyConnectedLayer
import numpy as np
from layer.SigmoidLayer import SigmoidLayer
from layer.SoftmaxLayer import SoftmaxLayer
from loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss
from training.Network import Network
from training.SGDFlavorVanilla import SGDFlavorVanilla
import mnist
import matplotlib.pyplot as plt


def preprocess_mnist_input(a):
    a = np.interp(a, (0, 255), (0, +1))
    a = np.expand_dims(a, axis=1)
    return a


def preprocess_mnist_label(a):
    a = np.squeeze(np.eye(10)[a.reshape(-1)])
    a = np.expand_dims(a, axis=1)
    return a


if __name__ == '__main__':

    mnist.init()

    x_train, t_train, x_test, t_test = mnist.load()

    x_train = preprocess_mnist_input(x_train)
    t_train = preprocess_mnist_label(t_train)
    x_test = preprocess_mnist_input(x_test)
    t_test = preprocess_mnist_label(t_test)

    activation_layer = SigmoidLayer()

    network = Network([

            FullyConnectedLayer(784, 128),
            SigmoidLayer(),
            FullyConnectedLayer(128, 10),
            SoftmaxLayer()

         ], MeanSquaredErrorLoss(), SGDFlavorVanilla(0.01))

    list_x = []

    accuracy_list_y = []
    loss_list_y = []

    for epoch in range(100):
        for i in range(len(x_train)):
            network.training_step(x_train[i], t_train[i])
            print(str(i) + " / " + str(len(x_train)) + " - " + str(network.right / (i+1) ))
        list_x.append(len(accuracy_list_y))
        accuracy_list_y.append(network.right / len(x_train))
        loss_list_y.append(network.accumulated_loss / len(x_train))
        plt.plot(list_x, accuracy_list_y, 'g', label='Training accuracy')
        plt.plot(list_x, loss_list_y, 'b', label='Training loss')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()
        network.right = 0
        network.accumulated_loss = 0

    network.right = 0
    for i in range(len(x_test)):
        network.inference_step(x_test[i], t_test[i])
        print(str(i) + " / " + str(len(x_test)) + " - " + str(network.right / (i + 1)))
