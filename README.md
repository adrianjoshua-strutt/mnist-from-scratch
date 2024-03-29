# MNIST From Scratch
#### A simple feedforward neural network (FNN) from scratch using numpy to solve MNIST

# Background:

I build a simple feedforward neural network (FNN) from scratch without relying on PyTorch or Tensorflow/Keras, using only numpy to solve the MNIST dataset.
It employs vanilla stochastic gradient descent and the MSELoss.
The project does not use Pytorch or Tensorflow/Keras and is only coded using numpy.

The network design involves simply adding layers to the array in the `Network` constructor:

     network = Network([

      FullyConnectedLayer(784, 128),
      SigmoidLayer(),
      FullyConnectedLayer(128, 10),
      SoftmaxLayer()

     ], MeanSquaredErrorLoss(), SGDFlavorVanilla(0.01))


# Key Takeaways:
### 1. A Deeper Understanding of Neural Networks
Implementing backpropagation, MSE loss, fully connected layers, and stochastic gradient descent from scratch.
### 2. Getting Used to Vanilla Numpy
Gain familiarity with pure numpy.

# Results:

### The Training Accuracy and Loss

![The Training Accuracy and Loss](./docs/training.png?raw=true "The Training Accuracy and Loss")

#### The final test accuracy for a 2-layer FNN is 96.39% after training for 100 epochs.
