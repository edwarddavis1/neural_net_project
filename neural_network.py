import numpy as np


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Differentiated sigmoid"""
    return 1 / (1 + np.exp(-x)) ^ 2


class NeuralNet:
    """
    Neural network with an arbitrary number of inputs
    Note that there is one hidden layer with four nodes and one output
    hard coded in this example
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        # Weights b/w input and hidden layer
        self.w1 = np.random.rand(self.input.shape[1], 4)
        # Weights b/w hidden layer and output layer
        self.w2 = np.random.rand(4, 1)
        self.output = np.zeros(y.shape)

    def feedforward(self):
        """Forward propagation of the weights throug the network"""
        # Hidden layer
        self.layer1 = sigmoid(np.dot(self.x, self.w1))
        # Output layer
        self.output = sigmoid(np.dot(self.layer1, self.w2))

    def backprop(self):
        """Backpropagation algorithm to update the weights of the model"""
        # Calculate the gradient of the loss function (in this case using LS)
        d_w2 = np.dot(self.layer1.T, (2 * (self.y - self.output) *
                                      sigmoid_derivative(self.output)))
        d_w1 = np.dot(self.x.T, (np.dot(2 * (self.y - self.output) *
                                        sigmoid_derivative(self.output), self.w2.T) *
                                 sigmoid_derivative(self.layer1)))

        # Update weights
        self.w2 += d_w2
        self.w1 += d_w1
