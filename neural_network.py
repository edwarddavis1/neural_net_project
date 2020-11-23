import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Differentiated sigmoid"""
    return sigmoid(x) * (1 - sigmoid(x))


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
        self.w1 = np.random.rand(self.x.shape[1], 4)
        # Weights b/w hidden layer and output layer
        self.w2 = np.random.rand(4, 1)
        self.output = np.zeros(self.y.shape)

        # Store the loss after each iteration
        self.losses = []
        self.iterations = 0

    def feedforward(self):
        """Forward propagation of the weights throug the network"""
        # Hidden layer
        self.layer1 = sigmoid(np.dot(self.x, self.w1))
        # Output layer
        self.output = sigmoid(np.dot(self.layer1, self.w2))

    def backprop(self):
        """Backpropagation algorithm to update the weights of the model"""
        # Calculate the gradient of the loss function (in this case using LS)
        z2 = np.vectorize(sigmoid_derivative)(self.output)
        d_w2 = self.layer1.T * 2 * np.multiply((self.y - self.output), z2)

        z1 = np.vectorize(sigmoid_derivative)(self.layer1)
        d_w1 = self.x.T * 2 * \
            np.multiply(np.multiply((self.y - self.output), z2) *
                        self.w2.T, z1)

        # Update weights
        self.w2 += d_w2
        self.w1 += d_w1

    def train(self, max_iter):
        """Train the network by forward- and backpropatating weights"""
        for i in range(max_iter):
            # Propagate weights forward
            self.feedforward()
            self.iterations += 1
            self.losses.append(np.square(NN.y - NN.output).sum())
            # Update the weights and backpropagate
            self.backprop()
        # Calculate the output for the most up-to-date weights
        self.feedforward()

    def plot_training(self):
        """Plot the loss as a function of iterations"""
        fig = plt.plot(np.linspace(1, self.iterations,
                                   self.iterations), self.losses)
        return fig

    def predict(self, x):
        """Predicts an output given some input x"""
        # Hidden layer
        layer1_prediction = sigmoid(np.dot(x, self.w1))
        # Output layer
        output_prediction = sigmoid(np.dot(layer1_prediction, self.w2))
        return output_prediction


# Read in training data
# train_data = pd.read_csv("Audiobooks_data.csv")
train_data = pd.DataFrame({
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "x3": [1, 1, 1, 1],
    "y": [0, 1, 1, 0]
})

inputs = np.matrix(train_data.drop("y", axis=1).values)
y = np.matrix(train_data.y.values).T

NN = NeuralNet(inputs, y)
NN.train(max_iter=10)

new_inputs = inputs[:-1, 1].T
NN.predict(new_inputs)

NN.plot_training()
