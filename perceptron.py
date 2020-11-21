import pandas as pd
import plotly.express as px
import numpy as np
from ggplot import *

# Read in the training data - two separate MVN distributions
train_data = pd.read_csv("perceptron_training_data.csv")
train_data = train_data.drop(train_data.columns[0], axis=1)

# Visualise Training data
train_plot_data = train_data.copy()
train_plot_data['y'] = train_plot_data["y"].astype(str)
fig = px.scatter(train_plot_data, x='x_1', y='x_2', color='y')
fig.show()

# Initialise model = sum(w_i*x_i)
x = np.matrix(train_data.drop('y', axis = 1).values) # drop output column y
n = x.shape[0] # number data points
d = x.shape[1] # feautre space dimensions
ones = np.asmatrix(np.ones(n)) # ones for bias terms
X = np.concatenate((x.T,ones))
w = np.matrix(np.random.uniform(-1,1,d+1)) # ROW vector with 3 weights for 2D feature space (2 for data points + one bias)
# model
f = np.dot(w,X)
# see initial model

ggplot(aes("x_1", "x_2", color = 'y'), data = train_plot_data) +\
    geom_point() +\
    geom_abline(intercept = -w.item(2)/w.item(1), slope = -w.item(0) / w.item(1)) #  point = scatter, abline = straight line

# train the model
y = train_data["y"].values
# Hyperparamters
max_iter = 100  # Maximum iterations
lr_0 = 10       # Initial learning rate

# Training Loop
for i in range(max_iter):
    # Set learning rate
    lr = lr_0 / (i + 1)

    # Loop over data points
    for j in range(n):
        # Calculate model with most recent weights
        f = np.dot(w, X[:, j])
        # Check optimisation condition
        if f.item(0) * y[j] <= 0:
            w = w + np.asmatrix(lr * y[j] * X[:, j]).T


w

ggplot(aes("x_1", "x_2", color = 'y'), data = train_plot_data) + geom_point() + geom_abline(intercept = -w.item(2)/w.item(1), slope = -w.item(0) / w.item(1))








#
