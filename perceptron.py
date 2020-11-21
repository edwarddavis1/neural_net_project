import numpy as np
import pandas as pd
import plotly.express as px
from ggplot import *


# Read in the training data - two separate MVN distributions
train_data = pd.read_csv("perceptron_training_data.csv")
train_data = train_data.drop(train_data.columns[0], axis=1)

# Visualise Training data
train_plot_data = train_data
train_plot_data['y'] = train_plot_data["y"].astype(str)
fig = px.scatter(train_plot_data, x='x_1', y='x_2', color='y')
fig.show()

# Initialise the model
x = np.matrix(train_data.drop('y', axis=1).values)
n = x.shape[0]      # Number of data points
d = x.shape[1]      # Input feature dimensions
ones = np.asmatrix(np.ones(n))
X = np.concatenate((x.T, ones))                 # Input feature space
w = np.matrix(np.random.uniform(-1, 1, d + 1))  # Weights (row vector)
f = np.dot(w, X)    # Initialised model

# See initial model
ggplot(aes('x_1', 'x_2', color='y'), data=train_data) +\
    geom_point() +\
    geom_abline(intercept=-w.item(2) / w.item(1), slope=-w.item(0) / w.item(1))

# Train the model
