import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import plotly.express as px


# Read in the training data - two separate MVN distributions
train_data = pd.read_csv("perceptron_training_data.csv")
train_data = train_data.drop(train_data.columns[0], axis=1)

# Visualise Training data
train_plot_data = train_data
train_plot_data['y'] = train_plot_data["y"].astype(str)
fig = px.scatter(train_plot_data, x='x_1', y='x_2', color='y')
fig.show()
