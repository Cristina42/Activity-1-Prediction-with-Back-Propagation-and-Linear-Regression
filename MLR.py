from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Load the training and testing data from CSV files
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Separate the features (X) and target variable (y) for training and testing
X_train = train_data.drop(columns=['Life expectancy '])  # Drop the target column from training data
y_train = train_data['Life expectancy ']  # Extract the target column from training data

X_test = test_data.drop(columns=['Life expectancy '])  # Drop the target column from testing data
y_test = test_data['Life expectancy ']  # Extract the target column from testing data
