import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and testing data from CSV files
train_data = pd.read_csv('traindata.csv')
test_data = pd.read_csv('testdata.csv')

# Separate the features (X) and target variable (y) for training and testing
X_train = train_data.drop(columns=['Life expectancy '])  # Drop the target column from training data
y_train = train_data['Life expectancy ']  # Extract the target column from training data

X_test = test_data.drop(columns=['Life expectancy '])  # Drop the target column from testing data
y_test = test_data['Life expectancy ']  # Extract the target column from testing data

# Initialize MinMaxScaler for scaling the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale the features (X)
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)  # Use the same scaler to transform test data

# Scale the target variable (y)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))  # Reshape for a single feature
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))  # Use the same scaler for test target

# Initialize the regressor
regressor = LinearRegression()

# Fit the regressor on the scaled training data
regressor.fit(X_train_scaled, y_train_scaled)

# Make predictions on the scaled test data
y_pred_scaled = regressor.predict(X_test_scaled)

# Reverse the scaling (descale) to get predictions in the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line of perfect prediction
plt.title('MLR Model: Predicted vs Actual Values')
plt.xlabel('Actual Values (zμ)')
plt.ylabel('Predicted Values (ŷμ)')
plt.grid(True)
plt.show()
