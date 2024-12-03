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

# Initialize the regressor
regressor = LinearRegression()

# Fit the regressor on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Mean Absolute Percentage Error (MAPE)
# Avoid division by zero by checking for zero values in y_test
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Scatter plot for MLR model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line of perfect prediction
plt.title('MLR Model: Predicted vs Actual Values')
plt.xlabel('Actual Values (zμ)')
plt.ylabel('Predicted Values (ŷμ)')
plt.grid(True)
plt.show()
