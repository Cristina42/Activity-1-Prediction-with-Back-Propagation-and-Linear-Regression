import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the training and testing data
train_data = pd.read_csv('traindata.csv')
test_data = pd.read_csv('testdata.csv')

# Separate the features (X) and target variable (y) for training and testing
X_train = train_data.drop(columns=['Life expectancy ']).values
y_train = train_data['Life expectancy '].values

X_test = test_data.drop(columns=['Life expectancy ']).values
y_test = test_data['Life expectancy '].values

# Initialize MinMaxScaler for scaling the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale the features (X)
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target variable (y)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
# Define the Neural Network
# class FeedforwardNN(nn.Module):
#     def __init__(self, input_size, hidden_layers, output_size, activation_func):
#         super(FeedforwardNN, self).__init__()
#         layers = []
#         sizes = [input_size] + hidden_layers + [output_size]
#         for i in range(len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#             if i < len(sizes) - 2:  
#                 if activation_func == 'relu':
#                     layers.append(nn.ReLU())
#                 elif activation_func == 'tanh':
#                     layers.append(nn.Tanh())
#                 elif activation_func == 'sigmoid':
#                     layers.append(nn.Sigmoid())
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

# # Initialize the neural network
# input_size = X_train_scaled.shape[1]
# hidden_layers = [16, 8]
# output_size = 1
# activation_func = 'relu'

# model = FeedforwardNN(input_size, hidden_layers, output_size, activation_func)

# Define the Neural Network
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16),  # Input to hidden layer 1
    nn.ReLU(),                       # Activation
    nn.Linear(16, 8),                # Hidden layer 1 to hidden layer 2
    nn.ReLU(),                       # Activation
    nn.Linear(8, 1)                  # Output layer
)

# Set training parameters
learning_rate = 0.001
epochs = 1000
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
train_losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

# Reverse the scaling for predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

# Scatter plot for BP-F model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line of perfect prediction
plt.title('BP-F Model: Predicted vs Actual Values')
plt.xlabel('Actual Values (zμ)')
plt.ylabel('Predicted Values (ŷμ)')
plt.grid(True)
plt.show()