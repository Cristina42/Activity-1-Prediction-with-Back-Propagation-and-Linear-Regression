# our own implemented BP model
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, activation_function, validation_split=0.2):
        self.L = len(layers)  # Number of layers
        self.n = layers  # Number of neurons in each layer (including input and output layers)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.validation_split = validation_split
        self.fact = activation_function

 # Initialize activations, weights, biases, and error terms
        self.xi = [np.zeros((layer, 1)) for layer in layers]  # Activations (ξ)
        self.w = [np.random.randn(layers[l], layers[l-1]) * 0.01 for l in range(1, self.L)] # initializing of weights
        self.theta = [np.zeros((layer, 1)) for layer in layers[1:]]  # Thresholds (biases θ)
        self.delta = [np.zeros((layer, 1)) for layer in layers[1:]]  # Error terms (Δ)
        self.d_w_prev = [np.zeros_like(w) for w in self.w]  # Previous weight changes
        self.d_theta_prev = [np.zeros_like(t) for t in self.theta]  # Previous bias changes 

 # Activation functions and their derivatives
        self.activations = {
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'relu': (self.relu, self.relu_derivative),
            'tanh': (self.tanh, self.tanh_derivative),
            'linear': (self.linear, self.linear_derivative),
        }

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def linear(self, z):
        return z

    def linear_derivative(self, z):
        return np.ones_like(z)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def activation_function(self, z, is_output_layer=False):
    # Dynamically choose the activation function based on self.fact
        activation_func, _ = self.activations[self.fact]
    
    # Apply the chosen activation function to both hidden and output layers
        return activation_func(z)

    def activation_derivative(self, z, is_output_layer=False):
    # Dynamically choose the activation derivative based on the one set in self.fact
        _, activation_deriv = self.activations[self.fact]
    
    # Apply the derivative of the chosen activation function to both hidden and output layers
        return activation_deriv(z)

    def forward(self, X):
        self.xi[0] = X  # Input layer activations
        for l in range(1, self.L - 1):  # Hidden layers
            z = np.dot(self.w[l - 1], self.xi[l - 1]) + self.theta[l - 1]
            self.xi[l] = self.activation_function(z)
        z_output = np.dot(self.w[-1], self.xi[-2]) + self.theta[-1]
        self.xi[-1] = self.activation_function(z_output, is_output_layer=True)

    def backpropagate(self, X, y):
        self.forward(X)
    
    # Output layer error
        self.delta[-1] = (self.xi[-1] - y) * self.activation_derivative(self.xi[-1], is_output_layer=True)
    
    # Gradient clipping
        max_grad_norm = 1.0  # Maximum gradient norm value
        for l in range(self.L - 1):
        # Clip the gradients to ensure they don't explode
            self.delta[l] = np.clip(self.delta[l], -max_grad_norm, max_grad_norm)
    
    # Backpropagate the error to the hidden layers
        for l in range(self.L - 2, 0, -1):  # Hidden layers
            self.delta[l - 1] = np.dot(self.w[l].T, self.delta[l]) * self.activation_derivative(self.xi[l])

    # Update weights and biases
        for l in range(self.L - 1):
            self.d_w_prev[l] = self.learning_rate * np.dot(self.delta[l], self.xi[l].T) + self.momentum * self.d_w_prev[l]
            self.d_theta_prev[l] = self.learning_rate * np.sum(self.delta[l], axis=1, keepdims=True) + self.momentum * self.d_theta_prev[l]
            self.w[l] -= self.d_w_prev[l]
            self.theta[l] -= self.d_theta_prev[l]

    def fit(self, X, y):
    # Train the network using backpropagation.
        for epoch in range(self.epochs):
            self.backpropagate(X, y)
            if epoch % 100 == 0:
                loss = np.mean((self.xi[-1] - y) ** 2)
                # print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
    # Perform a forward pass and return predictions.
        self.forward(X)
        return self.xi[-1]

    def loss_epochs(self, X_train, y_train, X_val, y_val):
    # Track training and validation loss over epochs.
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            self.fit(X_train, y_train)
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)

            train_loss = np.mean((train_pred - y_train) ** 2)  # MSE for training
            val_loss = np.mean((val_pred - y_val) ** 2)  # MSE for validation

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        return np.array(train_losses), np.array(val_losses)

    def split_data(self, X, y):
        # Perform validation split
        validation_size = int(len(X) * self.validation_split)
        X_train = X[validation_size:]
        y_train = y[validation_size:]
        X_val = X[:validation_size]
        y_val = y[:validation_size]
        return X_train, y_train, X_val, y_val

# Standardize Functions
def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std


def destandardize(data, mean, std):
    return data * std + mean


# Load Data
def load_data(train_data, test_data):
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)

    # Separate features and targets
    X_train_val = train.iloc[:, :-1].values
    y_train_val = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Standardize
    X_train_std, X_mean, X_std = standardize(X_train_val)
    X_test_std = (X_test - X_mean) / X_std

    y_train_std, y_mean, y_std = standardize(y_train_val)
    y_test_std = (y_test - y_mean) / y_std

    return X_train_std, y_train_std, X_test_std, y_test_std, y_mean, y_std


# Main Code
X_train, y_train, X_test, y_test, y_mean, y_std = load_data("traindata.csv", "testdata.csv")

layers = [14, 19, 10, 1]
nn = NeuralNet(layers, epochs=10, learning_rate=0.001, momentum=0.9, activation_function="tanh", validation_split=0.2)

# Split data within NeuralNet
X_train_split, y_train_split, X_val, y_val = nn.split_data(X_train, y_train)

# Train
train_losses, val_losses = nn.loss_epochs(X_train_split.T, y_train_split.T, X_val.T, y_val.T)

# Predict and destandardize
predictions_std = nn.predict(X_test.T)
predictions = destandardize(predictions_std, y_mean, y_std)

# Evaluate
mse = np.mean((predictions - y_test) ** 2)
mae = np.mean(np.abs(predictions - y_test))
mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100

print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")

# Define a list of hyperparameter combinations (increase number of epochs, small number of epochs was used to check if it works)
combinations = [
    {'layers': [14, 9, 1], 'epochs': 500, 'learning_rate': 0.01, 'momentum': 0.9, 'activation': 'relu'},
    {'layers': [14, 16, 8, 1], 'epochs': 1000, 'learning_rate': 0.001, 'momentum': 0.8, 'activation': 'tanh'},
    {'layers': [14, 9, 5, 1], 'epochs': 500, 'learning_rate': 0.01, 'momentum': 0.7, 'activation': 'sigmoid'},
    {'layers': [14, 10, 1], 'epochs': 750, 'learning_rate': 0.005, 'momentum': 0.7, 'activation': 'relu'},
    {'layers': [14, 12, 8, 1], 'epochs': 1000, 'learning_rate': 0.001, 'momentum': 0.9, 'activation': 'tanh'},
    {'layers': [14, 16, 1], 'epochs': 1000, 'learning_rate': 0.0005, 'momentum': 0.9, 'activation': 'relu'},
    {'layers': [14, 20, 15, 5, 1], 'epochs': 500, 'learning_rate': 0.01, 'momentum': 0.9, 'activation': 'sigmoid'},
    {'layers': [14, 8, 1], 'epochs': 1000, 'learning_rate': 0.005, 'momentum': 0.8, 'activation': 'relu'},
    {'layers': [14, 9, 1], 'epochs': 1500, 'learning_rate': 0.0001, 'momentum': 0.9, 'activation': 'tanh'},
    {'layers': [14, 18, 10, 1], 'epochs': 1000, 'learning_rate': 0.001, 'momentum': 0.8, 'activation': 'relu'}
]

# Function to calculate MSE, MAE, and MAPE
def calculate_metrics(predictions, actual):
    mse = np.mean((predictions - actual) ** 2)
    mae = np.mean(np.abs(predictions - actual))
    mape = np.mean(np.abs((predictions - actual) / actual)) * 100
    return mse, mae, mape

# Function to evaluate each combination
def evaluate_combinations():
    for i, params in enumerate(combinations):
        print(f"\nTesting Combination {i+1}: {params}")
        
        layers = params['layers']
        epochs = params['epochs']
        learning_rate = params['learning_rate']
        momentum = params['momentum']
        activation = params['activation']

        # Initialize NeuralNet with the current combination of hyperparameters
        nn = NeuralNet(layers, epochs=epochs, learning_rate=learning_rate, momentum=momentum, activation_function=activation)

        # Train the model and track loss for each epoch
        train_losses, val_losses = nn.loss_epochs(X_train.T, y_train.T, X_val.T, y_val.T)

        # Predict the output on the test set
        predictions_scaled = nn.predict(X_test.T)
        predictions = destandardize(predictions_scaled, y_mean, y_std)

        # Calculate and print metrics for MSE, MAE, and MAPE
        mse, mae, mape = calculate_metrics(predictions, y_test)
        print(f"Combination {i+1} Metrics:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")

# Call the evaluation function
evaluate_combinations()

def plot_scatter(actual, predicted, title, figure_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.6, color="blue")
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], linestyle="--", color="red")
    plt.title(title)
    plt.xlabel("Actual Values (zμ)")
    plt.ylabel("Predicted Values (ŷμ)")
    plt.grid(True)
    plt.savefig(figure_path)
    plt.show()

def plot_loss_curves(train_losses, val_losses, title, figure_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss", linestyle="--")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_path)
    plt.show()

# Initialize the neural network with the best parameters
best_nn = NeuralNet(
    layers=[14, 16, 8, 1],
    epochs=1000,
    learning_rate=0.001,
    momentum=0.8,
    activation_function="tanh"
)

# Split the training data
X_train_split, y_train_split, X_val, y_val = best_nn.split_data(X_train, y_train)

# Train the model
train_losses_best, val_losses_best = best_nn.loss_epochs(
    X_train_split.T,
    y_train_split.T,
    X_val.T,
    y_val.T
)

# Make predictions on the test set
best_predictions = best_nn.predict(X_test.T)

plot_scatter(
    y_test, 
    best_predictions, 
    "Figure 5: Best Combination - Actual vs Predicted Values", 
    "figure5_best_combination.png"
)

plot_loss_curves(
    train_losses_best,
    val_losses_best,
    "Figure 7: Best Combination - Loss Curves",
    "figure7_loss_best_combination.png"
)

# Initialize the neural network with the representative parameters
rep_nn = NeuralNet(
    layers=[14, 10, 1],
    epochs=750,
    learning_rate=0.005,
    momentum=0.7,
    activation_function="relu"
)

# Split the training data
X_train_split, y_train_split, X_val, y_val = rep_nn.split_data(X_train, y_train)

# Train the model
train_losses_rep, val_losses_rep = rep_nn.loss_epochs(
    X_train_split.T,
    y_train_split.T,
    X_val.T,
    y_val.T
)

# Make predictions on the test set
rep_predictions = rep_nn.predict(X_test.T)

plot_scatter(
    y_test, 
    rep_predictions, 
    "Figure 6: Representative Combination - Actual vs Predicted Values", 
    "figure6_representative_combination.png"
)

plot_loss_curves(
    train_losses_rep,
    val_losses_rep,
    "Figure 8: Representative Combination - Loss Curves",
    "figure8_loss_representative_combination.png"
)
