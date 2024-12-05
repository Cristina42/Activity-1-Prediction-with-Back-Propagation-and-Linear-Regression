import numpy as np
import pandas as pd

class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, activation_function):
        self.L = len(layers)  # Number of layers
        self.n = layers  # Number of neurons in each layer (including input and output layers)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.fact = activation_function

 # Initialize activations, weights, biases, and error terms
        self.xi = [np.zeros((layer, 1)) for layer in layers]  # Activations (ξ)
        self.w = [np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1]) for i in range(1, self.L)] 
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
        # Dynamically choose the activation function based on the one set in self.fact
        activation_func, _ = self.activations[self.fact]
        return activation_func(z) if not is_output_layer else self.linear(z)  # Linear activation for the output layer

    def activation_derivative(self, z, is_output_layer=False):
        # Dynamically choose the activation derivative based on the one set in self.fact
        _, activation_deriv = self.activations[self.fact]
        return activation_deriv(z) if not is_output_layer else self.linear_derivative(z)

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
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")

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

# Scaling and Descaling Functions
def scale(data, s_min=0, s_max=1):
    """Scale data to a given range [s_min, s_max]"""
    x_min = np.min(data, axis=0)
    x_max = np.max(data, axis=0)
    scaled_data = s_min + (s_max - s_min) * (data - x_min) / (x_max - x_min)
    return scaled_data, x_min, x_max

def descale(scaled_data, x_min, x_max, s_min=0, s_max=1):
    """Inverse scale transformation to return data to its original range"""
    return x_min + (x_max - x_min) * (scaled_data - s_min) / (s_max - s_min)

# Load and preprocess data
def load_data(train_data, test_data, validation_split=0.2):
    # Load data from CSV files
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)

    # Separate features and target (assumes target is the last column)
    X_train_val = train.iloc[:, :-1].values  # All columns except the last one (features)
    y_train_val = train.iloc[:, -1].values   # Last column (target)
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Split training and validation data
    validation_size = int(len(X_train_val) * validation_split)
    X_train = X_train_val[validation_size:]
    y_train = y_train_val[validation_size:]
    X_val = X_train_val[:validation_size]
    y_val = y_train_val[:validation_size]

    # Scale the target variable (y)
    y_train_scaled, y_min, y_max = scale(y_train, s_min=0, s_max=1)
    y_test_scaled = (y_test - y_min) / (y_max - y_min)  # Scale test data using train min/max

    # Scale features (X)
    X_train_scaled, x_min, x_max = scale(X_train, s_min=0, s_max=1)
    X_test_scaled = (X_test - x_min) / (x_max - x_min)

    return X_train_scaled, y_train_scaled, X_val, y_val, X_test_scaled, y_test_scaled, y_min, y_max
        
# Main Code
X_train, y_train, X_val, y_val, X_test, y_test, y_min, y_max = load_data('traindata.csv', 'testdata.csv', validation_split=0.2)

layers = [14, 19, 10, 1]
nn = NeuralNet(layers, epochs=100, learning_rate=0.0001, momentum=0.9, activation_function='tanh')  

# Call loss_epochs() correctly
train_losses, val_losses = nn.loss_epochs(X_train.T, y_train.T, X_val.T, y_val.T)

# Get predictions
predictions_scaled = nn.predict(X_test.T)  # This would be the scaled predictions
predictions = descale(predictions_scaled, y_min, y_max)  # Reverse the scaling

# Evaluate performance
mse = np.mean((predictions - y_test) ** 2)
mae = np.mean(np.abs(predictions - y_test))
mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100

print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")

# Define a list of hyperparameter combinations (increase number of epochs, small number of epochs was used to check if it works)
combinations = [
    {'layers': [14, 9, 1], 'epochs': 100, 'learning_rate': 0.01, 'momentum': 0.9, 'activation': 'relu'},
    {'layers': [14, 16, 8, 1], 'epochs': 100, 'learning_rate': 0.001, 'momentum': 0.8, 'activation': 'tanh'},
    {'layers': [14, 9, 5, 1], 'epochs': 100, 'learning_rate': 0.01, 'momentum': 0.7, 'activation': 'sigmoid'},
    {'layers': [14, 10, 1], 'epochs': 100, 'learning_rate': 0.005, 'momentum': 0.7, 'activation': 'relu'},
    {'layers': [14, 12, 8, 1], 'epochs': 100, 'learning_rate': 0.001, 'momentum': 0.9, 'activation': 'tanh'},
    {'layers': [14, 16, 1], 'epochs': 100, 'learning_rate': 0.0005, 'momentum': 0.9, 'activation': 'relu'},
    {'layers': [14, 20, 15, 5, 1], 'epochs': 100, 'learning_rate': 0.01, 'momentum': 0.9, 'activation': 'sigmoid'},
    {'layers': [14, 8, 1], 'epochs': 100, 'learning_rate': 0.005, 'momentum': 0.8, 'activation': 'relu'},
    {'layers': [14, 9, 1], 'epochs': 100, 'learning_rate': 0.0001, 'momentum': 0.9, 'activation': 'tanh'},
    {'layers': [14, 18, 10, 1], 'epochs': 100, 'learning_rate': 0.001, 'momentum': 0.8, 'activation': 'relu'}
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
        predictions = descale(predictions_scaled, y_min, y_max)  # Descale predictions

        # Calculate and print metrics for MSE, MAE, and MAPE
        mse, mae, mape = calculate_metrics(predictions, y_test)
        print(f"Combination {i+1} Metrics:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")

# Call the evaluation function
evaluate_combinations()

