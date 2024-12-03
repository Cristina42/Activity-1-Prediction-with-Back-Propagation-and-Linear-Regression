import numpy as np

class NeuralNet:
    def __init__(self, layers, epochs=1000, learning_rate=0.01, momentum=0.9, activation_function='relu', validation_split=0.2):
        self.L = len(layers)  # Number of layers
        self.n = layers  # Number of neurons in each layer (including input and output layers)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.validation_split = validation_split
        self.fact = activation_function

 # Initialize activations, weights, biases, and error terms
        self.xi = [np.zeros((layer, 1)) for layer in layers]  # Activations (ξ)
        self.w = [np.random.randn(layers[i], layers[i - 1]) * 0.01 for i in range(1, self.L)]  # Weights (w)
        self.theta = [np.zeros((layer, 1)) for layer in layers[1:]]  # Thresholds (biases θ)
        self.delta = [np.zeros((layer, 1)) for layer in layers[1:]]  # Error terms (Δ)
        self.d_w_prev = [np.zeros_like(w) for w in self.w]  # Previous weight changes
        self.d_theta_prev = [np.zeros_like(t) for t in self.theta]  # Previous bias changes

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def linear(self, z):
        return z

    def linear_derivative(self, z):
        return np.ones_like(z)

    def activation_function(self, z, is_output_layer=False):
        """
        Applies ReLU for hidden layers and Linear for the output layer.
        """
        if is_output_layer:
            return self.linear(z)  # Linear activation for output layer
        else:
            return self.relu(z)  # ReLU for hidden layers

    def activation_derivative(self, z, is_output_layer=False):
        """
        Derivative of activation functions: Linear for output, ReLU for hidden layers.
        """
        if is_output_layer:
            return self.linear_derivative(z)
        else:
            return self.relu_derivative(z)

    def forward(self, X):
        """
        Perform feed-forward propagation using h (fields) and xi (activations).
        """
        self.xi[0] = X  # Input layer activations
        for l in range(1, self.L - 1):  # Hidden layers
            z = np.dot(self.w[l - 1], self.xi[l - 1]) + self.theta[l - 1]
            self.xi[l] = self.activation_function(z)  # Apply ReLU

        # Output layer
        z_output = np.dot(self.w[-1], self.xi[-2]) + self.theta[-1]
        self.xi[-1] = self.activation_function(z_output, is_output_layer=True)  # Apply Linear

    def backpropagate(self, X, y):
        """
        Perform backpropagation and update weights and thresholds using momentum.
        """
        m = X.shape[1]  # Number of samples
        self.forward(X)  # Forward pass

        # Output layer error
        self.delta[-1] = (self.xi[-1] - y) * self.activation_derivative(self.xi[-1], is_output_layer=True)

        # Backpropagate the error
        for l in range(self.L - 2, 0, -1):  # Hidden layers
            self.delta[l - 1] = np.dot(self.w[l].T, self.delta[l]) * self.activation_derivative(self.xi[l])

        # Update weights and biases
        for l in range(self.L - 1):
            self.d_w_prev[l] = self.learning_rate * np.dot(self.delta[l], self.xi[l].T) + self.momentum * self.d_w_prev[l]
            self.d_theta_prev[l] = self.learning_rate * np.sum(self.delta[l], axis=1, keepdims=True) + self.momentum * self.d_theta_prev[l]

            self.w[l] -= self.d_w_prev[l]
            self.theta[l] -= self.d_theta_prev[l]

    def fit(self, X, y):
        """
        Train the network using backpropagation.
        """
        for epoch in range(self.epochs):
            self.backpropagate(X, y)
            if epoch % 100 == 0:
                loss = np.mean((self.xi[-1] - y) ** 2)
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Perform a forward pass and return predictions.
        """
        self.forward(X)
        return self.xi[-1]

    def loss_epochs(self, X_train, y_train, X_val, y_val):
        """
        Track training and validation loss over epochs.
        """
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

# Example usage
layers = [4, 9, 5, 1]  # Example: 4 input features, 9 hidden neurons, 5 hidden neurons, 1 output neuron
nn = NeuralNet(layers)

# Example input (4 features, 5 samples)
X = np.random.randn(4, 5)

# Example output (1 target value per sample)
y = np.random.randn(1, 5)

# Train the network
nn.fit(X, y)

# Predict on new data
predictions = nn.predict(X)
print(predictions)
