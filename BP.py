import numpy as np

class NeuralNet:
  class NeuralNet:
    def __init__(self, layers, epochs=1000, learning_rate=0.01, momentum=0.9, activation_function='sigmoid', validation_split=0.2):
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

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def forward(self, X):
        self.xi[0] = X  # Set input data as the first layer activations
        for i in range(1, self.L):
            z = np.dot(self.w[i - 1], self.xi[i - 1]) + self.theta[i - 1]  # Weighted sum + bias
            self.xi[i] = self.activation_function(z)  # Activation function applied

    def backpropagate(self, X, y):
        m = X.shape[1]  # Number of samples
        self.forward(X)  # Perform forward pass
        
        # Compute error for the output layer
        delta_output = self.xi[-1] - y
        self.delta[-1] = delta_output * self.activation_derivative(self.xi[-1])  # Delta for output layer

        # Backpropagate the error to the hidden layers
        for i in range(self.L - 2, -1, -1):
            self.delta[i] = np.dot(self.w[i + 1].T, self.delta[i + 1]) * self.activation_derivative(self.xi[i + 1])

        # Compute the gradients
        for i in range(self.L - 1):
            self.d_w[i] = np.dot(self.delta[i + 1], self.xi[i].T) / m
            self.d_theta[i] = np.sum(self.delta[i + 1], axis=1, keepdims=True) / m

        # Update weights and biases using gradient descent with momentum
        for i in range(self.L - 1):
            self.d_w[i] = self.d_w[i] + self.momentum * self.d_w_prev[i]  # Add momentum term
            self.d_theta[i] = self.d_theta[i] + self.momentum * self.d_theta_prev[i]  # Add momentum term
            self.w[i] -= self.learning_rate * self.d_w[i]  # Update weights
            self.theta[i] -= self.learning_rate * self.d_theta[i]  # Update biases

            # Store previous changes for momentum
            self.d_w_prev[i] = self.d_w[i]
            self.d_theta_prev[i] = self.d_theta[i]

    def fit(self, X, y):
        # Train the network using backpropagation
        for epoch in range(self.epochs):
            self.backpropagate(X, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs} completed.")

    def predict(self, X):
        # Perform a forward pass and return the prediction
        self.forward(X)
        return self.xi[-1]

    def loss_epochs(self, X_train, y_train, X_val, y_val):
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            self.fit(X_train, y_train)
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)

            train_loss = np.mean((train_pred - y_train) ** 2)  # Mean Squared Error for training set
            val_loss = np.mean((val_pred - y_val) ** 2)  # Mean Squared Error for validation set

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
