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

 # Initialize activations, weights, biases, and other necessary parameters
        self.xi = [np.zeros((layer, 1)) for layer in layers]  # Activations
        self.w = [np.random.randn(layers[i], layers[i-1]) * 0.01 for i in range(1, self.L)]  # Weights
        self.theta = [np.zeros((layer, 1)) for layer in layers[1:]]  # Biases
        self.delta = [np.zeros((layer, 1)) for layer in layers[1:]]  # Error terms
        self.d_w = [np.zeros_like(w) for w in self.w]  # Weight changes
        self.d_theta = [np.zeros_like(t) for t in self.theta]  # Bias changes
        self.d_w_prev = [np.zeros_like(w) for w in self.w]  # Previous weight changes (for momentum)
        self.d_theta_prev = [np.zeros_like(t) for t in self.theta]  # Previous bias changes (for momentum)

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
        return 1 - np.tanh(z)**2

    def activation_function(self, z):
        if self.fact == 'sigmoid':
            return self.sigmoid(z)
        elif self.fact == 'relu':
            return self.relu(z)
        elif self.fact == 'linear':
            return self.linear(z)
        elif self.fact == 'tanh':
            return self.tanh(z)

    def activation_derivative(self, z):
        if self.fact == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.fact == 'relu':
            return self.relu_derivative(z)
        elif self.fact == 'linear':
            return self.linear_derivative(z)
        elif self.fact == 'tanh':
            return self.tanh_derivative(z)

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

    self.xi = []
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))


layers = [4, 9, 5, 1]
nn = NeuralNet(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
