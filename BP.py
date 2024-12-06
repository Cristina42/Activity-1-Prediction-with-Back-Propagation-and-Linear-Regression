# our own implemented BP model

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
        self.w = [np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1]) for i in range(1, self.L)] # check this
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
