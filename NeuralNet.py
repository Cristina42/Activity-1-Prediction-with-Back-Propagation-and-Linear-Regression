import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load the dataset
data = pd.read_csv('Life Expectancy Data.csv')

# Remove rows with missing data
data = data.dropna()

# Drop the 'Country' column if it exists
if 'Country' in data.columns:
    data = data.drop(columns=['Country'])

# Encode the 'Status' column
encoder = OneHotEncoder(sparse_output=False, drop='first')  
status_encoded = encoder.fit_transform(data[['Status']]) 
status_columns = encoder.get_feature_names_out(['Status'])  
status_df = pd.DataFrame(status_encoded, columns=status_columns) 
data = pd.concat([data.drop(columns=['Status']).reset_index(drop=True), status_df], axis=1) 

# Normalize the data
scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns  
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])  

# Save the processed data to a new CSV file
data.to_csv('Processed_Life_Expectancy_Data.csv', index=False)

print("Preprocessing complete. Processed data saved to 'Processed_Life_Expectancy_Data.csv'.")


# Split dataset 
X = data.drop(columns=['Life expectancy '])  # Features
y = data['Life expectancy ']  # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

class NeuralNet:
    def __init__(self, layers, learning_rate=0.01, epochs=100, activation='sigmoid'):
        self.L = len(layers)  
        self.n = layers  
        self.learning_rate = learning_rate
        self.epochs = epochs

      
        self.w = [np.random.randn(layers[l], layers[l-1]) * 0.01 for l in range(1, self.L)]
        self.theta = [np.zeros((layers[l], 1)) for l in range(1, self.L)]
        self.xi = [np.zeros((layers[l], 1)) for l in range(self.L)]
        self.delta = [np.zeros((layers[l], 1)) for l in range(self.L)]

       
        if activation == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'relu':
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative
        else:
            raise ValueError("Activation function must be 'sigmoid' or 'relu'.")

   
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)


    def forward_propagation(self, X):

        A = X.T  
        for l in range(len(self.w)):
            Z = np.dot(self.w[l], A) 
            A = np.maximum(0, Z) if l < len(self.w) - 1 else Z  
        return A.T

    def backward_propagation(self, X, y, learning_rate):
    
        A = self.forward_propagation(X)
        dA = A.T - y.values.reshape(-1, 1)
        for l in reversed(range(len(self.w))):
            dW = np.dot(dA, X)  
            self.w[l] -= learning_rate * dW

    def fit(self, X, y, epochs, learning_rate):

        for epoch in range(epochs):
            self.backward_propagation(X, y, learning_rate)
            if epoch % 10 == 0:
                predictions = self.forward_propagation(X)
                loss = np.mean((predictions - y.values.reshape(-1, 1)) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward_propagation(X)



# print("L = ", nn.L, end="\n")
# print("n = ", nn.n, end="\n")

# print("xi = ", nn.xi, end="\n")
# print("xi[0] = ", nn.xi[0], end="\n")
# print("xi[1] = ", nn.xi[0], end="\n")

# print("wh = ", nn.w, end="\n")
# print("wh[1] = ", nn.w[1], end="\n")
