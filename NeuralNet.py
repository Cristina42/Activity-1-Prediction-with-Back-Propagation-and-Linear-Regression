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
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Convert 'Status' to numbers
status_encoded = encoder.fit_transform(data[['Status']])  # One-hot encoding
status_columns = encoder.get_feature_names_out(['Status'])  # Get new column names
status_df = pd.DataFrame(status_encoded, columns=status_columns)  # Create a DataFrame
data = pd.concat([data.drop(columns=['Status']).reset_index(drop=True), status_df], axis=1)  # Add encoded columns

# Normalize the data
scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns  # Select numerical columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])  # Normalize the data

# Save the processed data to a new CSV file
data.to_csv('Processed_Life_Expectancy_Data.csv', index=False)

print("Preprocessing complete. Processed data saved to 'Processed_Life_Expectancy_Data.csv'.")


# Split dataset 
X = data.drop(columns=['Life expectancy '])  # Features
y = data['Life expectancy ']  # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

class NeuralNet:
  def __init__(self, layers):
    self.L = len(layers)
    self.n = layers.copy()

    self.xi = []
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

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





layers = [4, 9, 5, 1]
nn = NeuralNet(layers)


print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
