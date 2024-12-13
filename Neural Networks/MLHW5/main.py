import numpy as np
import pandas as pd

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Load Data Function (unchanged)
def load_data():
    train_data = pd.read_csv('./BankNote/train.csv', header=None)
    test_data = pd.read_csv('./BankNote/test.csv', header=None)

    attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'genuineorforged']
    train_data.columns = attributes
    test_data.columns = attributes

    return train_data, test_data, attributes

# Initialize parameters
def initialize_parameters(input_dim, hidden_dim1, hidden_dim2, output_dim):
    params = {
        'W1': np.random.randn(input_dim, hidden_dim1) * 0.01,
        'b1': np.zeros((1, hidden_dim1)),
        'W2': np.random.randn(hidden_dim1, hidden_dim2) * 0.01,
        'b2': np.zeros((1, hidden_dim2)),
        'W3': np.random.randn(hidden_dim2, output_dim) * 0.01,
        'b3': np.zeros((1, output_dim))
    }
    return params

# Forward propagation for one example
def forward_propagation_single(X, params):
    # Layer 1
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = sigmoid(Z1)
    A1[0, 0] = params['b1'][0, 0]  # Disconnect z01

    # Layer 2
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = sigmoid(Z2)
    A2[0, 1] = params['b2'][0, 1]  # Disconnect z02

    # Layer 3 (Output)
    Z3 = np.dot(A2, params['W3']) + params['b3']
    A3 = sigmoid(Z3)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache

# Backward propagation for one example
def backward_propagation_single(X, y, params, cache):
    # Extract cached values
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    Z1, Z2, Z3 = cache['Z1'], cache['Z2'], cache['Z3']

    # Gradients for output layer
    dZ3 = A3 - y  # Derivative of loss w.r.t Z3
    dW3 = np.outer(A2, dZ3)  # Gradient of W3
    db3 = dZ3  # Gradient of b3

    # Gradients for Layer 2
    dZ2 = np.dot(dZ3, params['W3'].T) * sigmoid_derivative(Z2)
    dZ2[0, 1] = 0  # Disconnect z02
    dW2 = np.outer(A1, dZ2)
    db2 = dZ2

    # Gradients for Layer 1
    dZ1 = np.dot(dZ2, params['W2'].T) * sigmoid_derivative(Z1)
    dZ1[0, 0] = 0  # Disconnect z01
    dW1 = np.outer(X, dZ1)
    db1 = dZ1

    # Store gradients
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return grads

# Test the backpropagation with one example
if __name__ == "__main__":
    # Load data
    train_data, test_data, attributes = load_data()
    X_train = train_data[attributes[:-1]].values
    y_train = train_data['genuineorforged'].values

    # Use a single training example for debugging
    X_example = X_train[0].reshape(1, -1)  # First example
    y_example = y_train[0]  # First label

    # Initialize parameters
    input_dim = X_example.shape[1]
    hidden_dim1 = 3  # Example: 3 nodes in Layer 1
    hidden_dim2 = 3  # Example: 3 nodes in Layer 2
    output_dim = 1  # Binary classification
    params = initialize_parameters(input_dim, hidden_dim1, hidden_dim2, output_dim)

    # Forward propagation
    A3, cache = forward_propagation_single(X_example, params)

    # Backward propagation
    grads = backward_propagation_single(X_example, y_example, params, cache)

    # Print results for debugging
    print("Output of forward pass (A3):", A3)
    print("Gradients:")
    for key, value in grads.items():
        print(f"{key}: {value}")
