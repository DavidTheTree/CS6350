import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
def initialize_parameters(input_dim, hidden_dim, output_dim):
    params = {
        'W1': np.zeros((input_dim, hidden_dim)),
        'b1': np.zeros((1, hidden_dim)),
        'W2': np.zeros((hidden_dim, hidden_dim)),
        'b2': np.zeros((1, hidden_dim)),
        'W3': np.zeros((hidden_dim, output_dim)),
        'b3': np.zeros((1, output_dim))
    }
    return params

# Forward propagation
def forward_propagation(X, params):
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, params['W3']) + params['b3']
    A3 = sigmoid(Z3)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache

# Backward propagation
def backward_propagation(X, y, params, cache):
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    Z1, Z2, Z3 = cache['Z1'], cache['Z2'], cache['Z3']

    dZ3 = A3 - y.reshape(-1, 1)
    dW3 = np.dot(A2.T, dZ3) / X.shape[0]
    db3 = np.sum(dZ3, axis=0, keepdims=True) / X.shape[0]

    dZ2 = np.dot(dZ3, params['W3'].T) * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

    dZ1 = np.dot(dZ2, params['W2'].T) * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return grads

# Update parameters
def update_parameters(params, grads, learning_rate):
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    params['W3'] -= learning_rate * grads['dW3']
    params['b3'] -= learning_rate * grads['db3']
    return params

# Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(X_train, y_train, X_test, y_test, widths, gamma0, d, epochs):
    input_dim = X_train.shape[1]
    output_dim = 1

    results = {}

    for hidden_dim in widths:
        params = initialize_parameters(input_dim, hidden_dim, output_dim)
        losses = []

        for epoch in range(epochs):
            # Shuffle the training data
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(X_train.shape[0]):
                X_sample = X_train_shuffled[i].reshape(1, -1)
                y_sample = y_train_shuffled[i]

                # Learning rate schedule
                learning_rate = gamma0 / (1 + (gamma0 / d) * (epoch * X_train.shape[0] + i))

                # Forward and backward propagation
                A3, cache = forward_propagation(X_sample, params)
                grads = backward_propagation(X_sample, y_sample, params, cache)

                # Update parameters
                params = update_parameters(params, grads, learning_rate)

            # Compute loss for the epoch
            A3_train, _ = forward_propagation(X_train, params)
            loss = -np.mean(y_train * np.log(A3_train) + (1 - y_train) * np.log(1 - A3_train))
            losses.append(loss)

            if epoch % 10 == 0:
                print(f"Width: {hidden_dim}, Epoch: {epoch}, Loss: {loss:.4f}")

        # Compute training and test error
        A3_train, _ = forward_propagation(X_train, params)
        train_error = np.mean((A3_train > 0.5).astype(int).flatten() != y_train)

        A3_test, _ = forward_propagation(X_test, params)
        test_error = np.mean((A3_test > 0.5).astype(int).flatten() != y_test)

        results[hidden_dim] = {
            'train_error': train_error,
            'test_error': test_error,
            'loss_curve': losses
        }

        # Plot loss curve
        plt.plot(losses, label=f'Width: {hidden_dim}')

    plt.title('Loss Curve for Different Hidden Layer Widths')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return results

# Main execution
if __name__ == "__main__":
    # Load data
    train_data, test_data, attributes = load_data()
    X_train = train_data[attributes[:-1]].values
    y_train = train_data['genuineorforged'].values
    X_test = test_data[attributes[:-1]].values
    y_test = test_data['genuineorforged'].values

    # Hyperparameters
    widths = [5, 10, 25, 50, 100]
    gamma0 = 0.1
    d = 100
    epochs = 50

    # Run SGD
    results = stochastic_gradient_descent(X_train, y_train, X_test, y_test, widths, gamma0, d, epochs)

    # Print results
    for width, result in results.items():
        print(f"Width: {width}")
        print(f"  Train Error: {result['train_error']:.4f}")
        print(f"  Test Error: {result['test_error']:.4f}")
