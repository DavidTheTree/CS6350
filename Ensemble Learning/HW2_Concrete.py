import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    train_data = pd.read_csv('./data/concrete/train.csv', header=None)
    test_data = pd.read_csv('./data/concrete/test.csv', header=None)

    attributes = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'label']
    train_data.columns = attributes
    test_data.columns = attributes

    # Int to binary conversion
    int_columns = [col for col in train_data.columns if train_data[col].dtype == 'int64']

    def numerical_to_binary(data, int_columns):
        for column in int_columns:
            median = data[column].median()
            data[column] = np.where(data[column] > median, 1, 0)
        return data

    train_data = numerical_to_binary(train_data, int_columns)
    test_data = numerical_to_binary(test_data, int_columns)

    return train_data, test_data, attributes


def batch_gradient_descent(X, y, learning_rate, tolerance, max_iterations):
    m, n = X.shape
    weights = np.zeros(n)  # Initialize weight vector to zero
    cost_history = []

    for iteration in range(max_iterations):
        # Calculate predictions
        predictions = X.dot(weights)

        # Compute errors
        errors = predictions - y

        # Compute gradient
        gradient = (1 / m) * X.T.dot(errors)

        # Update weights
        weights_new = weights - learning_rate * gradient

        # Compute the change in weights (norm)
        weight_diff = np.linalg.norm(weights_new - weights)

        # Update weights
        weights = weights_new

        # Calculate cost (Mean Squared Error)
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)

        # Check for convergence
        if weight_diff < tolerance:
            print(f'Convergence reached at iteration {iteration} with learning rate {learning_rate}')
            break

    return weights, cost_history


def normalize_features(X):
    return (X - X.mean()) / X.std()


def plot_cost(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label='Cost Function')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function vs. Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()


# Load the data
train_data, test_data, attributes = load_data()

# Prepare data for gradient descent
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_train = normalize_features(X_train)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Add bias term

X_test = test_data.drop('label', axis=1)
y_test = test_data['label']
X_test = normalize_features(X_test)
X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Add bias term

# Parameters
learning_rate = 1.0
tolerance = 1e-6
max_iterations = 10000

# Start with a high learning rate and decrease until convergence
while True:
    weights, cost_history = batch_gradient_descent(X_train, y_train, learning_rate, tolerance, max_iterations)
    if len(cost_history) < max_iterations:
        break
    else:
        learning_rate /= 2  # Decrease learning rate

# Plot the cost function
plot_cost(cost_history)

# Calculate cost function for test data using final weights
test_predictions = X_test.dot(weights)
test_cost = (1 / (2 * X_test.shape[0])) * np.sum((test_predictions - y_test) ** 2)

# Report results
print(f'Learned weights: {weights}')
print(f'Final learning rate: {learning_rate}')
print(f'Test cost: {test_cost}')