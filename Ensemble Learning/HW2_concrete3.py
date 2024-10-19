import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load data function
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

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


def analytical_solution(X, y):
    X_transpose = X.T
    optimal_weights = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    return optimal_weights

# Batch Gradient Descent
def batch_gradient_descent(X, y, learning_rate=0.01, num_epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    cost_history = []

    for epoch in range(num_epochs):
        # Calculate predictions
        predictions = np.dot(X, weights)

        # Calculate the error
        error = predictions - y

        # Calculate the gradient
        gradient = (1 / n_samples) * np.dot(X.T, error)

        # Update the weights
        weights -= learning_rate * gradient

        # Calculate the cost (mean squared error)
        cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
        cost_history.append(cost)

    return weights, cost_history

# Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    cost_history = []

    for epoch in range(num_epochs):
        for i in range(n_samples):
            # Randomly sample an index
            random_index = np.random.randint(n_samples)
            X_i = X[random_index]
            y_i = y[random_index]

            # Calculate the prediction
            prediction = np.dot(X_i, weights)

            # Calculate the error
            error = prediction - y_i

            # Calculate the stochastic gradient
            gradient = X_i * error

            # Update the weights
            weights -= learning_rate * gradient

            # Calculate the cost (mean squared error)
            cost = (1 / (2 * n_samples)) * np.sum((np.dot(X, weights) - y) ** 2)
            cost_history.append(cost)

    return weights, cost_history

# Set the paths for your train dataset
train_path = './data/concrete/train.csv'  # Replace with your actual path
test_path = './data/concrete/test.csv'    # Replace with your actual path

# Load the data
train_data, test_data, attributes = load_data(train_path, test_path)

# Split data into features and labels
X_train = train_data.drop(columns='label').values
y_train = train_data['label'].values

# Calculate weights using different methods
analytical_weights = analytical_solution(X_train, y_train)

# Set learning parameters
learning_rate = 0.01
num_epochs = 100

# Batch Gradient Descent
batch_weights, batch_cost_history = batch_gradient_descent(X_train, y_train, learning_rate, num_epochs)

# Stochastic Gradient Descent
sgd_weights, sgd_cost_history = stochastic_gradient_descent(X_train, y_train, learning_rate, num_epochs)

# Print the results
print(f'Optimal Weights (Analytical): {analytical_weights}')
print(f'Weights (Batch GD): {batch_weights}')
print(f'Weights (SGD): {sgd_weights}')

# Plot cost function convergence
plt.figure(figsize=(12, 6))
plt.plot(batch_cost_history, label='Batch GD Cost')
plt.plot(sgd_cost_history, label='SGD Cost', alpha=0.7)
plt.xlabel('Iterations')
plt.ylabel('Cost (Mean Squared Error)')
plt.title('Cost Function Convergence')
plt.legend()
plt.grid(True)
plt.show()