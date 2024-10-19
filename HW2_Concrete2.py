import pandas as pd
import numpy as np
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


# Define the stochastic gradient descent (SGD) algorithm
def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=100, convergence_threshold=1e-6):
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

            # Check for convergence
            if epoch > 0 and abs(cost_history[-1] - cost_history[-2]) < convergence_threshold:
                break

    return weights, cost_history


# Set the paths for your train and test datasets
train_path = './data/concrete/train.csv'  # Replace with your actual path
test_path = './data/concrete/test.csv'  # Replace with your actual path

# Load the data
train_data, test_data, attributes = load_data(train_path, test_path)

# Split data into features and labels
X_train = train_data.drop(columns='label').values
y_train = train_data['label'].values
X_test = test_data.drop(columns='label').values
y_test = test_data['label'].values

# Set the parameters for SGD
learning_rate = 0.01
num_epochs = 50

# Run SGD
weights, cost_history = stochastic_gradient_descent(X_train, y_train, learning_rate, num_epochs)

# Plot the cost function over iterations
plt.figure(figsize=(10, 6))
plt.plot(cost_history, label='Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost (Mean Squared Error)')
plt.title('Cost Function Convergence using SGD')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate on test data
test_predictions = np.dot(X_test, weights)
test_cost = (1 / (2 * len(y_test))) * np.sum((test_predictions - y_test) ** 2)

# Output the results
print(f'Learned Weights: {weights}')
print(f'Chosen Learning Rate: {learning_rate}')
print(f'Test Cost: {test_cost}')



