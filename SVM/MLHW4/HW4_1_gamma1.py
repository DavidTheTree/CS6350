import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Main_HW3 import load_data

# Load data
train_data, test_data, attributes = load_data()

# Convert labels to {1, -1}
train_data['genuineorforged'] = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

# Extract features and labels
X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_train = train_data['genuineorforged'].values

# Hyperparameters
C_values = [100 / 873, 500 / 873, 700 / 873]
T = 100  # Number of epochs

'''
# SVM with Stochastic Sub-Gradient Descent
def svm_sgd(X, y, C, T):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights
    b = 0  # Initialize bias
    obj_values = []  # Track objective function values

    for epoch in range(T):
        # Shuffle training data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        for i in range(n_samples):
            # Compute sub-gradient
            if y[i] * (np.dot(w, X[i]) + b) < 1:
                # Misclassified point
                w = (1 - 1 / (epoch + 1)) * w + C * y[i] * X[i]
                b += C * y[i]
            else:
                # Correctly classified point
                w = (1 - 1 / (epoch + 1)) * w

        # Compute objective function value
        hinge_loss = np.maximum(0, 1 - y * (np.dot(X, w) + b)).sum()
        obj_value = 0.5 * np.dot(w, w) + C * hinge_loss
        obj_values.append(obj_value)

    return w, b, obj_values


# Run for each value of C and plot objective function curve
plt.figure(figsize=(10, 6))

for C in C_values:
    _, _, obj_values = svm_sgd(X_train, y_train, C, T)
    plt.plot(obj_values, label=f'C={C}')

plt.xlabel('Number of Updates')
plt.ylabel('Objective Function Value')
plt.title('Convergence of Objective Function')
plt.legend()
plt.grid()
plt.show()
'''


def svm_sgd_with_schedule(X_train, y_train, X_test, y_test, C, T, gamma_0, a):
    """
    SVM in primal domain with stochastic sub-gradient descent and learning rate schedule.
    """
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)  # Initialize weights
    b = 0  # Initialize bias
    train_errors = []  # Track training error for each epoch
    test_errors = []  # Track test error for each epoch
    obj_values = []  # Track objective function values

    for epoch in range(T):
        # Shuffle training data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        for i in range(n_samples):
            # Calculate learning rate for current iteration
            t = epoch * n_samples + i + 1
            gamma_t = gamma_0 / (1 + (gamma_0 / a) * t)

            # Compute sub-gradient
            if y_train[i] * (np.dot(w, X_train[i]) + b) < 1:
                # Misclassified point
                w = (1 - gamma_t) * w + gamma_t * C * y_train[i] * X_train[i]
                b += gamma_t * C * y_train[i]
            else:
                # Correctly classified point
                w = (1 - gamma_t) * w

        # Compute objective function value
        hinge_loss = np.maximum(0, 1 - y_train * (np.dot(X_train, w) + b)).sum()
        obj_value = 0.5 * np.dot(w, w) + C * hinge_loss
        obj_values.append(obj_value)

        # Calculate training and test error
        train_pred = np.sign(np.dot(X_train, w) + b)
        train_error = np.mean(train_pred != y_train)
        train_errors.append(train_error)

        test_pred = np.sign(np.dot(X_test, w) + b)
        test_error = np.mean(test_pred != y_test)
        test_errors.append(test_error)

    return w, b, obj_values, train_errors, test_errors


# Load data
train_data, test_data, attributes = load_data()
train_data['genuineorforged'] = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)
test_data['genuineorforged'] = test_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_train = train_data['genuineorforged'].values
X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_test = test_data['genuineorforged'].values

# Parameters
C_values = [100 / 873, 500 / 873, 700 / 873]
T = 100  # Epochs
gamma_0 = 0.01  # Initial learning rate
a = 1  # Learning rate decay factor

# Run and report results
for C in C_values:
    w, b, obj_values, train_errors, test_errors = svm_sgd_with_schedule(X_train, y_train, X_test, y_test, C, T, gamma_0,
                                                                        a)

    print(f"\nResults for C = {C}")
    print(f"Final Training Error: {train_errors[-1]:.4f}")
    print(f"Final Test Error: {test_errors[-1]:.4f}")

    # Plot the objective function curve
    plt.figure()
    plt.plot(obj_values, label="Objective Function")
    plt.title(f"Convergence for C = {C}")
    plt.xlabel("Number of Updates")
    plt.ylabel("Objective Function Value")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the training and test error
    plt.figure()
    plt.plot(train_errors, label="Training Error")
    plt.plot(test_errors, label="Test Error")
    plt.title(f"Error Rates for C = {C}")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.grid()
    plt.show()

