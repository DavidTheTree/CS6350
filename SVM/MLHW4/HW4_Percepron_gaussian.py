import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Main_HW3 import load_data
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Gaussian kernel function
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / gamma)

# Kernel Perceptron
def kernel_perceptron(X_train, y_train, X_test, y_test, gamma, max_iter=10):
    n_samples = X_train.shape[0]
    alphas = np.zeros(n_samples)  # Coefficients for the kernel perceptron
    support_vectors = []  # Store support vector indices

    # Training
    for _ in range(max_iter):
        for i in range(n_samples):
            # Compute kernelized prediction
            f_xi = sum(
                alphas[j] * y_train[j] * gaussian_kernel(X_train[j], X_train[i], gamma)
                for j in range(n_samples)
            )
            # Update if misclassified
            if y_train[i] * f_xi <= 0:
                alphas[i] += 1

    # Store support vector indices
    support_vectors = np.where(alphas > 0)[0]

    # Predictions
    def predict(X):
        n_test = X.shape[0]
        predictions = np.zeros(n_test)
        for i in range(n_test):
            predictions[i] = sum(
                alphas[j] * y_train[j] * gaussian_kernel(X_train[j], X[i], gamma)
                for j in range(n_samples)
            )
        return np.sign(predictions)

    # Compute training and test errors
    train_predictions = predict(X_train)
    test_predictions = predict(X_test)
    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    return train_error, test_error, len(support_vectors)

# Main script
if __name__ == "__main__":
    # Load data
    train_data, test_data, attributes = load_data()
    train_data['genuineorforged'] = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)
    test_data['genuineorforged'] = test_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

    X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
    y_train = train_data['genuineorforged'].values
    X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values
    y_test = test_data['genuineorforged'].values

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameters
    gamma_values = [0.1, 0.5, 1, 5, 100]

    # Record results
    results = []
    for gamma in gamma_values:
        train_error, test_error, num_support_vectors = kernel_perceptron(X_train, y_train, X_test, y_test, gamma)
        results.append((gamma, train_error, test_error, num_support_vectors))
        print(f"Gamma={gamma}, Train Error={train_error:.4f}, Test Error={test_error:.4f}, Support Vectors={num_support_vectors}")

    # Display results
    print("\nResults Summary:")
    for result in results:
        print(f"Gamma={result[0]}, Train Error={result[1]:.4f}, Test Error={result[2]:.4f}, Support Vectors={result[3]}")
