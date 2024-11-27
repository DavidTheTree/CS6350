import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Main_HW3 import load_data
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


# Gaussian kernel function
def gaussian_kernel(X, gamma):
    """
    Computes the Gaussian kernel Gram matrix for the dataset X.
    """
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-np.linalg.norm(X[i] - X[j]) ** 2 / gamma)
    return K


# Dual objective function with Gaussian kernel
def dual_objective_gaussian(alpha, K, y):
    """
    Dual objective function using Gaussian kernel.
    Minimize: 1/2 * alpha^T G alpha - 1^T alpha
    """
    G = np.outer(y, y) * K
    obj = 0.5 * np.dot(alpha, np.dot(G, alpha)) - np.sum(alpha)
    return obj


def equality_constraint(alpha, y):
    """
    Constraint: sum(alpha * y) = 0
    """
    return np.dot(alpha, y)


def predict_gaussian(X_train, X_test, alpha, y_train, gamma, b):
    """
    Make predictions using the Gaussian kernel.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    predictions = np.zeros(n_test)

    for i in range(n_test):
        for j in range(n_train):
            kernel_value = np.exp(-np.linalg.norm(X_train[j] - X_test[i]) ** 2 / gamma)
            predictions[i] += alpha[j] * y_train[j] * kernel_value
        predictions[i] += b
    return np.sign(predictions)


def dual_svm_gaussian(X_train, y_train, X_test, y_test, C, gamma):
    """
    Solves the dual form of SVM using Gaussian kernel.
    """
    n_samples = X_train.shape[0]
    K = gaussian_kernel(X_train, gamma)  # Compute the Gaussian kernel matrix

    initial_alpha = np.zeros(n_samples)
    bounds = [(0, C) for _ in range(n_samples)]
    constraints = {"type": "eq", "fun": equality_constraint, "args": (y_train,)}

    print(f"\nStarting dual SVM with Gaussian kernel (C={C}, gamma={gamma})...")
    result = minimize(
        fun=dual_objective_gaussian,
        x0=initial_alpha,
        args=(K, y_train),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-6, "maxiter": 500, "disp": False}
    )

    if not result.success:
        print(f"Optimization failed: {result.message}")
        return None, None, None

    alpha = result.x
    w = None  # w is not explicitly computed in non-linear SVM
    support_vectors = np.where((alpha > 1e-5) & (alpha < C))[0]
    b = np.mean([y_train[k] - np.sum(
        alpha * y_train * np.exp(-np.linalg.norm(X_train - X_train[k], axis=1) ** 2 / gamma)
    ) for k in support_vectors])

    # Training and test predictions
    train_predictions = predict_gaussian(X_train, X_train, alpha, y_train, gamma, b)
    test_predictions = predict_gaussian(X_train, X_test, alpha, y_train, gamma, b)

    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    return train_error, test_error


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
    C_values = [100 / 873, 500 / 873, 700 / 873]
    gamma_values = [0.1, 0.5, 1, 5, 100]

    results = []

    for C in C_values:
        for gamma in gamma_values:
            train_error, test_error = dual_svm_gaussian(X_train, y_train, X_test, y_test, C, gamma)
            results.append((C, gamma, train_error, test_error))
            print(f"C={C}, gamma={gamma}, Train Error={train_error:.4f}, Test Error={test_error:.4f}")

    # Display results
    print("\nResults Summary:")
    for result in results:
        print(f"C={result[0]}, gamma={result[1]}, Train Error={result[2]:.4f}, Test Error={result[3]:.4f}")