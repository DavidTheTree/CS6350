import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Main_HW3 import load_data
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Gaussian kernel matrix computation
def gaussian_kernel_matrix(X, gamma):
    """
    Computes the Gaussian kernel Gram matrix for dataset X.
    """
    pairwise_sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
                        np.sum(X**2, axis=1).reshape(1, -1) - \
                        2 * np.dot(X, X.T)
    return np.exp(-pairwise_sq_dists / gamma)

# Dual objective function in matrix form
def dual_objective(alpha, G):
    """
    Computes the dual objective function using matrix operations.
    """
    obj = 0.5 * np.dot(alpha, np.dot(G, alpha)) - np.sum(alpha)
    return obj

# Equality constraint: sum(alpha * y) = 0
def equality_constraint(alpha, y):
    """
    Constraint: sum(alpha * y) = 0
    """
    return np.dot(alpha, y)

# Dual SVM optimization
def dual_svm_gaussian_matrix(X_train, y_train, C, gamma):
    """
    Solves the dual SVM problem using Gaussian kernel and matrix operations.
    """
    n_samples = X_train.shape[0]
    K = gaussian_kernel_matrix(X_train, gamma)  # Compute Gaussian kernel matrix
    G = np.outer(y_train, y_train) * K  # Gram matrix incorporating labels

    initial_alpha = np.zeros(n_samples)  # Initial guess for alpha
    bounds = [(0, C) for _ in range(n_samples)]  # Bounds: 0 <= alpha_i <= C
    constraints = {"type": "eq", "fun": equality_constraint, "args": (y_train,)}

    # Minimize the dual objective
    print(f"Starting optimization with C={C}, gamma={gamma}...")
    result = minimize(
        fun=dual_objective,
        x0=initial_alpha,
        args=(G,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-6, "maxiter": 500, "disp": False}
    )

    if not result.success:
        print(f"Optimization failed: {result.message}")
        return None, None, None

    alpha = result.x

    # Compute bias b
    support_vectors = np.where((alpha > 1e-5) & (alpha < C))[0]
    b = np.mean([
        y_train[k] - np.sum(alpha * y_train * K[:, k])
        for k in support_vectors
    ])

    return alpha, b

# Prediction using Gaussian kernel
def predict_gaussian(X_train, X_test, alpha, y_train, gamma, b):
    """
    Predict labels for test data using the Gaussian kernel.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    pairwise_sq_dists = np.sum(X_test**2, axis=1).reshape(-1, 1) + \
                        np.sum(X_train**2, axis=1).reshape(1, -1) - \
                        2 * np.dot(X_test, X_train.T)
    K_test = np.exp(-pairwise_sq_dists / gamma)

    predictions = np.dot(K_test, alpha * y_train) + b
    return np.sign(predictions)

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

    # Record results
    results = []

    for C in C_values:
        for gamma in gamma_values:
            alpha, b = dual_svm_gaussian_matrix(X_train, y_train, C, gamma)
            if alpha is not None:
                train_predictions = predict_gaussian(X_train, X_train, alpha, y_train, gamma, b)
                test_predictions = predict_gaussian(X_train, X_test, alpha, y_train, gamma, b)
                train_error = np.mean(train_predictions != y_train)
                test_error = np.mean(test_predictions != y_test)
                results.append((C, gamma, train_error, test_error))
                print(f"C={C}, gamma={gamma}, Train Error={train_error:.4f}, Test Error={test_error:.4f}")

    # Display results
    print("\nResults Summary:")
    for result in results:
        print(f"C={result[0]}, gamma={result[1]}, Train Error={result[2]:.4f}, Test Error={result[3]:.4f}")