import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Main_HW3 import load_data
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Gaussian kernel matrix computation
def gaussian_kernel_matrix(X, gamma):
    pairwise_sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
                        np.sum(X**2, axis=1).reshape(1, -1) - \
                        2 * np.dot(X, X.T)
    return np.exp(-pairwise_sq_dists / gamma)

# Dual objective function in matrix form
def dual_objective(alpha, G):
    return 0.5 * np.dot(alpha, np.dot(G, alpha)) - np.sum(alpha)

# Equality constraint: sum(alpha * y) = 0
def equality_constraint(alpha, y):
    return np.dot(alpha, y)

# Dual SVM optimization
def dual_svm_gaussian_matrix(X_train, y_train, C, gamma):
    n_samples = X_train.shape[0]
    K = gaussian_kernel_matrix(X_train, gamma)
    G = np.outer(y_train, y_train) * K

    initial_alpha = np.zeros(n_samples)
    bounds = [(0, C) for _ in range(n_samples)]
    constraints = {"type": "eq", "fun": equality_constraint, "args": (y_train,)}

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
        return None, None

    alpha = result.x
    support_vectors = np.where((alpha > 1e-5) & (alpha < C))[0]

    # Compute bias
    b = np.mean([
        y_train[k] - np.sum(alpha * y_train * K[:, k])
        for k in support_vectors
    ])

    return alpha, b, support_vectors

# Main script
if __name__ == "__main__":
    # Load data
    train_data, test_data, attributes = load_data()
    train_data['genuineorforged'] = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

    X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
    y_train = train_data['genuineorforged'].values

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Hyperparameters
    C_values = [100 / 873, 500 / 873, 700 / 873]
    gamma_values = [0.01, 0.1, 0.5, 1, 5]

    # Record results
    results = {}
    for C in C_values:
        results[C] = {}
        for gamma in gamma_values:
            alpha, b, support_vectors = dual_svm_gaussian_matrix(X_train, y_train, C, gamma)
            if support_vectors is not None:
                results[C][gamma] = {
                    "support_vectors": support_vectors,
                    "num_support_vectors": len(support_vectors),
                }
                print(f"C={C}, gamma={gamma}, Support Vectors={len(support_vectors)}")

    # Compute overlap for C=500/873
    overlap_results = []
    C_target = 500 / 873
    prev_support_vectors = None
    for gamma in gamma_values:
        if prev_support_vectors is not None:
            current_support_vectors = results[C_target][gamma]["support_vectors"]
            overlap = len(np.intersect1d(prev_support_vectors, current_support_vectors))
            overlap_results.append((prev_gamma, gamma, overlap))
            print(f"Overlap between gamma={prev_gamma} and gamma={gamma}: {overlap}")
        prev_support_vectors = results[C_target][gamma]["support_vectors"]
        prev_gamma = gamma

    # Summary of overlaps
    print("\nOverlap Results for C=500/873:")
    for prev_gamma, gamma, overlap in overlap_results:
        print(f"Gamma={prev_gamma} -> Gamma={gamma}, Overlap={overlap}")