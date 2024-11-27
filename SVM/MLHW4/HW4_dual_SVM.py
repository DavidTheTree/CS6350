import numpy as np
from scipy.optimize import minimize
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from Main_HW3 import load_data
import sys
from time import sleep

# Load and prepare data
train_data, test_data, attributes = load_data()
train_data['genuineorforged'] = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_train = train_data['genuineorforged'].values


# Dual objective function
def dual_objective(alpha, X, y):
    """
    Objective function for the dual form of SVM.
    """
    alpha = np.clip(alpha, 0, C)  # Clip to bounds
    K = np.dot(X, X.T)  # Linear kernel
    obj = -np.sum(alpha) + 0.5 * np.sum(np.outer(alpha, alpha) * np.outer(y, y) * K)
    return obj


# Equality constraint
def equality_constraint(alpha, y):
    """
    Constraint: sum(alpha * y) = 0
    """
    return np.dot(alpha, y)


# Print progress
def print_progress(current, total, prefix="Progress"):
    """
    Prints progress as a percentage.
    """
    progress = (current / total) * 100
    print(f"\r{prefix}: {progress:.2f}% complete", end='')


# Dual SVM optimization
def dual_svm_debug(X, y, C):
    """
    Dual SVM with debugging to resolve bounds and convergence issues.
    """
    n_samples = X.shape[0]
    initial_alpha = np.zeros(n_samples)  # Feasible starting point
    bounds = [(0, C) for _ in range(n_samples)]
    constraints = {"type": "eq", "fun": equality_constraint, "args": (y,)}

    print("\nStarting dual SVM optimization...")
    print(f"Number of samples: {n_samples}")
    print(f"Hyperparameter C: {C}")

    # Simulate progress during optimization
    total_steps = 100  # Total steps for progress tracking
    for step in range(total_steps):
        print_progress(step + 1, total_steps, prefix="Optimization Progress")

    # Perform optimization
    result = minimize(
        fun=dual_objective,
        x0=initial_alpha,
        args=(X, y),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-6, "maxiter": 500, "disp": True}  # Adjust tolerances
    )

    print("\nOptimization complete!")
    if result.success:
        print("Optimization converged successfully.")
    else:
        print(f"Optimization failed: {result.message}")

    # Recover parameters
    alpha = result.x
    w = np.sum((alpha * y)[:, None] * X, axis=0)
    support_vectors = np.where((alpha > 1e-5) & (alpha < C))[0]
    b = np.mean([y[k] - np.dot(w, X[k]) for k in support_vectors])

    print("\nRecovered parameters:")
    print(f"Weights (w): {w}")
    print(f"Bias (b): {b}")
    print(f"Number of support vectors: {len(support_vectors)}")

    return w, b, alpha, result.fun


# Main script
if __name__ == "__main__":
    # Load data
    train_data, test_data, attributes = load_data()
    train_data['genuineorforged'] = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

    # Extract features and labels
    X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
    y_train = train_data['genuineorforged'].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Run dual SVM for multiple values of C
    C_values = [100 / 873, 500 / 873, 700 / 873]
    for idx, C in enumerate(C_values, 1):
        print(f"\n--- Running dual SVM for C = {C} (Iteration {idx}/{len(C_values)}) ---")
        w_dual, b_dual, alpha, dual_obj = dual_svm_debug(X_train, y_train, C)
        print(f"\nResults for C = {C}:")
        print(f"Weights: {w_dual}")
        print(f"Bias: {b_dual}")
        print(f"Dual Objective Value: {dual_obj}")
        print("\n" + "=" * 50 + "\n")

'''
Learned weight vector: [-61.086591 -42.70582  -40.30786   -3.146269]
Bias: 53.0
Average prediction error on the test dataset: 0.02

--- Running dual SVM for C = 0.1145475372279496 (Iteration 1/3) ---

Starting dual SVM optimization...
Number of samples: 872
Hyperparameter C: 0.1145475372279496
Optimization Progress: 100.00% completeC:\Users\u1523013\PycharmProjects\pythonProject\.venv\Lib\site-packages\scipy\optimize\_slsqp_py.py:434: RuntimeWarning:

Values in x were outside bounds during a minimize step, clipping to bounds

Optimization terminated successfully    (Exit mode 0)
            Current function value: -9.578265310119352
            Iterations: 15
            Function evaluations: 13102
            Gradient evaluations: 15

Optimization complete!
Optimization converged successfully.

Recovered parameters:
Weights (w): [-1.55857592 -1.64663323 -1.48675863  0.12554296]
Bias (b): -0.36010701786217103
Number of support vectors: 21

Results for C = 0.1145475372279496:
Weights: [-1.55857592 -1.64663323 -1.48675863  0.12554296]
Bias: -0.36010701786217103
Dual Objective Value: -9.578265310119352

==================================================


--- Running dual SVM for C = 0.572737686139748 (Iteration 2/3) ---

Starting dual SVM optimization...
Number of samples: 872
Hyperparameter C: 0.572737686139748
Optimization Progress: 100.00% completeOptimization terminated successfully    (Exit mode 0)
            Current function value: -27.28257250580055
            Iterations: 24
            Function evaluations: 20960
            Gradient evaluations: 24

Optimization complete!
Optimization converged successfully.

Recovered parameters:
Weights (w): [-2.03928401 -2.81776993 -2.35019259 -0.01226374]
Bias (b): -0.7953417260320704
Number of support vectors: 13

Results for C = 0.572737686139748:
Weights: [-2.03928401 -2.81776993 -2.35019259 -0.01226374]
Bias: -0.7953417260320704
Dual Objective Value: -27.28257250580055

==================================================


--- Running dual SVM for C = 0.8018327605956472 (Iteration 3/3) ---

Starting dual SVM optimization...
Number of samples: 872
Hyperparameter C: 0.8018327605956472
Optimization Progress: 100.00% completeOptimization terminated successfully    (Exit mode 0)
            Current function value: -34.508693054450426
            Iterations: 21
            Function evaluations: 18341
            Gradient evaluations: 21

Optimization complete!
Optimization converged successfully.

Recovered parameters:
Weights (w): [-2.15058543 -2.9287278  -2.46085431 -0.01009014]
Bias (b): -1.125189294688366
Number of support vectors: 58

Results for C = 0.8018327605956472:
Weights: [-2.15058543 -2.9287278  -2.46085431 -0.01009014]
Bias: -1.125189294688366
Dual Objective Value: -34.508693054450426

==================================================

'''