
def load_data():
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    attributes = ['age', 'job', 'martial', 'education', 'default', 'balance', 'housing',
                  'loan', 'contact', 'day', 'month', 'duration', 'campign', 'pdays',
                  'previous', 'poutcome', 'label']
    train_data.columns = attributes
    test_data.columns = attributes

    # Convert integer columns to binary
    int_columns = [col for col in train_data.columns if train_data[col].dtype == 'int64']

    def numerical_to_binary(data, int_columns):
        for column in int_columns:
            median = data[column].median()
            data[column] = np.where(data[column] > median, 1, 0)
        return data

    train_data = numerical_to_binary(train_data, int_columns)
    test_data = numerical_to_binary(test_data, int_columns)

    return train_data, test_data, attributes

# Step 2: Load data and preprocess it
train_data, test_data, attributes = load_data()
import numpy as np
from collections import Counter

# Function to train a fully expanded decision tree
def fully_expanded_decision_tree(X, y):
    """
    Build a fully expanded decision tree based on information gain.
    """
    n_samples, n_features = X.shape
    best_info_gain = -np.inf
    best_feature = None
    best_threshold = None
    best_left_indices = None
    best_right_indices = None

    # Iterate over each feature
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        # Try splitting at each threshold
        for threshold in thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold

            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue

            y_left, y_right = y[left_indices], y[right_indices]
            info_gain = calculate_information_gain(y, y_left, y_right)

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
                best_threshold = threshold
                best_left_indices = left_indices
                best_right_indices = right_indices

    # Create a leaf node if no further split is possible
    if best_info_gain == -np.inf:
        return Counter(y).most_common(1)[0][0]

    # Build left and right branches
    left_tree = fully_expanded_decision_tree(X[best_left_indices], y[best_left_indices])
    right_tree = fully_expanded_decision_tree(X[best_right_indices], y[best_right_indices])

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_tree,
        "right": right_tree
    }

def tree_predict(tree, x):
    """
    Predict the class of a sample using a fully expanded decision tree.
    """
    if isinstance(tree, dict):
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return tree_predict(tree["left"], x)
        else:
            return tree_predict(tree["right"], x)
    else:
        return tree

def bootstrap_sample(X, y, n_samples):
    """
    Generate a bootstrap sample from the dataset.
    """
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    return X[indices], y[indices]

def compute_bias_variance(y_true, y_preds):
    """
    Compute bias and variance for predictions.
    """
    avg_pred = np.mean(y_preds)
    bias = (avg_pred - y_true) ** 2
    variance = np.mean((y_preds - avg_pred) ** 2)
    return bias, variance

def bias_variance_experiment(X_train, y_train, X_test, y_test, num_trees=500, num_repeats=100, sample_size=1000):
    """
    Bias-variance experiment to compare single decision trees and bagged trees.
    """
    n_test = len(y_test)
    single_tree_bias = np.zeros(n_test)
    single_tree_variance = np.zeros(n_test)
    bagged_tree_bias = np.zeros(n_test)
    bagged_tree_variance = np.zeros(n_test)

    # Repeat experiment 100 times
    for _ in range(num_repeats):
        # STEP 1: Sample 1,000 examples from the training dataset
        X_sample, y_sample = bootstrap_sample(X_train, y_train, sample_size)

        # Train 500 bagged decision trees
        single_tree_preds = []
        bagged_preds = []

        trees = []

        for t in range(num_trees):
            # Train a single decision tree
            X_bootstrap, y_bootstrap = bootstrap_sample(X_sample, y_sample, sample_size)
            tree = fully_expanded_decision_tree(X_bootstrap, y_bootstrap)
            trees.append(tree)

            # Make predictions on test data
            if t == 0:  # First tree (single decision tree)
                single_tree_preds.append([tree_predict(tree, x) for x in X_test])

            # Bagged trees predictions
            bagged_preds.append([tree_predict(tree, x) for x in X_test])

        # Convert lists to numpy arrays for computation
        single_tree_preds = np.array(single_tree_preds[0])
        bagged_preds = np.mean(bagged_preds, axis=0)

        # Compute bias and variance for single tree
        for i in range(n_test):
            bias, variance = compute_bias_variance(y_test[i], single_tree_preds)
            single_tree_bias[i] += bias
            single_tree_variance[i] += variance

        # Compute bias and variance for bagged trees
        for i in range(n_test):
            bias, variance = compute_bias_variance(y_test[i], bagged_preds)
            bagged_tree_bias[i] += bias
            bagged_tree_variance[i] += variance

    # Average over the repeats
    single_tree_bias /= num_repeats
    single_tree_variance /= num_repeats
    bagged_tree_bias /= num_repeats
    bagged_tree_variance /= num_repeats

    # Calculate general squared error for both models
    single_tree_error = single_tree_bias + single_tree_variance
    bagged_tree_error = bagged_tree_bias + bagged_tree_variance

    # Compute average results across all test samples
    avg_single_tree_bias = np.mean(single_tree_bias)
    avg_single_tree_variance = np.mean(single_tree_variance)
    avg_single_tree_error = np.mean(single_tree_error)

    avg_bagged_tree_bias = np.mean(bagged_tree_bias)
    avg_bagged_tree_variance = np.mean(bagged_tree_variance)
    avg_bagged_tree_error = np.mean(bagged_tree_error)

    return (avg_single_tree_bias, avg_single_tree_variance, avg_single_tree_error,
            avg_bagged_tree_bias, avg_bagged_tree_variance, avg_bagged_tree_error)

# Run the bias-variance experiment
num_trees = 500
num_repeats = 100
sample_size = 1000

(avg_single_tree_bias, avg_single_tree_variance, avg_single_tree_error,
 avg_bagged_tree_bias, avg_bagged_tree_variance, avg_bagged_tree_error) = bias_variance_experiment(
    X_train, y_train, X_test, y_test, num_trees=num_trees, num_repeats=num_repeats, sample_size=sample_size)

# Print the results
print("Single Tree Bias:", avg_single_tree_bias)
print("Single Tree Variance:", avg_single_tree_variance)
print("Single Tree Error:", avg_single_tree_error)

print("Bagged Tree Bias:", avg_bagged_tree_bias)
print("Bagged Tree Variance:", avg_bagged_tree_variance)
print("Bagged Tree Error:", avg_bagged_tree_error)
