import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Data Function with One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder


def load_data():
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    attributes = ['age', 'job', 'martial', 'education', 'default', 'balance', 'housing',
                  'loan', 'contact', 'day', 'month', 'duration', 'campign', 'pdays',
                  'previous', 'poutcome', 'label']

    train_data.columns = attributes
    test_data.columns = attributes

    # Convert labels 'yes'/'no' to {-1, 1}
    train_data['label'] = train_data['label'].map({'no': -1, 'yes': 1})
    test_data['label'] = test_data['label'].map({'no': -1, 'yes': 1})

    # One-hot encode categorical features
    categorical_columns = train_data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')

    train_encoded = encoder.fit_transform(train_data[categorical_columns])
    train_encoded = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    test_encoded = encoder.transform(test_data[categorical_columns])
    test_encoded = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop original categorical columns and concatenate with encoded columns
    train_data = train_data.drop(columns=categorical_columns).reset_index(drop=True)
    test_data = test_data.drop(columns=categorical_columns).reset_index(drop=True)

    train_data = pd.concat([train_data, train_encoded], axis=1)
    test_data = pd.concat([test_data, test_encoded], axis=1)

    return train_data, test_data


# Random Tree Classifier
class RandomTree:
    def __init__(self, max_features):
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1 or X.shape[1] == 0:
            return np.sign(np.mean(y))

        # Randomly select a subset of features
        features = np.random.choice(X.columns, self.max_features, replace=False)

        # Find the best feature to split
        best_feature, best_threshold = self._best_split(X[features], y)

        if best_feature is None:
            return np.sign(np.mean(y))

        tree = {best_feature: {}}
        left_mask = X[best_feature].astype(float) <= best_threshold
        right_mask = X[best_feature].astype(float) > best_threshold

        # Recursively build the left and right branches
        tree[best_feature]['threshold'] = best_threshold
        tree[best_feature]['left'] = self._build_tree(X[left_mask], y[left_mask])
        tree[best_feature]['right'] = self._build_tree(X[right_mask], y[right_mask])

        return tree

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in X.columns:
            # Ensure the feature values are numeric
            feature_values = pd.to_numeric(X[feature], errors='coerce').dropna().unique()

            for threshold in feature_values:
                gain = self._information_gain(X[feature].astype(float), y, float(threshold))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _information_gain(self, feature, y, threshold):
        left_mask = feature <= threshold
        right_mask = feature > threshold

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0

        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])

        n = len(y)
        n_left = len(y[left_mask])
        n_right = len(y[right_mask])

        weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - weighted_entropy

    def _entropy(self, y):
        proportions = np.bincount(y + 1) / len(y)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))

    def predict(self, X):
        # Replace NaNs in features with column means (handling missing values)
        X_filled = X.fillna(X.mean())
        predictions = X_filled.apply(self._predict_row, axis=1)

        # Handle remaining NaNs by replacing them with the majority class
        majority_class = np.sign(np.mean(predictions.dropna()))
        predictions.fillna(majority_class, inplace=True)

        return predictions

    def _predict_row(self, x):
        node = self.tree
        while isinstance(node, dict):
            feature = list(node.keys())[0]
            threshold = node[feature]['threshold']

            # Convert feature value to float for comparison
            feature_value = pd.to_numeric(x[feature], errors='coerce')

            # If feature value is NaN, default to majority class at this node
            if pd.isna(feature_value):
                return np.sign(np.mean([node[feature]['left'], node[feature]['right']]))

            # Ensure the threshold is numerical for comparison
            if feature_value <= threshold:
                node = node[feature]['left']
            else:
                node = node[feature]['right']

        return node
# Random Forest Classifier
class RandomForest:
    def __init__(self, n_trees=15, max_features=2):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = resample(X, y)
            tree = RandomTree(max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.sign(np.sum(tree_preds, axis=0))


# Experiment with Random Forests
def run_random_forest_experiment(X_train, y_train, X_test, y_test, n_trees_list, feature_subset_sizes):
    results = {}

    for max_features in feature_subset_sizes:
        train_errors = []
        test_errors = []

        for n_trees in n_trees_list:
            rf = RandomForest(n_trees=n_trees, max_features=max_features)
            rf.fit(X_train, y_train)

            y_train_pred = rf.predict(X_train)
            y_test_pred = rf.predict(X_test)

            train_error = 1 - accuracy_score(y_train, y_train_pred)
            test_error = 1 - accuracy_score(y_test, y_test_pred)

            train_errors.append(train_error)
            test_errors.append(test_error)

        results[max_features] = (train_errors, test_errors)

    return results


# Load the data
train_data, test_data = load_data()
X_train, y_train = train_data.iloc[:, :-1], train_data['label']
X_test, y_test = test_data.iloc[:, :-1], test_data['label']

# Parameters for the experiment
n_trees_list = range(1, 501, 10)
feature_subset_sizes = [2, 4, 6]

# Run the experiment
results = run_random_forest_experiment(X_train, y_train, X_test, y_test, n_trees_list, feature_subset_sizes)

# Plot the results
plt.figure(figsize=(12, 8))
for max_features, (train_errors, test_errors) in results.items():
    plt.plot(n_trees_list, train_errors, label=f'Train Error (max_features={max_features})')
    plt.plot(n_trees_list, test_errors, linestyle='--', label=f'Test Error (max_features={max_features})')

plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Number of Trees (Random Forest)')
plt.legend()
plt.show()