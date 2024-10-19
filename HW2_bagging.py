import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fully expanded Decision Tree using original ID3
class DecisionTreeID3:
    def __init__(self, criterion='entropy'):
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        data = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
        self.tree = self._build_tree(data)

    def _build_tree(self, data):
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        unique_classes = y.unique()

        if len(unique_classes) == 1:  # Pure node
            return unique_classes[0]

        if len(X.columns) == 0:
            return y.mode()[0]  # Majority class in the node

        best_feature = self._choose_best_feature(X, y)
        tree = {best_feature: {}}
        feature_values = X[best_feature].unique()

        for value in feature_values:
            subset = data[data[best_feature] == value]
            subset = subset.drop(columns=best_feature)
            subtree = self._build_tree(subset)
            tree[best_feature][value] = subtree

        return tree

    def _choose_best_feature(self, X, y):
        max_info_gain = -float('inf')
        best_feature = None

        for feature in X.columns:
            info_gain = self._information_gain(X[feature], y)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature

        return best_feature

    def _information_gain(self, feature, y):
        total_entropy = self._entropy(y)
        weighted_entropy = 0
        for value in feature.unique():
            subset = y[feature == value]
            weighted_entropy += (len(subset) / len(y)) * self._entropy(subset)
        return total_entropy - weighted_entropy

    def _entropy(self, y):
        proportions = y.value_counts(normalize=True)
        return -sum(proportions * np.log2(proportions))

    def predict(self, X):
        return X.apply(self._predict_single, axis=1)

    def _predict_single(self, x):
        node = self.tree
        while isinstance(node, dict):
            feature = next(iter(node))
            if x[feature] in node[feature]:
                node = node[feature][x[feature]]
            else:
                return None
        return node

# Bagged Trees Algorithm
class BaggedTrees:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Train a fully expanded decision tree
            tree = DecisionTreeID3(criterion='entropy')
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

            # Notification for each tree
            print(f"Tree {i + 1}/{self.n_estimators} trained.")

    def predict(self, X):
        # Aggregate predictions using majority voting
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return pd.DataFrame(tree_predictions).mode().iloc[0].values

# Load data function
def load_data():
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    attributes = ['age', 'job', 'martial', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
                  'duration', 'campign', 'pdays', 'previous', 'poutcome', 'label']
    train_data.columns = attributes
    test_data.columns = attributes

    int_columns = [col for col in train_data.columns if train_data[col].dtype == 'int64']

    def numerical_to_binary(data, int_columns):
        for column in int_columns:
            median = data[column].median()
            data[column] = np.where(data[column] > median, 1, 0)

        return data

    train_data = numerical_to_binary(train_data, int_columns)
    test_data = numerical_to_binary(test_data, int_columns)

    return train_data, test_data, attributes

# Load data
train_data, test_data, attributes = load_data()
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Bagged Trees with varying number of estimators
max_estimators = 10
train_errors = []
test_errors = []

for n_trees in range(1, max_estimators + 1):
    model = BaggedTrees(n_estimators=n_trees)
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate errors
    train_error = np.mean(y_train != y_train_pred)
    test_error = np.mean(y_test != y_test_pred)

    train_errors.append(train_error)
    test_errors.append(test_error)

    # Notification for each ensemble
    print(f"Ensemble size: {n_trees} | Training Error: {train_error:.4f} | Test Error: {test_error:.4f}")

# Plotting training and test errors vs. number of trees
plt.figure(figsize=(12, 6))
plt.plot(range(1, max_estimators + 1), train_errors, label='Training Error')
plt.plot(range(1, max_estimators + 1), test_errors, label='Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Training and Test Errors vs. Number of Bagged Trees')
plt.legend()
plt.show()