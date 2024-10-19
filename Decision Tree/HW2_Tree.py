import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data Function (unchanged)
def load_data():
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    attributes = ['age', 'job', 'martial', 'education', 'default', 'balance', 'housing',
                  'loan', 'contact', 'day', 'month', 'duration', 'campign', 'pdays',
                  'previous', 'poutcome', 'label']
    train_data.columns = attributes
    test_data.columns = attributes

    # Convert numerical columns to binary
    int_columns = [col for col in train_data.columns if train_data[col].dtype == 'int64']
    def numerical_to_binary(data, int_columns):
        for column in int_columns:
            median = data[column].median()
            data[column] = np.where(data[column] > median, 1, 0)
        return data

    train_data = numerical_to_binary(train_data, int_columns)
    test_data = numerical_to_binary(test_data, int_columns)

    # Replace 'unknown' with majority value
    columns_with_unknown = [col for col in train_data.columns if 'unknown' in train_data[col].values]
    def replace_unknown(data, columns_with_unknown):
        for col in columns_with_unknown:
            majority_value = data[col][data[col] != 'unknown'].mode()[0]
            data[col] = data[col].replace('unknown', majority_value)
        return data

    train_data = replace_unknown(train_data, columns_with_unknown)
    test_data = replace_unknown(test_data, columns_with_unknown)

    return train_data, test_data, attributes

# Define the ID3 Decision Tree Classifier
class ID3DecisionTreeClassifier:
    def __init__(self, criterion='entropy', max_depth=None):
        self.criterion = criterion  # 'gini', 'entropy', 'majority_error'
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        self.tree = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth):
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        unique_classes = y.unique()

        if len(unique_classes) == 1:  # Pure node
            return unique_classes[0]
        if len(X.columns) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return y.mode()[0]  # Majority class in the node

        best_feature = self._choose_best_feature(X, y)
        tree = {best_feature: {}}
        feature_values = X[best_feature].unique()

        for value in feature_values:
            subset = data[data[best_feature] == value]
            subset = subset.drop(columns=best_feature)
            subtree = self._build_tree(subset, depth + 1)
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
        if self.criterion == 'entropy':
            return self._entropy_gain(feature, y)
        elif self.criterion == 'gini':
            return self._gini_gain(feature, y)
        elif self.criterion == 'majority_error':
            return self._majority_error_gain(feature, y)
        else:
            raise ValueError("Invalid criterion: Use 'entropy', 'gini', or 'majority_error'")

    def _entropy_gain(self, feature, y):
        total_entropy = self._entropy(y)
        weighted_entropy = 0
        for value in feature.unique():
            subset = y[feature == value]
            weighted_entropy += (len(subset) / len(y)) * self._entropy(subset)
        return total_entropy - weighted_entropy

    def _gini_gain(self, feature, y):
        total_gini = self._gini(y)
        weighted_gini = 0
        for value in feature.unique():
            subset = y[feature == value]
            weighted_gini += (len(subset) / len(y)) * self._gini(subset)
        return total_gini - weighted_gini

    def _majority_error_gain(self, feature, y):
        total_majority_error = self._majority_error(y)
        weighted_majority_error = 0
        for value in feature.unique():
            subset = y[feature == value]
            weighted_majority_error += (len(subset) / len(y)) * self._majority_error(subset)
        return total_majority_error - weighted_majority_error

    def _entropy(self, y):
        proportions = y.value_counts(normalize=True)
        return -sum(proportions * np.log2(proportions))

    def _gini(self, y):
        proportions = y.value_counts(normalize=True)
        return 1 - sum(proportions ** 2)

    def _majority_error(self, y):
        proportions = y.value_counts(normalize=True)
        return 1 - proportions.max()

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

# Custom function for accuracy calculation
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Custom function for train-test split
def train_test_split(data, test_size=0.2):
    test_data = data.sample(frac=test_size, random_state=42)
    train_data = data.drop(test_data.index)
    return train_data, test_data

# Load the data
train_data, test_data, attributes = load_data()

# Split train and test data
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Train the ID3 Decision Tree Model
criterion = 'entropy'  # Choose between 'entropy', 'gini', or 'majority_error'
model = ID3DecisionTreeClassifier(criterion=criterion, max_depth=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
model_accuracy = accuracy(y_test, y_pred)
print(f"Accuracy: {model_accuracy * 100:.2f}%")

# Print the tree structure
print("Trained ID3 Decision Tree:")
print(model.tree)
