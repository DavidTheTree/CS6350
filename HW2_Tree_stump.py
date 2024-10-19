import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data Function
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

# Decision Stump Classifier
class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        # Iterate through each feature
        for feature in range(n_features):
            feature_values = X.iloc[:, feature]
            unique_values = feature_values.unique()

            # Test all unique values as potential thresholds
            for threshold in unique_values:
                for polarity in [1, -1]:
                    # Make predictions
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values <= threshold] = -1
                    else:
                        predictions[feature_values > threshold] = -1

                    # Calculate error with sample weights
                    misclassified = sample_weights[predictions != y]
                    error = np.sum(misclassified)

                    if error < min_error:
                        min_error = error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature = feature

    def predict(self, X):
        n_samples = X.shape[0]
        X_feature = X.iloc[:, self.feature]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_feature <= self.threshold] = -1
        else:
            predictions[X_feature > self.threshold] = -1
        return predictions

# AdaBoost Implementation
class AdaBoost:
    def __init__(self, n_estimators=500):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []
        self.train_errors = []
        self.stump_errors = []  # Initialize stump errors

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, sample_weights)
            predictions = stump.predict(X)

            error = np.sum(sample_weights[predictions != y]) / np.sum(sample_weights)
            error = max(error, 1e-10)  # Avoid divide-by-zero error

            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            self.stumps.append(stump)

            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            y_train_pred = self.predict(X)
            train_error = np.mean(y != y_train_pred)
            self.train_errors.append(train_error)

            # Track individual stump errors
            self.stump_errors.append(error)

    def predict(self, X):
        stump_predictions = np.array([stump.predict(X) for stump in self.stumps])
        final_predictions = np.sign(np.dot(self.alphas, stump_predictions))
        return final_predictions
# Convert labels to {-1, 1}
def convert_labels(y):
    return y.map({'no': -1, 'yes': 1})

# Load the data
train_data, test_data, attributes = load_data()
X_train, y_train = train_data.iloc[:, :-1], convert_labels(train_data.iloc[:, -1])
X_test, y_test = test_data.iloc[:, :-1], convert_labels(test_data.iloc[:, -1])

# Train AdaBoost with decision stumps
T = 500
adaboost = AdaBoost(n_estimators=T)
adaboost.fit(X_train, y_train)

# Make predictions and calculate test errors
y_train_pred = adaboost.predict(X_train)
y_test_pred = adaboost.predict(X_test)

train_error = np.mean(y_train != y_train_pred)
test_error = np.mean(y_test != y_test_pred)

# Plotting training and test errors over iterations
#plt.figure(figsize=(12, 6))
#plt.plot(range(1, T + 1), adaboost.train_errors, label='Training Error')
#plt.xlabel('Number of Iterations (T)')
#plt.ylabel('Error')
#plt.title('Training Error vs. Iterations (AdaBoost with Decision Stumps)')
#plt.legend()
#plt.show()

#print(f"Final Training Error: {train_error * 100:.2f}%")
#print(f"Final Test Error: {test_error * 100:.2f}%")
plt.figure(figsize=(12, 6))
plt.plot(range(1, T + 1), adaboost.train_errors, label='Training Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.title('Training Error vs. Iterations (AdaBoost with Decision Stumps)')
plt.legend()
plt.show()

# Plotting training and test errors of individual stumps
plt.figure(figsize=(12, 6))
plt.plot(range(1, T + 1), adaboost.stump_errors, label='Individual Stump Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.title('Stump Errors over Iterations (AdaBoost with Decision Stumps)')
plt.legend()
plt.show()


