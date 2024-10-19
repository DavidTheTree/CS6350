# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load Data Function (same as before)
def load_data():
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    attributes = ['age', 'job', 'martial', 'education', 'default', 'balance', 'housing', 'loan',
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
                  'poutcome', 'label']
    train_data.columns = attributes
    test_data.columns = attributes

    # Int to binary conversion
    int_columns = [col for col in train_data.columns if train_data[col].dtype == 'int64']
    def numerical_to_binary(data, int_columns):
        for column in int_columns:
            median = data[column].median()
            data[column] = np.where(data[column] > median, 1, 0)
        return data

    train_data = numerical_to_binary(train_data, int_columns)
    test_data = numerical_to_binary(test_data, int_columns)

    # Replace 'unknown' values
    columns_with_unknown = [col for col in train_data.columns if 'unknown' in train_data[col].values]
    def replace_unknown(data, columns_with_unknown):
        for col in columns_with_unknown:
            majority_value = data[col][data[col] != 'unknown'].mode()[0]
            data[col] = data[col].replace('unknown', majority_value)
        return data

    train_data = replace_unknown(train_data, columns_with_unknown)
    test_data = replace_unknown(test_data, columns_with_unknown)

    return train_data, test_data, attributes

# Load the data
train_data, test_data, attributes = load_data()

# Split data into features and labels
X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# Encode categorical features using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Store errors
bagging_train_errors = []
bagging_test_errors = []
random_forest_train_errors = []
random_forest_test_errors = []
adaboost_train_errors = []
adaboost_test_errors = []

# Function to compute errors for BaggingClassifier and RandomForestClassifier
def compute_model_errors(model, X_train, y_train, X_test, y_test, train_errors, test_errors, max_estimators=500):
    for n_estimators in range(1, max_estimators + 1):
        model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_errors.append(1 - accuracy_score(y_train, y_train_pred))
        test_errors.append(1 - accuracy_score(y_test, y_test_pred))

# Compute errors for Bagged Trees
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=42)
compute_model_errors(bagging_model, X_train_encoded, y_train, X_test_encoded, y_test, bagging_train_errors, bagging_test_errors)

# Compute errors for Random Forest
random_forest_model = RandomForestClassifier(max_depth=None, random_state=42)
compute_model_errors(random_forest_model, X_train_encoded, y_train, X_test_encoded, y_test, random_forest_train_errors, random_forest_test_errors)

# Compute errors for AdaBoost
adaboost_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=42)
for n_estimators in range(1, 501):
    adaboost_model.set_params(n_estimators=n_estimators)
    adaboost_model.fit(X_train_encoded, y_train)
    y_train_pred = adaboost_model.predict(X_train_encoded)
    y_test_pred = adaboost_model.predict(X_test_encoded)
    adaboost_train_errors.append(1 - accuracy_score(y_train, y_train_pred))
    adaboost_test_errors.append(1 - accuracy_score(y_test, y_test_pred))

# Plotting errors
plt.figure(figsize=(12, 8))
plt.plot(bagging_train_errors, label='Bagged Trees - Train Error')
plt.plot(bagging_test_errors, label='Bagged Trees - Test Error')
plt.plot(random_forest_train_errors, label='Random Forest - Train Error')
plt.plot(random_forest_test_errors, label='Random Forest - Test Error')
plt.plot(adaboost_train_errors, label='AdaBoost - Train Error')
plt.plot(adaboost_test_errors, label='AdaBoost - Test Error')

plt.xlabel('Number of Estimators')
plt.ylabel('Error')
plt.title('Training and Test Errors for Bagged Trees, Random Forest, and AdaBoost')
plt.legend()
plt.show()
