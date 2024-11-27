import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data Function (unchanged)
def load_data():
    train_data = pd.read_csv('./BankNote/train.csv', header=None)
    test_data = pd.read_csv('./BankNote/test.csv', header=None)

    attributes = ['variance', 'skewness', 'curtosis', 'entropy',  'genuineorforged']
    train_data.columns = attributes
    test_data.columns = attributes

    return train_data, test_data, attributes


# Perceptron Function
def perceptron_train(train_data, T=10, learning_rate=1.0):
    # Initialize weights and bias
    weights = np.zeros(train_data.shape[1] - 1)
    bias = 0

    # Convert labels to {-1, 1} for Perceptron
    train_labels = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

    for epoch in range(T):
        for index, row in train_data.iterrows():
            x = row[:-1].values
            y = train_labels.iloc[index]
            # Perceptron update rule
            if y * (np.dot(weights, x) + bias) <= 0:
                weights += learning_rate * y * x
                bias += learning_rate * y

    return weights, bias


# Prediction function
def perceptron_predict(data, weights, bias):
    predictions = []
    for _, row in data.iterrows():
        x = row[:-1].values
        prediction = 1 if np.dot(weights, x) + bias > 0 else -1
        predictions.append(prediction)
    return predictions


# Calculate average prediction error
def calculate_error(predictions, true_labels):
    true_labels = true_labels.apply(lambda x: 1 if x == 1 else -1)
    errors = sum(predictions != true_labels)
    return errors / len(true_labels)


# Load the data
train_data, test_data, attributes = load_data()

# Train the Perceptron
weights, bias = perceptron_train(train_data)

# Make predictions on the test data
test_labels = test_data['genuineorforged']
predictions = perceptron_predict(test_data, weights, bias)

# Calculate average prediction error
error = calculate_error(np.array(predictions), test_labels)

print("Learned weight vector:", weights)
print("Bias:", bias)
print("Average prediction error on the test dataset:", error)
#Learned weight vector: [-61.086591 -42.70582  -40.30786   -3.146269]
#Bias: 53.0
#Average prediction error on the test dataset: 0.02