import numpy as np
import pandas as pd
from Main_HW3 import load_data


# Average Perceptron Training Function
def average_perceptron_train(train_data, T=10, learning_rate=1.0):
    # Initialize weights, bias, and the average weights
    weights = np.zeros(train_data.shape[1] - 1)
    bias = 0
    avg_weights = np.zeros(train_data.shape[1] - 1)  # For storing averaged weights
    avg_bias = 0  # For storing averaged bias

    # Convert labels to {-1, 1} for Perceptron
    train_labels = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

    for epoch in range(T):
        for index, row in train_data.iterrows():
            x = row[:-1].values
            y = train_labels.iloc[index]
            # Check for misclassification
            if y * (np.dot(weights, x) + bias) <= 0:
                # Update weights and bias if there is a misclassification
                weights += learning_rate * y * x
                bias += learning_rate * y

            # Accumulate weights and bias for averaging
            avg_weights += weights
            avg_bias += bias

    # Return the averaged weights and bias
    avg_weights /= (T * len(train_data))
    avg_bias /= (T * len(train_data))

    return avg_weights, avg_bias


# Prediction function using average weights
def average_perceptron_predict(test_data, avg_weights, avg_bias):
    predictions = []
    for _, row in test_data.iterrows():
        x = row[:-1].values
        prediction = 1 if np.dot(avg_weights, x) + avg_bias > 0 else -1
        predictions.append(prediction)
    return predictions


# Calculate average prediction error
def calculate_error(predictions, true_labels):
    true_labels = true_labels.apply(lambda x: 1 if x == 1 else -1)
    errors = sum(predictions != true_labels)
    return errors / len(true_labels)


# Load the data
train_data, test_data, attributes = load_data()

# Train the Average Perceptron
avg_weights, avg_bias = average_perceptron_train(train_data)

# Make predictions on the test data
test_labels = test_data['genuineorforged']
predictions = average_perceptron_predict(test_data, avg_weights, avg_bias)

# Calculate average test error
error = calculate_error(np.array(predictions), test_labels)

# Display the learned average weight vector and test error
print("Learned average weight vector:", avg_weights)
print("Average bias:", avg_bias)
print("Average test error on the test dataset:", error)