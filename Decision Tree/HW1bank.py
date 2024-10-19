# Import necessary modules
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def load_data():
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    # print(train_data.head())
    attributes = ['age', 'job', 'martial', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
                  'duration', 'campign', 'pdays', 'previous', 'poutcome', 'label']
    train_data.columns = attributes
    test_data.columns = attributes
    #print(train_data.dtypes)
    #Int to binary
    int_columns = [col for col in train_data.columns if train_data[col].dtype == 'int64' and 'int66']
    def numerical_to_binary(data, int_columns): # using median
        for column in int_columns:
            median = data[column].median()
            data[column] = np.where(data[column] > median, 1, 0)

        return data
    #print(int_columns)
    train_data = numerical_to_binary(train_data, int_columns)
    test_data = numerical_to_binary(test_data, int_columns)
    #print(train_data.head())
    #Replace Unknown
    columns_with_unknown = [col for col in train_data.columns if 'unknown' in train_data[col].values]
    def repalce_unknown(data,col_with_unknown):
        for col in columns_with_unknown:
            # Find the most frequent value (excluding 'unknown')
            majority_value = data[col][data[col] != 'unknown'].mode()[0]

            # Replace 'unknown' with the majority value
            data[col] = data[col].replace('unknown', majority_value)

        return data
    train_data= repalce_unknown(train_data, columns_with_unknown)
    test_data = repalce_unknown(test_data, columns_with_unknown)

    return train_data, test_data,attributes

# Heuristic calculation functions
def entropy(data):
    labels = data['label']
    label_counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * np.log2(count / total) for count in label_counts.values())


def majority_error(data):
    labels = data['label']
    label_counts = Counter(labels)
    majority_class_count = label_counts.most_common(1)[0][1]
    total = len(labels)
    return 1 - (majority_class_count / total)


def gini_index(data):
    labels = data['label']
    label_counts = Counter(labels)
    total = len(labels)
    return 1 - sum((count / total) ** 2 for count in label_counts.values())


# Data splitting function
def split_data(data, attribute):
    unique_values = data[attribute].unique()
    return {value: data[data[attribute] == value] for value in unique_values}


# Decision tree node class
class TreeNode:
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute  # Node attribute
        self.value = value  # Node value
        self.label = label  # Label for leaf node
        self.children = {}  # Children nodes

    def add_child(self, value, node):
        self.children[value] = node


# Tree-building function
def build_tree(data, attributes, heuristic, max_depth, current_depth=0):
    labels = data['label']

    # If all labels are the same, return a leaf node
    if len(set(labels)) == 1:
        return TreeNode(label=labels.iloc[0])

    # If no attributes left or max depth reached, return the majority label
    if not attributes or current_depth == max_depth:
        return TreeNode(label=Counter(labels).most_common(1)[0][0])

    # Choose best attribute and split data
    best_attribute = choose_attribute(data, attributes, heuristic)
    tree = TreeNode(attribute=best_attribute)
    splits = split_data(data, best_attribute)

    # Remaining attributes for the next recursive calls
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]

    # Build tree recursively for each split
    for value, subset in splits.items():
        if subset.empty:
            tree.children[value] = TreeNode(label=Counter(labels).most_common(1)[0][0])
        else:
            tree.children[value] = build_tree(subset, remaining_attributes, heuristic, max_depth, current_depth + 1)

    return tree


# Attribute selection based on heuristic
def choose_attribute(data, attributes, heuristic):
    if heuristic == 'entropy':
        initial_heuristic = entropy(data)
    elif heuristic == 'majority_error':
        initial_heuristic = majority_error(data)
    elif heuristic == 'gini_index':
        initial_heuristic = gini_index(data)

    best_gain = 0
    best_attribute = None

    for attribute in attributes:
        data_split = split_data(data, attribute)
        weighted_avg = sum(
            (len(subset) / len(data)) * (entropy(subset) if heuristic == 'entropy'
                                         else majority_error(subset) if heuristic == 'majority_error'
            else gini_index(subset))
            for subset in data_split.values())

        current_gain = initial_heuristic - weighted_avg

        if current_gain > best_gain:
            best_gain = current_gain
            best_attribute = attribute

    return best_attribute


# Prediction and evaluation functions
def predict(tree, example):
    if tree.label is not None:
        return tree.label
    attribute_value = example[tree.attribute]
    if attribute_value in tree.children:
        return predict(tree.children[attribute_value], example)
    else:
        return None


def evaluate(tree, data):
    predictions = data.apply(lambda row: predict(tree, row), axis=1)
    accuracy = (predictions == data['label']).mean()
    return 1 - accuracy  # Returning error (1 - accuracy)


# Experiment runner
def run_experiment(train_data, test_data, attributes, max_depth):
    heuristics = ['entropy', 'majority_error', 'gini_index']
    results = {}

    for heuristic in heuristics:
        tree = build_tree(train_data, attributes, heuristic, max_depth)
        train_error = evaluate(tree, train_data)
        test_error = evaluate(tree, test_data)
        results[heuristic] = (train_error, test_error)

    return results


# Main function to execute the experiments
def main():
    train_data, test_data, attributes = load_data()

    # Iterate through different tree depths
    for depth in range(1, 7):
        results = run_experiment(train_data, test_data, attributes, depth)
        print(f"Results for max depth: {depth}")
        for heuristic, (train_error, test_error) in results.items():
            print(f"Heuristic: {heuristic}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")


if __name__ == "__main__":
    main()


#print(train_data.head())



#for col in col_with_unknown:
    #correct_rows = data[data[col] != "unknown"]
    #correct_values = correct_rows[col]
    #major_value = Counter(correct_values).most_common(1)[0][1]
    #data[col] = data[col].replace('unknown', major_value)