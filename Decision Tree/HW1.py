#Import Needed modules.
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt



#Load train and test data for Car
train_data = pd.read_csv('./data/car/train.csv', header=None)
test_data = pd.read_csv('./data/car/test.csv', header=None)
#Test and review data
#print(train_data.head())
#(train_data.head())
# Name each attribute from its variable table
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
train_data.columns = attributes
test_data.columns = attributes
#print(train_data.head())
#print(test_data.head())


# Calculate entropy for information gain
def entropy(data):
    labels = data['label']
    label_counts = Counter(labels)
    total = len(labels)
    entropy_value = -sum((count / total) * np.log2(count / total) for count in label_counts.values())

    return entropy_value


#Test if it works.
#labels = train_data['label']
#label_counts = Counter(labels)
#total = len(labels)
#entropy_value = -sum((count / total) * np.log2(count / total) for count in label_counts.values())

#print(entropy_value)
#print(label_counts)
#print(total) #1.21

# Calculate majority error
def majority_error(data):
    labels = data['label']
    label_counts = Counter(labels)
    majority_Class = label_counts.most_common(1)[0][1]
    total = len(labels)
    majority_error = 1 - (majority_Class / total)

    return majority_error
#labels = train_data['label']
#label_counts = Counter(labels)
#majority_Class = label_counts.most_common(1)[0][1]
#total = len(labels)
#majority_error = 1 - (majority_Class / total)
#print(majority_error) #0.3002

# Calculate Gini index
def gini_index(data):
    labels = data['label']
    label_counts = Counter(labels)
    total = len(labels)
    gini_value = 1 - sum((count / total) ** 2 for count in label_counts.values())

    return gini_value
# End of 3 base measurements for information gain.

#Split data by attribute
def split_data(data,attribute):
    unique_values =data[attribute].unique()
    splits = {value : data[data[attribute] == value] for value in unique_values}

    return splits

#Function to choose attribute
def attribute_choose(data, attributes, heuristic):

    # Entropy, ME, Gini
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
        weighted_average = sum((len(subset) / len(data)) * (entropy(subset) if heuristic == 'entropy'
                                                            else majority_error(subset) if heuristic == 'majority_error'
        else gini_index(subset))
                               for subset in data_split.values())
        current_gain = initial_heuristic - weighted_average

        if current_gain > best_gain:
            best_gain = current_gain
            best_attribute = attribute
    return best_attribute
# end of attribute selection

#ID3 tree
class TreeNode:
    def __init__(self, attribute=None, value=None, label=None):
        # Attribute used for splitting the node (internal nodes)
        self.attribute = attribute

        # Value of the current node (for child nodes, this stores the parent's splitting value)
        self.value = value

        # Label assigned if this is a leaf node (classification result)
        self.label = label

        # Dictionary to hold the children of this node.
        # Keys are values of the attribute, and values are the child nodes
        self.children = {}

    def add_child(self, value, node):
        # Adds a child node to the current node. 'self.children' keeps track of child nodes.
        self.children[value] = node
def build_tree(data, attributes, heuristic, max_depth, current_depth=0):
    labels = data['label']

    if len(set(labels)) == 1:
        return TreeNode(label=labels.iloc[0])
    if not attributes or current_depth == max_depth:
        return TreeNode(label=Counter(labels).most_common(1)[0][0])

    best_attribute = attribute_choose(data, attributes, heuristic)
    tree = TreeNode(attribute=best_attribute)
    splits = split_data(data, best_attribute)
    remaining_attributes = [attribute for attribute in attributes if attribute != best_attribute]
    for value, subset in splits.items():
        if subset.empty:
            tree.children[value] = TreeNode(label=Counter(labels).most_common(1)[0][0])
        else:
            tree.children[value] = build_tree(subset, remaining_attributes, heuristic, max_depth, current_depth + 1)

    return tree
# end of build tree

# predict label:
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
    error = 1 - accuracy

    return error

def run_experiment(max_depth):
    heuristics = ['entropy', 'majority_error', 'gini_index']
    results = {}

    for heuristic in heuristics:
        tree = build_tree(train_data, attributes, heuristic, max_depth)
        train_error = evaluate(tree, train_data)
        test_error = evaluate(tree, test_data)
        results[heuristic] = (train_error, test_error)

    return results

for i in range(1, 7):
    results = run_experiment(i)
    print(f"Results for max depth: {i}")
    for heuristic, (train_error, test_error) in results.items():
        print(f"Heuristic: {heuristic}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")





