import pandas as pd
import numpy as np
import math
class DecisionTreeNode:
    def __init__(self, attribute=None, label=None, branches={}):
        self.attribute = attribute
        self.label = label
        self.branches = branches
def entropy(data):
    target = data['target']
    n = len(target)
    unique, counts = np.unique(target, return_counts=True)
    entropy = 0
    for i in range(len(unique)):
        p = counts[i] / n
        entropy -= p * math.log2(p)
    return entropy
def information_gain(data, attribute):
    n = len(data)
    values = data[attribute].unique()
    entropy_s = entropy(data)
    entropy_attr = 0
    for value in values:
        subset = data[data[attribute] == value]
        subset_n = len(subset)
        subset_entropy = entropy(subset)
        entropy_attr += subset_n / n * subset_entropy
    return entropy_s - entropy_attr
def id3(data, attributes):
    target = data['target']
    if len(target.unique()) == 1:
        return DecisionTreeNode(label=target.iloc[0])
    if len(attributes) == 0:
        return DecisionTreeNode(label=target.value_counts().idxmax())
    gains = {attr: information_gain(data, attr) for attr in attributes}
    best_attribute = max(gains, key=gains.get)
    node = DecisionTreeNode(attribute=best_attribute)
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value].drop(best_attribute, axis=1)
        if len(subset) == 0:
            node.branches[value] = DecisionTreeNode(label=target.value_counts().idxmax())
        else:
            new_attributes = attributes.copy()
            new_attributes.remove(best_attribute)
            node.branches[value] = id3(subset, new_attributes)
    return node
data = pd.read_csv(r"D:\Machine_Learinng\id3.csv")
attributes = data.columns[:-1].tolist()
root = id3(data, attributes)
def classify(sample, tree):
    if tree.label is not None:
        return tree.label
    attribute = tree.attribute
    value = sample[attribute]
    if value not in tree.branches:
        return tree.branches[max(tree.branches.keys(), key=int)]
    subtree = tree
