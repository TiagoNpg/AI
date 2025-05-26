import numpy as np
from collections import Counter

import csv

class DecisionTree:
    def __init__(self, X, y, threshold=1.0, max_depth=None): # Additional optional arguments can be added, but the default value needs to be provided
        self.threshold = threshold
        self.max_depth = max_depth
        self.tree = self.__build_tree(X, y)
    
    def __entropy(self, y):
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def __information_gain(self,X, y, attr_index):
        total_entropy = self.__entropy(y)
        values = set(x[attr_index] for x in X)
        split_entropy = 0.0
        for value in values:
            subset_y = [y[i] for i in range(len(X)) if X[i][attr_index] == value]
            proportion = len(subset_y) / len(y)
            split_entropy += proportion * self.__entropy(subset_y)

        return total_entropy - split_entropy
    
    def __majority_class(self, y):
        return Counter(y).most_common(1)[0][0]
    
    def __build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return y[0]
        if not X or (self.max_depth is not None and depth >= self.max_depth):
            return self.__majority_class(y)
        best_ig = -1
        best_attr = None
        for attr in range(len(X[0])):
            ig = self.__information_gain(X, y, attr)
            if ig > best_ig:
                best_ig = ig
                best_attr = attr
        if best_ig < 1e-6:
            return self.__majority_class(y)
        tree = {'attr': best_attr, 'branches': {}}
        values = set(x[best_attr] for x in X)
        for value in values:
            subset_X = [x for i, x in enumerate(X) if x[best_attr] == value]
            subset_y = [y[i] for i, x in enumerate(X) if x[best_attr] == value]
            tree['branches'][value] = self.__build_tree(subset_X, subset_y, depth + 1)
        return tree
        
        
    def predict(self, x):
        node = self.tree
        while isinstance(node, dict):
            attr = node['attr']
            value = x[attr]
            if value in node['branches']:
                node = node['branches'][value]
            else:
                # Valor desconhecido, retorna a classe majoritária
                return 1
        return node


def train_decision_tree(X, y):
    return DecisionTree(X, y)    

def test_last_3_items():
    X = []
    Xi = []
    y = []
    yi = []

    with open('items.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for i, row in enumerate(reader):
            if i >= 12:  # Apenas usa as primeiras 12 linhas para treinar, as últimas 3 serão usadas para testar
                Xi.append([row['name'], row['color'], row['format']])
                yi.append(int(row['is_fruit']))
            else:
                X.append([row['name'], row['color'], row['format']])
                y.append(int(row['is_fruit']))

    # Instância e treina a árvore de decisão
    dt = train_decision_tree(X, y)

    # Teste nos últimos 3 itens
    print("\nTesting on the last 3 items from items.csv:")
    for item, isFruit in zip(Xi, yi):
        pred = dt.predict(item)
        print(f"Input: {item} | True: {isFruit} | Predicted: {pred}")

if __name__ == "__main__":
    test_last_3_items()
