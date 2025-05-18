import numpy as np
from collections import Counter

class DecisionTree:

    def __init__(self, X, y, threshold=1.0, max_depth=None): # Additional optional arguments can be added, but the default value needs to be provided
        
        '''
        X: list of features
        y: column feature target
        
        '''
        
        self.X = X
        self.y = y
        self.threshold = threshold
        self.max_depth = max_depth
        self.dataset = np.genfromtxt('train.csv', delimiter=';', dtype=None, encoding=None)
        
    
    def __entropy(self, y):
        """Calcular a entropia"""
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def __information_gain(self,X, y, attr_index):
        '''Calcular IG'''
        total_entropy = self._entropy(y)
        values = set(x[attr_index] for x in X)

        split_entropy = 0.0
        for value in values:
            subset_y = [y[i] for i in range(len(X)) if X[i][attr_index] == value]
            proportion = len(subset_y) / len(y)
            split_entropy += proportion * self._entropy(subset_y)

        return total_entropy - split_entropy
        
        
    def predict(self, x): # (e.g. x = ['apple', 'green', 'circle'] -> 1 or -1)
        # Implement this
        pass


def train_decision_tree(X, y):
    # Replace with your configuration
    return DecisionTree(X, y)
    


def test():
    
    #N funciona pq a dataset contem strings e ints
    #dataset = np.loadtxt('items.csv', delimiter=';')
    #print(dataset.shape)
    
    dataset = np.genfromtxt('items.csv', delimiter=';', dtype=None, encoding=None)
    print(dataset)
    print (np.size(dataset[1:,0]))
    print(dataset[0,1:-1])
    
if __name__ == "__main__":
    test()
