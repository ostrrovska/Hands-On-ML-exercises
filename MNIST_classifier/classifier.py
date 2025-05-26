import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import os
import struct

def read_idx(filename):
    """Read IDX file format used for MNIST dataset"""
    filepath = os.path.join('archive', filename)
    with open(filepath, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_train_data():
    """Load MNIST training data from ubyte files"""
    try:
        # Adjust these paths based on your actual file locations
        train_images = read_idx('train-images.idx3-ubyte')
        train_labels = read_idx('train-labels.idx1-ubyte')
        
        # Reshape and normalize the images
        X = train_images.reshape(len(train_images), -1).astype('float32') / 255.0
        y = train_labels
        
        return X, y
    except FileNotFoundError:
        raise Exception("MNIST ubyte files not found. Please ensure they are in the correct directory.")

def load_test_data():
    """Load MNIST test data from ubyte files"""
    try:
        test_images = read_idx('t10k-images.idx3-ubyte')
        test_labels = read_idx('t10k-labels.idx1-ubyte')
        
        X = test_images.reshape(len(test_images), -1).astype('float32') / 255.0
        y = test_labels
        
        return X, y
    except FileNotFoundError:
        raise Exception("MNIST ubyte files not found. Please ensure they are in the correct directory.")


model = KNeighborsClassifier()
X_train, y_train = load_train_data()
X_test, y_test = load_test_data()

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print(accuracy_score(y_test, y_pred))







