import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.ndimage import shift
from scipy.stats import randint
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

def shift_image_by_one(image, direction):
    """Shift image by one pixel in the specified direction.
    direction should be a tuple indicating the shift in each dimension"""
    return shift(image.reshape(28, 28), direction, cval=0).reshape(784)

model = KNeighborsClassifier()
X_train, y_train = load_train_data()
X_test, y_test = load_test_data()

# Create shifted versions of training images
X_train_augmented = []
y_train_augmented = []

print("Augmenting training data with shifted images...")
for idx, image in enumerate(X_train):
    # Add original image
    X_train_augmented.append(image)
    y_train_augmented.append(y_train[idx])
    
    # Add shifted versions
    shifts = [(0, 1), (0, -1)]  # down, up
    for direction in shifts:
        X_train_augmented.append(shift_image_by_one(image, direction))
        y_train_augmented.append(y_train[idx])

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

print(f"Training set expanded from {len(X_train)} to {len(X_train_augmented)} images")

param_grid = {
    'n_neighbors': randint(low=3, high=50),
    'weights': ['uniform', 'distance']
}

print("Starting RandomizedSearchCV...")
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_grid, 
    n_iter=10, 
    cv=5,
    verbose=2    # Show detailed progress
)
random_search.fit(X_train_augmented, y_train_augmented)

best_model = random_search.best_estimator_
print("\nSearch completed!")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

y_pred = best_model.predict(X_test)

print(f"\nFinal test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Total number of training samples used: {len(X_train_augmented)}")







