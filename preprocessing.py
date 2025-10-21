import os
import time
import numpy as np
import pandas as pd
from scipy.linalg import orth
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatter
from scipy.optimize import differential_evolution, NonlinearConstraint
from utils import *
np.random.seed(20)

def load_npy_files(path):
    train_images = np.load(f"{path}/train_images.npy")
    train_labels = np.load(f"{path}/train_labels.npy")
    test_images = np.load(f"{path}/test_images.npy")
    test_labels = np.load(f"{path}/test_labels.npy")
    return train_images,train_labels,test_images,test_labels

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def remove_nan_entries(images, labels):
    valid_idx = ~np.isnan(images).any(axis=1)
    return images[valid_idx], labels[valid_idx]

def save_processed_data(images, path):
    np.save(f"{path}.npy", images)

data_path = "data/mnist_c"
processed_path = "data/mnist_c_full_processed"

os.makedirs(processed_path, exist_ok=True)

corruption_types = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]

for corruption in corruption_types:
    train_images, train_labels, _ ,_ = load_npy_files(os.path.join(data_path, corruption))
    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)
    index_subsets = np.array_split(indices, 10)
    for i, index_subset in enumerate(index_subsets):
        subset_path = os.path.join(processed_path,corruption, f"{corruption}_{i}")
        os.makedirs(subset_path, exist_ok=True)

        subset_images = train_images[index_subset]
        subset_labels = train_labels[index_subset]

        binary_labels = (subset_labels == i).astype(int)
        np.save(os.path.join(subset_path, 'train_images.npy'), subset_images)
        np.save(os.path.join(subset_path, 'train_labels.npy'), binary_labels)
print("Data processing complete. Data saved to:", processed_path)


    
