import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import os

def prepare_data():
    print("Downloading and preparing Fashion MNIST...")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Flatten if needed or keep as images. For agnostic sklearn tools, we flatten.
    X_train_flat = X_train.reshape(len(X_train), -1) / 255.0
    X_test_flat = X_test.reshape(len(X_test), -1) / 255.0
    
    # Subsample for faster testing
    X_train_sub = X_train_flat[:5000]
    y_train_sub = y_train[:5000]
    X_test_sub = X_test_flat[:1000]
    y_test_sub = y_test[:1000]
    
    os.makedirs("data", exist_ok=True)
    file_path = "data/fashion_mnist_agnostic.npz"
    np.savez(file_path, 
             X_train=X_train_sub, 
             y_train=y_train_sub, 
             X_test=X_test_sub, 
             y_test=y_test_sub)
    
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    prepare_data()
