import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR, BATCH_DIR

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import pickle
import numpy as np


# Load CIFAR-10 data from multiple training files and one test file
def load_cifar10_data(data_files, test_file):
    x_train, y_train = [], []
    for file_path in data_files:
        with open(file_path, 'rb') as file:
            batch_data = pickle.load(file, encoding='latin1')
            images = batch_data.get('data').reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
            labels = batch_data.get('labels')
            x_train.append(images)
            y_train.extend(labels)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.array(y_train)

    with open(test_file, 'rb') as file:
        test_batch_data = pickle.load(file, encoding='latin1')
        x_test = test_batch_data.get('data').reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        y_test = np.array(test_batch_data.get('labels'))

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # List of paths to the 5 training data files
    training_data_files = [str(BATCH_DIR / f"data_batch_{i}") for i in range(1, 6)]
    # Path to the test data file
    test_file = str(BATCH_DIR / "test_batch")

    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(training_data_files, test_file)

    # Build the neural network model
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    accuracy = model.evaluate(x_test, y_test)[1]
    print(accuracy)

    # Test 1: 47.089
    #2 hidden layers: 128, 128

    # Test 2: 47.43
    # 3 hidded layers: 128, 256, 128

    # Test 3: 47.43
    # 3 hidden layers: 128, 128, 128
