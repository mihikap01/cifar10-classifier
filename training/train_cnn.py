import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR, BATCH_DIR

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle
import numpy as np


# Function to load CIFAR-10 data
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

    # Build the neural network model with CNN layers
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    model.save(str(MODELS_DIR / "CIFAR-10_CNN.keras"))

    # Predicting the test set
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    from sklearn.metrics import confusion_matrix

    # Predict the labels of the test set
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
