import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR, BATCH_DIR

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix


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

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    # Model Architecture
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Optimizer and Compilation
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(str(MODELS_DIR / "best_model.h5"), monitor='val_loss', save_best_only=True)

    # Training the model
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=30,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint])

    # Load the best model
    model.load_weights(str(MODELS_DIR / "best_model.h5"))

    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy*100:.2f}%")

    # Confusion Matrix
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(conf_matrix)
