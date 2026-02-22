# CIFAR-10 Classifier

## What This Does

Image classification on the CIFAR-10 dataset comparing Convolutional Neural Networks (with and without hyperparameter tuning) against a Logistic Regression baseline. The dataset contains 60,000 32x32 color images across 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The project includes Streamlit applications for interactive prediction on uploaded images.

## How It Works

The pipeline loads CIFAR-10 from pickle batch files, normalizes pixel values to the [0, 1] range, and trains three model variants.

**Baseline CNN (`train_cnn.py`)**

A straightforward convolutional network:

```
Conv2D(32, 3x3, relu) -> MaxPooling2D(2x2)
-> Conv2D(64, 3x3, relu) -> MaxPooling2D(2x2)
-> Conv2D(64, 3x3, relu)
-> Flatten -> Dense(64, relu) -> Dense(10, softmax)
```

Trained for 10 epochs with the Adam optimizer and sparse categorical crossentropy loss.

**Tuned CNN (`train_cnn_tuned.py`)**

A deeper architecture with regularization and data augmentation:

- **Architecture**: 6 `Conv2D` layers with `BatchNormalization` after each convolutional block. Progressive `Dropout` rates (0.2 -> 0.3 -> 0.4 -> 0.5) applied between blocks to reduce overfitting.
- **Data augmentation**: `ImageDataGenerator` applies random rotation (up to 15 degrees), width/height shifts (up to 10%), horizontal flips, and zoom (up to 10%) during training.
- **Callbacks**: `EarlyStopping` monitors validation loss with `patience=5` and restores the best weights. `ModelCheckpoint` saves the model whenever validation accuracy improves.
- **Training**: Adam optimizer with `lr=0.001`, up to 30 epochs (typically stops early around epoch 15-20).

**Logistic Regression (`train_logistic_regression.py`)**

A simple dense network used as a baseline comparison:

```
Flatten(32x32x3) -> Dense(128, relu) -> Dense(10, softmax)
```

Trained for 10 epochs. Treats the raw pixel values as flat feature vectors without convolutional feature extraction.

**Streamlit Inference**

Two Streamlit applications (`streamlit_cnn.py` and `streamlit_lr.py`) allow users to upload an image, which is resized to 32x32, normalized, and classified by the selected model. The app displays the predicted class along with confidence scores for all 10 categories.

## Sample Output

**Tuned CNN Training**

```
Epoch 1/30 - loss: 1.4532 - accuracy: 0.4621 - val_loss: 1.0234 - val_accuracy: 0.6312
Epoch 5/30 - loss: 0.7812 - accuracy: 0.7234 - val_loss: 0.7123 - val_accuracy: 0.7521
Epoch 10/30 - loss: 0.5432 - accuracy: 0.8112 - val_loss: 0.5678 - val_accuracy: 0.8056
Epoch 15/30 - loss: 0.4123 - accuracy: 0.8567 - val_loss: 0.5234 - val_accuracy: 0.8234
EarlyStopping triggered at epoch 18
Test accuracy: 84.56%
```

**Confusion Matrix (excerpt)**

```
Confusion Matrix (top-left corner):
         airplane  auto  bird   cat  deer
airplane   [892    12    34     8    15 ...]
auto       [  8   945     3     5     2 ...]
bird       [ 25     5   821    42    38 ...]
```

## Quick Start

```bash
# Install dependencies and set up the environment
./setup.sh

# Train the baseline CNN
python training/train_cnn.py

# Train the tuned CNN (with data augmentation and EarlyStopping)
python training/train_cnn_tuned.py

# Train the Logistic Regression baseline
python training/train_logistic_regression.py

# Run the Streamlit app for CNN predictions
streamlit run inference/streamlit_cnn.py

# Run the Streamlit app for Logistic Regression predictions
streamlit run inference/streamlit_lr.py
```

### Prerequisites

- Python 3.8+
- pip

## Configuration

Configuration is managed through `config.py` and environment variables loaded from a `.env` file. Key configuration values include `BASE_DIR`, `DATA_DIR`, `MODELS_DIR`, and `BATCH_DIR`, which control where the CIFAR-10 data and trained models are stored.

### Dependencies

Key libraries: tensorflow, numpy, scikit-learn, streamlit, Pillow, python-dotenv, matplotlib

All dependencies are listed in `requirements.txt`.

## Project Structure

```
cifar10-classifier/
├── config.py                        # BASE_DIR, DATA_DIR, MODELS_DIR, BATCH_DIR
├── data/cifar-10-batches-py/        # CIFAR-10 pickle batch files
├── models/                          # Saved .keras and .h5 model files
├── training/
│   ├── train_cnn.py                 # Baseline CNN (3 conv layers, 10 epochs)
│   ├── train_cnn_tuned.py           # Tuned CNN with augmentation + EarlyStopping
│   ├── train_logistic_regression.py # Simple dense network baseline
│   └── train_lg_experiment.py       # Dense network experiment variant
├── inference/
│   ├── streamlit_cnn.py             # Interactive CNN prediction UI
│   └── streamlit_lr.py              # Interactive LR prediction UI
├── requirements.txt
└── setup.sh
```

### Key Directories

- **training/** -- Model training scripts. `train_cnn.py` trains a baseline 3-layer CNN. `train_cnn_tuned.py` adds BatchNormalization, progressive Dropout, data augmentation, and EarlyStopping for improved generalization. `train_logistic_regression.py` provides a flat dense network baseline for comparison. `train_lg_experiment.py` is an experimental variant of the dense network.
- **inference/** -- Streamlit applications for interactive image classification. Upload any image and get predictions with confidence scores from either the CNN or Logistic Regression model.
- **models/** -- Saved trained models in Keras format (`.keras`, `.h5`). Generated by the training scripts and loaded by the Streamlit inference apps.
- **data/** -- CIFAR-10 dataset stored as Python pickle batch files (`data_batch_1` through `data_batch_5` for training, `test_batch` for evaluation).
