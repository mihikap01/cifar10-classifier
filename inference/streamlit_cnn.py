import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model(str(MODELS_DIR / "best_model.h5"))

# Streamlit web application
st.title("CIFAR-10 Image Classifier using CNN")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Processing the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image = image.resize((32, 32))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape((1, 32, 32, 3))

    # Making a prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Displaying the top prediction
    st.write(f"Top Prediction: {class_names[predicted_class]} (Class {predicted_class})")

    # Formatting and displaying prediction probabilities
    probabilities = np.round(prediction[0], 2)
    probability_str = "Probabilities:\n" + "\n".join(
        [f"{class_names[i]}: {probabilities[i]}" for i in range(len(class_names))]
    )
    st.markdown(probability_str)
