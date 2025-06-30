import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

# Function to load the model
def load_model_by_name(model_name):
    if model_name == 'DenseNet201':
        return tf.keras.models.load_model('DenseNet201_Model.keras')
    elif model_name == 'VGG16':
        return tf.keras.models.load_model('VGG16_Model.keras')
    elif model_name == 'ResNet50':
        return tf.keras.models.load_model('ResNet50_Model.keras')
    else:
        raise ValueError("Model not recognized")

# Load the class labels from the JSON file
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Reverse the dictionary to map indices to class labels
index_to_label = {v: k for k, v in class_labels.items()}

# Function to preprocess the image for prediction
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image to [0, 1]
    return img_array

# Streamlit App UI
st.title("Tomato Disease Classifier with Multiple Models")
st.write("Upload an image of a tomato plant leaf, select a model, and the model will predict the disease.")

# Model selection
model_option = st.selectbox(
    'Select Model:',
    ['DenseNet201', 'VGG16', 'ResNet50']
)

# Load selected model
model = load_model_by_name(model_option)

# Image uploader widget
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(uploaded_image)

    # Predict the class of the image
    predictions = model.predict(img_array)

    # Get the predicted class (class with the highest probability)
    predicted_class_index = np.argmax(predictions, axis=1)

    # Get the predicted class label name from the index_to_label dictionary
    predicted_class_label = index_to_label.get(predicted_class_index[0], "Unknown Class")
    predicted_class_prob = predictions[0][predicted_class_index[0]]

    # Display the predicted result
    st.write(f"Predicted Class Label: **{predicted_class_label}**")
    st.write(f"Prediction Probability: **{predicted_class_prob:.4f}**")
