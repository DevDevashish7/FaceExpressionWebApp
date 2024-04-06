import os
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
# Load the model
model = load_model("model2.h5")
# Print model summary
model.summary()
app = Flask(__name__)

# Function to load and preprocess an image with resizing
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load the class labels (replace [...] with your actual class labels)
class_labels = ['Ahegao', 'Angry', 'Happy','Neutral','Sad','Surprise'] 
# Function to predict expression from an image
def predict_expression(image_path, model):
    img_array = load_and_preprocess_image(image_path, target_size=(128, 128))
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

# Function to display image with labels
def display_image_with_labels(image_path, actual_class, predicted_class):
    img = load_img(image_path, target_size=(224, 224))  # Resize to (224, 224)
    plt.imshow(img)
    plt.title(f'Actual: {actual_class}, Predicted: {predicted_class}')
    plt.axis('off')
    st.pyplot()

# Main Streamlit app
def main():
    st.title('Expression Detection')

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            # Get a random actual class (replace this with actual logic)
            actual_class = random.choice(class_labels)

            # Predict expression
            predicted_class = predict_expression(uploaded_file, model2)

            # Display the predicted class
            st.write(f"Predicted Expression: {predicted_class}")

            # Display the image with actual and predicted classes
            display_image_with_labels(uploaded_file, actual_class, predicted_class)

if __name__ == "__main__":
    main()
