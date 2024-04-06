import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model("model2.h5")

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

# Main Streamlit app
def main():
    st.title('Expression Detection')

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            try:
                # Predict expression
                predicted_class = predict_expression(uploaded_file, model)

                # Extract the actual class from the folder name where the image is located
                # actual_class = os.path.basename(os.path.dirname(uploaded_file.name))

                # Display the predicted and actual classes
                st.write(f"Predicted Expression: {predicted_class}")
                # st.write(f"Actual Expression: {actual_class}")

                # Plot the image with actual and predicted classes
                fig, ax = plt.subplots()
                ax.imshow(load_img(uploaded_file, target_size=(224, 224)))  # Resize to (224, 224)
                ax.set_title(f'Predicted: {predicted_class}')
                ax.axis('off')
                st.pyplot(fig)  # Pass the figure to st.pyplot()
            except Exception as e:
                st.write(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
