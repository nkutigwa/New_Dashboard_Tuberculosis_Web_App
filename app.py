import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Tuberculosis Detection Web App", layout="wide")

# Load the pre-trained ResNet-50 model
model = ResNet50(weights='imagenet')

# Function to process the uploaded image
def process_image(image):
    # Preprocess the image
    image = image.convert('RGB')  # Convert to RGB format
    image = image.resize((224, 224))  # Resize to match the input shape of the model
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    processed_image = tf.expand_dims(img_array, axis=0)

    return processed_image

# Streamlit app
def main():
    # Custom CSS styles
    st.markdown(
        """
        <style>
        /* Set header as fixed */
        .reportview-container .main .block-container {
            position: sticky;
            top: 0;
            z-index: 100;
        }

        /* Hide main content */
        .reportview-container .main .block-container:not(.custom-footer) {
            display: none;
        }

        /* Hide sidebar */
        .reportview-container .main .sidebar .sidebar-content {
            display: none;
        }

        /* Custom footer */
        .custom-footer {
            font-size: 12px;
            margin-top: 30px;
            color: #808080;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.title("Tuberculosis Detection Web App")
    st.write("Welcome to the Tuberculosis Detection web app. This app allows you to upload chest X-ray images and predicts the presence of Tuberculosis using a pre-trained deep learning model.")
    st.write("Simply upload images in PNG, JPG, or JPEG format using the file uploader. Once the images are uploaded, the app will process each image and display the prediction results along with the confidence scores.")

    st.sidebar.title("Upload Images")
    st.sidebar.write("Upload chest X-ray images to detect the presence of Tuberculosis.")

    # Display file uploader in the sidebar
    uploaded_files = st.sidebar.file_uploader("Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read and process the uploaded image
            image = Image.open(uploaded_file)
            processed_image = process_image(image)

            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True, width=100)

            # Make predictions
            predictions = model.predict(processed_image)
            decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0]
            predicted_label = decoded_predictions[0][1]
            probability = decoded_predictions[0][2]

            # Calculate complement of Tuberculosis percentage
            normal_percentage = 100 - (probability * 100)

            # Display the result
            st.write("Prediction:", predicted_label)
            st.write("Tuberculosis Percentage:", round(probability * 100, 2), "%")
            st.write("Normal Percentage:", round(normal_percentage, 2), "%")

    # Custom footer
    st.markdown(
        """
        <footer class='custom-footer'>
            <p>App designed by a team of data scientists from the College of Coict, UDSM.</p>
            <p>Contact us: nkw802640@gmail.com | Phone number: +255757650442</p>
            <p>&#169;2023. All rights reserved.</p>
        </footer>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
