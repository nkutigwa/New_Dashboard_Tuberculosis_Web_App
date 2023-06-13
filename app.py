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

# Custom CSS styles
st.markdown(
    """
    <style>
    /* Hide default Streamlit footer */
    .viewerFooter {
        display: none;
    }

    /* Position the title */
    .css-18ni7ap {
        position: sticky;
        top: 0;
        z-index: 999;
    }

    /* Hide Streamlit toolbar */
    [data-testid="stToolbar"] {
        display: none;
    }

    /* Set default caption width */
    [data-testid="caption"] {
        width: 300px;
    }

    /* Set default image width */
    img {
        max-width: 300px;
        display: block;
        margin: 0 auto;
    }

    /* Change sidebar background color */
    .sidebar .sidebar-content {
        background-color: #8a7878;
    }

    /* Center-align result text */
    .center-align {
        text-align: center;
    }

    /* Add a fixed footer */
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 10px;
        background-color: #f5f5f5;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
def main():
    # Header
    st.title("Tuberculosis Detection Web App")
    st.write("Welcome to the Tuberculosis Detection web app. This app allows you to upload chest X-ray images and predicts the presence of Tuberculosis using a deep learning model.")
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

            # Create two columns
            col1, col2 = st.columns([2, 1])

            # Display the uploaded image in the first column
            col1.image(image, caption='Uploaded Image', use_column_width=True, width=250)

            # Make predictions
            predictions = model.predict(processed_image)
            decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0]
            predicted_label = decoded_predictions[0][1]
            probability = decoded_predictions[0][2]

            # Calculate complement of Tuberculosis percentage
            normal_percentage = 100 - (probability * 100)

            # Display the result in the second column with center alignment
            with col2:
                st.markdown("<div class='center-align'>Prediction: {0}</div>".format(predicted_label), unsafe_allow_html=True)
                st.markdown("<div class='center-align'>Tuberculosis Percentage: {0}%</div>".format(round(probability * 100, 2)), unsafe_allow_html=True)
                st.markdown("<div class='center-align'>Normal Percentage: {0}%</div>".format(round(normal_percentage, 2)), unsafe_allow_html=True)

    # Fixed footer
    st.markdown(
        """
        <div class="fixed-footer">
        <p>App designed by a team of data scientists from the College of Coict, UDSM.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
