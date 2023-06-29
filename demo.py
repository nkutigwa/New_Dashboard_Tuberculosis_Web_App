import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import streamlit.components.v1 as components
import pandas as pd
import base64
import os
import pdfkit
from datetime import datetime


st.set_page_config(page_title="Tuberculosis Detection Web App", layout="wide")

model = ResNet50(weights='imagenet')

def process_image(image):
    image = image.convert('RGB') 
    image = image.resize((224, 224))  
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    processed_image = tf.expand_dims(img_array, axis=0)

    return processed_image

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

    /* Add custom CSS for the fixed header */
    .fixed-header {
        position: fixed;
        top: 0;
        width: 100%;
        background-color: #f5f5f5;
        padding: 10px;
        z-index: 999;
    }
    .header-title {
        font-size: 35px;
        font-weight: bold;
        color: #333333;
        margin: 0;
        font-family: "Arial", sans-serif; /* Custom font */
    }

    /* Set default image width */
    img {
        max-width: 300px;
        display: block;
        margin: 0 auto;
    }

    /* Change sidebar background color */
    .sidebar .sidebar-content {
        background-color: #f08db3;
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

    /* Customize the title style */
    .custom-title {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
        background-color: #FF3366; /* Custom background color */
        padding: 10px;
        margin-bottom: 10px;
        font-family: "Verdana", sans-serif; /* Custom font */
        text-align: center;
    }

    /* Modify the header section background color */
    .css-1629p8f {
        background-color: #f5f5f5;
    }

    /* Dropzone styles */
    .dropzone {
        border: 2px dashed #f08db3;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f5f5f5;
        cursor: pointer;
    }
    .dropzone:hover {
        background-color: #fce0ed;
    }
    .dropzone .dz-message {
        font-size: 18px;
        font-weight: bold;
        color: #333333;
    }
    .dropzone .dz-message span {
        color: #FF3366; /* Custom text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load custom CSS styles for PDF generation
pdfkit_css = """
    <style>
    /* Add custom styles for PDF generation */
    .pdf-header {
        text-align: center;
        padding: 20px;
    }
    .pdf-header h1 {
        font-size: 30px;
        color: #FF3366; /* Custom header color */
        margin: 0;
    }
    .pdf-header p {
        font-size: 18px;
        color: #333333;
        margin: 0;
    }
    .pdf-image {
        max-width: 100%;
        margin-bottom: 20px;
    }
    .pdf-results {
        margin-bottom: 20px;
    }
    .pdf-results h2 {
        font-size: 24px;
        color: #FF3366; /* Custom header color */
        margin-bottom: 10px;
    }
    .pdf-results table {
        width: 100%;
        border-collapse: collapse;
    }
    .pdf-results th,
    .pdf-results td {
        border: 1px solid #333333;
        padding: 10px;
        text-align: left;
    }
    </style>
"""

def main():
    st.markdown("<h1 class='custom-title'>Tuberculosis Detection Web App</h1>", unsafe_allow_html=True)

    components.html(
        """
        <div class="fixed-header">
            <h1 class="header-title">Tuberculosis Detection Web App</h1>
        </div>
        """,
        height=200,
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image = process_image(image)

        prediction = model.predict(processed_image)
        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(prediction, top=3)[0]

        # Calculate percentages
        probabilities = prediction[0]
        normal_probability = probabilities[0]
        tuberculosis_probability = probabilities[1]

        total_probability = normal_probability + tuberculosis_probability
        normal_percentage = normal_probability / total_probability
        tuberculosis_percentage = tuberculosis_probability / total_probability

        # Store prediction results
        prediction_results = []
        img_base64 = base64.b64encode(uploaded_file.read()).decode("utf-8")

        prediction_results.append({
            'Image': img_base64,
            'Predictions': decoded_predictions,
            'Tuberculosis Percentage': round(tuberculosis_percentage * 100, 2),
            'Normal Percentage': round(normal_percentage * 100, 2)
        })

        with st.beta_expander("View Prediction Results"):
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown("<div class='center-align'>Tuberculosis Percentage: {0}%</div>".format(round(tuberculosis_percentage * 100, 2)), unsafe_allow_html=True)
                st.markdown("<div class='center-align'>Normal Percentage: {0}%</div>".format(round(normal_percentage * 100, 2)), unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='center-align'>Prediction Results:</div>", unsafe_allow_html=True)
                st.write("##### Top Predictions:")
                for i, prediction in enumerate(decoded_predictions):
                    label = prediction[1]
                    score = prediction[2]
                    st.write("{0}. {1} ({2}%)".format(i+1, label, round(score*100, 2)))

        # Download results as PDF
        if st.button("Download Results as PDF"):
            # Generate HTML for PDF
            pdf_html = "<html><head>{}</head><body>".format(pdfkit_css)
            pdf_html += "<div class='pdf-header'>"
            pdf_html += "<h1>Tuberculosis Detection Results</h1>"
            pdf_html += "<p>Date: {}</p>".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            pdf_html += "</div>"
            pdf_html += "<img class='pdf-image' src='data:image/png;base64,{}'>".format(img_base64)
            pdf_html += "<div class='pdf-results'>"
            pdf_html += "<h2>Top Predictions:</h2>"
            pdf_html += "<table>"
            pdf_html += "<tr><th>Rank</th><th>Label</th><th>Score (%)</th></tr>"
            for i, prediction in enumerate(decoded_predictions):
                label = prediction[1]
                score = round(prediction[2] * 100, 2)
                pdf_html += "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(i + 1, label, score)
            pdf_html += "</table>"
            pdf_html += "</div>"
            pdf_html += "</body></html>"

            # Save HTML to file
            html_file = "prediction_results.html"
            with open(html_file, "w") as file:
                file.write(pdf_html)

            # Convert HTML to PDF
            pdf_file = "prediction_results.pdf"
            pdfkit.from_file(html_file, pdf_file)

            # Download PDF file
            with open(pdf_file, "rb") as file:
                b64_pdf = base64.b64encode(file.read()).decode("utf-8")
                href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_file}">Download PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

            # Cleanup temporary files
            os.remove(html_file)
            os.remove(pdf_file)

if __name__ == "__main__":
    main()
