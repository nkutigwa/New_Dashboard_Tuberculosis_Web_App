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

    .stButton > button[data-testid^="stButton"] {
        background-color: blue !important;
        color: white !important;
    }
    

    .sidebar .button-container button {
        background-color: blue;
        color: white;
    }
 
    .stButton > button[data-testid^="stButton"] {
        background-color: blue !important;
        color: white !important;
    }

    .sidebar .button-container button {
        background-color: blue;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.markdown(
        """
        <div class="fixed-header">
            <h1 class="header-title">Tuberculosis Detection App</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Welcome to the Tuberculosis Detection web app. This app allows you to upload chest X-ray images and predicts the presence of Tuberculosis using a deep learning model.")
    st.write("Simply drag and drop or click to upload images in PNG, JPG, or JPEG format. Once the images are uploaded, the app will process each image and display the prediction results.")
    
    


    st.sidebar.markdown("<div class='custom-title'>Upload Images</div>", unsafe_allow_html=True)
    st.sidebar.write("Upload chest X-ray images to detect the presence of Tuberculosis.")

    uploaded_files = st.sidebar.file_uploader(
        " ",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="fileUploader",
    )

    prediction_results = []
        

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            processed_image = process_image(image)

            col1, col2 = st.columns([2, 1])

            col1.image(image, caption='Uploaded Image', use_column_width=True, width=250)

            predictions = model.predict(processed_image)
            decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0]
            predicted_label = decoded_predictions[0][1]
            probability = decoded_predictions[0][2]

            normal_percentage = 100 - (probability * 100)

            img_str = uploaded_file.getvalue()
            img_base64 = base64.b64encode(img_str).decode()

            prediction_results.append({
                'Image': img_base64,
                'Filename': uploaded_file.name,
                'Tuberculosis Percentage': round(probability * 100, 2),
                'Normal Percentage': round(100 - (probability * 100), 2)
            })
        

            with col2:
                st.markdown("<div class='center-align'>Prediction Results:</div>", unsafe_allow_html=True)
                st.markdown("<div class='center-align'>Tuberculosis Percentage: {0}%</div>".format(round(probability * 100, 2)), unsafe_allow_html=True)
                st.markdown("<div class='center-align'>Normal Percentage: {0}%</div>".format(round(100 - (probability * 100), 2)), unsafe_allow_html=True)
    
    st.sidebar.write("To access further details about Tuberculosis, please click the button provided below.")


    if st.sidebar.button('Tuberculosis Information', key='tb_info_button', help='Tuberculosis Information'):
        with st.expander("Tuberculosis Information"):
            st.write("##### What is Tuberculosis?")
            st.write("Tuberculosis is a contagious bacterial infection that primarily affects the lungs. It is caused by the bacteria Mycobacterium tuberculosis.")
         
            st.write("##### Why is early detection of Tuberculosis crucial?")
            st.write("Early detection of Tuberculosis is crucial for the following reasons:")
            st.write("- **Effective treatment**: Early detection allows for timely initiation of treatment, which improves the chances of successful recovery.")
            st.write("- **Prevention of transmission**: Detecting and treating Tuberculosis early helps prevent the spread of the infection to others.")
            st.write("- **Reduced complications**: Early intervention reduces the risk of developing severe complications associated with Tuberculosis, such as organ damage or dissemination of the infection to other parts of the body.")
            st.write("- **Improved outcomes**: Timely detection and treatment increase the likelihood of a positive treatment outcome and minimize the impact of Tuberculosis on an individual's health and well-being.") 

            st.write("##### What are the causes of Tuberculosis?")
            st.write("Tuberculosis is primarily caused by inhaling air droplets containing the bacteria Mycobacterium tuberculosis. It can be spread when an infected person coughs, sneezes, or talks, releasing the bacteria into the air.")

            st.write("##### What are the symptoms of Tuberculosis?")
            st.write("Common symptoms of Tuberculosis include:")
            st.write ("- persistent coughing")
            st.write ("- chest pain ")
            st.write ("- coughing up blood") 
            st.write ("- fatigue, weight loss")
            st.write ("- fever")
            st.write ("- night sweats.")

            st.write("##### What are the preventive measures to be taken?")
            st.write("To prevent Tuberculosis, it is recommended to:")
            st.write("- Get vaccinated with the Bacillus Calmette-Guérin (BCG) vaccine.")
            st.write("- Maintain good ventilation and air circulation in living spaces.")
            st.write("- Practice good respiratory hygiene, such as covering the mouth and nose when coughing or sneezing.")
            st.write("- Avoid close contact with individuals who have active Tuberculosis.")

            st.write("##### Tips for someone suffering from Tuberculosis")
            st.write("If you are diagnosed with Tuberculosis, consider the following tips:")
            st.write("- Take the prescribed medication regularly and complete the full course of treatment.")
            st.write("- Follow the healthcare provider's instructions regarding respiratory hygiene and infection control.")
            st.write("- Maintain a healthy lifestyle, including a balanced diet, regular exercise, and adequate rest.")
            st.write("- Seek support from healthcare professionals, family, and friends to manage the condition effectively.")

            st.write("##### Educational Resources")
            st.write("If you want to learn more about Tuberculosis, here are some educational resources you can explore:")
            st.write("- [World Health Organization (WHO) - Tuberculosis](https://www.who.int/health-topics/tuberculosis)")
            st.write("- [Centers for Disease Control and Prevention (CDC) - Tuberculosis](https://www.cdc.gov/tb/index.html)")
            st.write("- [Mayo Clinic - Tuberculosis](https://www.mayoclinic.org/diseases-conditions/tuberculosis/symptoms-causes/syc-20351250)")
        


    if st.button('Export Data'):
        if len(prediction_results) > 0:
            df = pd.DataFrame(prediction_results)

            generate_pdf_report(df)


    st.markdown("<div class='fixed-footer'>Developed by team of data scientist from CoICT, UDSM.</div>", unsafe_allow_html=True)

def generate_pdf_report(df):
    report_html = """
    <html>
    <head>
        <style>
            h2 {
                text-align: center;
                color: #FF3366;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h2>Tuberculosis Detection Report</h2>
        <table>
            <thead>
                <tr>
                    <th>Image</th>
                    <th>Tuberculosis Percentage</th>
                    <th>Normal Percentage</th>
                </tr>
            </thead>
            <tbody>
    """

    # for _, row in df.iterrows():
    #     report_html += "<tr>"
    #     report_html += "<td><img src='data:image/png;base64,{0}' width='200px'></td>".format(row['Image'])
    #     report_html += "<td>{0}</td>".format(row['Filename'])
    #     report_html += "<td>{0}%</td>".format(row['Tuberculosis Percentage'])
    #     report_html += "<td>{0}%</td>".format(row['Normal Percentage'])
    #     report_html += "</tr>
    for _, row in df.iterrows():
        report_html += "<tr>"
        report_html += "<td><img src='data:image/png;base64,{0}' width='200px'></td>".format(row['Image'])
        report_html += "<td>{0}%</td>".format(row['Tuberculosis Percentage'])
        report_html += "<td>{0}%</td>".format(row['Normal Percentage'])
        report_html += "</tr>"
        report_html += "<tr>"
        report_html += "<td><strong>Filename:</strong> {0}</td>".format(row['Filename'])
        report_html += "<td colspan='2'></td>"
        report_html += "</tr>"


    report_html += """
            </tbody>
        </table>
    </body>
    </html>
    """

    with open("report.html", "w") as file:
        file.write(report_html)

    pdfkit.from_file("report.html", "report.pdf")

    with open("report.pdf", "rb") as file:
        b64_pdf = base64.b64encode(file.read()).decode()

    st.markdown(
        f'<a href="data:application/pdf;base64,{b64_pdf}" download="report.pdf">Download Report</a>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()