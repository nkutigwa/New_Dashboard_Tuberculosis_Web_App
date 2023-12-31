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
import json
import os
import pdfkit
from datetime import datetime
from streamlit_folium import folium_static  
import folium 



tb_cases_by_region = {}
user_credentials = {}
tb_map = None


if "show_demographic_content" not in st.session_state:
    st.session_state.show_demographic_content = False

def load_user_credentials():
    if os.path.exists("user_credentials.json"):
        with open("user_credentials.json") as f:
            return json.load(f)
    return {}

def save_user_credentials(credentials):
    with open("user_credentials.json", "w") as f:
        json.dump(credentials, f)

def sign_up():
    st.sidebar.markdown("<div class='custom-title'>Sign Up</div>", unsafe_allow_html=True)
    with st.sidebar.form("sign_up_form"):
        username = st.text_input("Username", key="sign_up_username")
        password = st.text_input("Password", type="password", key="sign_up_password")
        sign_up_button_clicked = st.form_submit_button("sig Up")

    if sign_up_button_clicked:
        if username == "":
            st.sidebar.error("Please enter a username.")
        elif password =="":
            st.sidebar.error("Please enter a password.")
        else:
            if username in user_credentials:
                st.sidebar.error("Username alredy exists. Please choose a different username.")
            else:
                user_credentials[username] = password
                save_user_credentials(user_credentials)
                st.sidebar.success("sign up successful. Please log in.")
                return True
            
    

    

def log_in():
    st.sidebar.markdown("<div class='custom-title'>Log In</div>", unsafe_allow_html=True)
    with st.sidebar.form("log_in_form"):
        username = st.text_input("Username", key="log_in_username")
        password = st.text_input("Password", type="password", key="log_in_password")
        log_in_button_clicked = st.form_submit_button("Log In")

    if log_in_button_clicked:
        if username == "":
            st.sidebar.error("Please enter a username.")
        elif password == "":
            st.sidebar.error("Please enter a password.")
        elif username in user_credentials and user_credentials[username] == password:
            st.sidebar.success("Log in successful!")
            return True
        else:
            st.sidebar.error("Invalid username or password.")
  
        


    
def get_region_coordinates(region):
    # Define a dictionary with coordinates for regions
    region_coordinates = {
        "Dar es Salaam": [-6.7924, 39.2083],
        "Arusha": [-3.3731, 36.6942],
        "Mwanza": [-2.5164, 32.9175],
        # Add more regions and their coordinates here
    }
    return region_coordinates.get(region, [0, 0])  

def record_tb_cases(region, date):
    if region in tb_cases_by_region:
        tb_cases_by_region[region].append(date)
    else:
        tb_cases_by_region[region] = [date]

    tb_map = folium.Map(location=get_region_coordinates(region), zoom_start=6)
    for r, cases in tb_cases_by_region.items():
        folium.Marker(
            location=get_region_coordinates(r),
            popup=f"Region: {r}\nCases: {cases}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(tb_map)

    folium_static(tb_map)




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



def generate_pdf_report(df):
    if df is None or df.empty:
        st.write("No prediction results to export.")
        return

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
    <th>Filename</th>
    </tr>
    </thead>
    <tbody>
    """

    for _, row in df.iterrows():
        report_html += "<tr>"
        report_html += f"<td><img src='data:image/png;base64,{row['Image']}' width='200px'></td>"
        report_html += f"<td>{row['Tuberculosis Percentage']}%</td>"
        report_html += f"<td>{row['Normal Percentage']}%</td>"
        report_html += f"<td>{row['Filename']}</td>"
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




def main():
    
    global show_demographic_content
    global tb_map

    prediction_results = []

    df = pd.DataFrame()


    global user_credentials
    user_credentials = load_user_credentials()

    # Function to toggle demographic content
    def toggle_demographic_content():
        st.session_state.show_demographic_content = not st.session_state.show_demographic_content


    

    st.markdown(
        """
        <div class="fixed-header">
            <h1 class="header-title">Tuberculosis Detection App</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

     
    default_image = "assets/tb.jpg"  
    st.image(default_image, use_column_width=True)

    is_logged_in = False

    if "is_logged_in" not in st.session_state:
        st.session_state.is_logged_in = False

    col1, col2 = st.columns(2)

    with col1:
        if not st.session_state.is_logged_in:
            st.sidebar.markdown("<div class='custom-title'>Sign Up</div>", unsafe_allow_html=True)
            with st.sidebar.form("sign_up_form"):
                username = st.text_input("Username", key="sign_up_username")
                password = st.text_input("Password", type="password", key="sign_up_password")
                sign_up_button_clicked = st.form_submit_button("Sign Up")

            if sign_up_button_clicked:
                if username == "":
                    st.sidebar.error("Please enter a username.")
                elif password == "":
                    st.sidebar.error("Please enter a password.")
                else:
                    if username in user_credentials:
                        st.sidebar.error("Username already exists. Please choose a different username.")
                    else:
                        user_credentials[username] = password
                        save_user_credentials(user_credentials)
                        st.sidebar.success("Sign up successful. Please log in.")
        
        else:
            st.sidebar.markdown("<div class='custom-title'>Logged In</div>", unsafe_allow_html=True)
            st.sidebar.write(f"Welcome, {st.session_state.logged_in_user}!")
            if st.sidebar.button("Logout"):
                st.session_state.is_logged_in = False
                st.session_state.logged_in_user = None

    with col2:
        if not st.session_state.is_logged_in:
            st.sidebar.markdown("<div class='custom-title'>Log In</div>", unsafe_allow_html=True)
            with st.sidebar.form("log_in_form"):
                username = st.text_input("Username", key="log_in_username")
                password = st.text_input("Password", type="password", key="log_in_password")
                log_in_button_clicked = st.form_submit_button("Log In")

            if log_in_button_clicked:
                if username == "":
                    st.sidebar.error("Please enter a username.")
                elif password == "":
                    st.sidebar.error("Please enter a password.")
                elif username in user_credentials and user_credentials[username] == password:
                    st.session_state.is_logged_in = True
                    st.session_state.logged_in_user = username
                    st.session_state.show_demographic_content = False

                    st.sidebar.success("Log in successful!")



    
    if st.session_state.is_logged_in:
        st.write("Welcome to the Tuberculosis Detection web app. This app allows you to upload chest X-ray images and predicts the presence of Tuberculosis using a deep learning model.")
        st.write("Simply fill in the demographic details and then drag and drop or click to upload images in PNG, JPG, or JPEG format. Once the images are uploaded, the app will process each image and display the prediction results.")

        # Demographic input fields
        st.sidebar.markdown("<div class='custom-title'>Demographic Details</div>", unsafe_allow_html=True)
        gender = st.sidebar.selectbox("Select Gender", ["Male", "Female", "Other"])
        age = st.sidebar.slider("Select Age", 1, 100, 30)
        location = st.sidebar.text_input("Enter Location")

        st.sidebar.markdown("<div class='custom-title'>Upload Images</div>", unsafe_allow_html=True)
        st.sidebar.write("Upload chest X-ray images to detect the presence of Tuberculosis.")

        uploaded_files = st.sidebar.file_uploader(
                " ",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="fileUploader",
                )

        

        if uploaded_files:
                for i, uploaded_file in enumerate(uploaded_files):
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


        # Only show "Show Demographic Content" checkbox if the user is logged in
        if st.session_state.logged_in_user:
            show_demographic_content = st.sidebar.checkbox("Show Demographic Content", key="show_demographic_content", value=st.session_state.show_demographic_content, on_change=toggle_demographic_content)
            
        
        if st.session_state.show_demographic_content:
            # Load the map and add markers for each region with TB cases
            tb_map = folium.Map(location=[-6.369028, 34.888822], zoom_start=6)  # Adjust the center and zoom level accordingly

            for region, cases in tb_cases_by_region.items():
                folium.Marker(
                    location=get_region_coordinates(region),
                    popup=f"Region: {region}\nCases: {cases}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(tb_map)

            if tb_map is not None:
                folium_static(tb_map)

        # Export functionality
        if st.button(f'Export Data'):
            if len(prediction_results) > 0:
                df = pd.DataFrame(prediction_results)
                generate_pdf_report(df)
            else:
                st.write("No prediction results to export.")    

         # Record TB cases by region and date
        date_today = datetime.today().strftime('%Y-%m-%d')
        record_tb_cases(region='Dar es Salaam', date=date_today)

        st.sidebar.write("**More TB resources.**")
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
                    st.write("- [TB prevalence in Tanzania](https://ntlp.go.tz/tuberculosis/tb-prevalence-in-tanzania/)")
                    st.write("- [National Tuberculosis and Leprosy Strategic Plan VI 2020-2025](https://ntlp.go.tz/resources/national-strategic-plan/)")
                    st.write("- [End TB Programme](https://endtb.org/)")
        



        st.markdown("<div class='fixed-footer'>Developed by team of data scientist from CoICT, UDSM.</div>", unsafe_allow_html=True)

        
if __name__ == '__main__':
    main()