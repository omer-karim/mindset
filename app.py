import streamlit as st
import pandas as pd
import pickle
import base64

# --- 1. SETUP PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hurricane Prediction Center",
    page_icon="🌪️",
    layout="centered" 
)

# --- 2. HELPER FUNCTION FOR BACKGROUND IMAGE ---
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and cover the whole screen.
    '''
    # set bg name
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
             background-attachment: fixed;
         }}
         /* Make the main container transparent background for glass effect */
         .block-container {{
             background-color: rgba(0, 0, 0, 0.65); /* Dark semi-transparent overlay */
             border-radius: 15px;
             padding: 3rem;
             margin-top: 50px;
         }}
         /* Change text color to white for contrast */
         h1, h2, h3, p, label {{
             color: white !important;
         }}
         .stNumberInput > label, .stSlider > label {{
             color: white !important;
             font-weight: bold;
             font-size: 16px;
         }}
         /* Style the buttons */
         .stButton > button {{
             background-color: #ff4b4b;
             color: white;
             font-weight: bold;
             border: none;
             border-radius: 5px;
             height: 3em;
             width: 100%;
         }}
         .stButton > button:hover {{
             background-color: #ff3333;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# --- 3. APPLY BACKGROUND ---
# Make sure the filename matches EXACTLY what you have in your folder
try:
    set_bg_hack('1714343434-nasa-hurricane-isabel.jpg')
except FileNotFoundError:
    st.warning("⚠️ Image not found! Please check the filename '1714343434-nasa-hurricane-isabel.jpg'")

# --- 4. LOAD MODEL ---
try:
    with open('hurricane_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Run the training script first.")
    st.stop()

# --- 5. THE INTERFACE ---

st.title("🌪️ HURRICANE WARNING SYSTEM")
st.markdown("### Meteorological Analysis Terminal")

st.write("---")

# Layout: 2 Columns for Inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📅 Temporal Data")
    year = st.number_input("Year", value=2023, step=1)
    month = st.slider("Month", 1, 12, 9)
    day = st.slider("Day", 1, 31, 15)
    hour = st.slider("Hour (UTC)", 0, 23, 12)

with col2:
    st.markdown("#### 🧭 Geospatial & Physics")
    lat = st.number_input("Latitude", value=25.0, format="%.2f")
    long = st.number_input("Longitude", value=-75.0, format="%.2f")
    pressure = st.number_input("Pressure (mb)", value=980, help="Lower pressure = Stronger storm")
    ts_diameter = st.number_input("Storm Diameter (nm)", value=120.0)

# Hidden Feature Engineering
pressure_anomaly = 1013 - pressure
abs_lat = abs(lat)

st.write("") # Spacer

# --- 6. PREDICTION ACTION ---
if st.button("RUN PREDICTION MODEL"):
    # Prepare input
    input_df = pd.DataFrame({
        'year': [year], 'month': [month], 'day': [day], 'hour': [hour],
        'lat': [lat], 'long': [long], 'pressure': [pressure],
        'tropicalstorm_force_diameter': [ts_diameter],
        'pressure_anomaly': [pressure_anomaly], 'abs_lat': [abs_lat]
    })
    
    # Predict
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    st.write("---")
    
    # Display Result with custom HTML for impact
    if pred == 1:
        st.markdown(f"""
            <div style="background-color: rgba(255, 0, 0, 0.8); padding: 20px; border-radius: 10px; text-align: center; border: 2px solid white;">
                <h1 style="margin:0; color: white;">🚨 ALERT: HURRICANE DETECTED 🚨</h1>
                <h3 style="color: white;">Probability: {prob:.1%}</h3>
                <p style="color: white;">Conditions indicate high likelihood of hurricane formation.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color: rgba(0, 128, 0, 0.8); padding: 20px; border-radius: 10px; text-align: center; border: 2px solid white;">
                <h1 style="margin:0; color: white;">✅ STATUS: TROPICAL STORM</h1>
                <h3 style="color: white;">Hurricane Probability: {prob:.1%}</h3>
                <p style="color: white;">Conditions do not currently meet hurricane criteria.</p>
            </div>
        """, unsafe_allow_html=True)