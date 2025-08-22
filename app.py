import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("water_potability_model.pkl")

st.set_page_config(page_title="💧 Water Potability Prediction", page_icon="💧", layout="wide")

# Custom CSS for styling
# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        color: #000000;
    }

    /* Input Labels */
    .stSlider label {
        font-size:20px !important;
        font-weight: 700 !important;
        color: #1b1b1b !important;
    }

    /* 🎯 Slider customization */

    /* Track line (gray) */
    div[data-baseweb="slider"] [data-testid="stTrack"] {
        background: gray !important;
    }

    /* Active filled part (orange) */
    div[data-baseweb="slider"] [data-testid="stTrack"] > div {
        background: orange !important;
    }

    /* Thumb (circle) */
    div[data-baseweb="slider"] [role="slider"] {
        background: black !important;
        border: 2px solid white !important;
    }

    /* Value text above thumb */
    div[data-baseweb="slider"] [data-testid="stThumbValue"] {
        color: black !important;
        font-weight: 700 !important;
    }

    /* Info Circle */
    .info-icon {
        display: inline-block;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background-color: #007BFF;
        color: white;
        text-align: center;
        font-size: 14px;
        font-weight: bold;
        cursor: pointer;
        margin-left: 8px;
        position: relative;
    }

    /* Tooltip */
    .info-icon:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        background: #333;
        color: #fff;
        padding: 8px;
        border-radius: 6px;
        top: 30px;
        left: -10px;
        font-size: 12px;
        white-space: normal;
        width: 220px;
        z-index: 999;
    }
    .stAlert p {
        color: black !important;
        font-weight: bold;
    }
    /* Button */
    .stButton button {
        font-size:18px;
        font-weight:600;
        border-radius: 10px;
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        border: none;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
    }

    /* Info Box Styling */
    .info-box {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 50px;
    }
    .info-box h3 {
        margin-bottom: 15px;
        font-size: 22px;
        font-weight: bold;
        color: #003366;
    }
    .info-box p {
        font-size: 16px;
        margin: 5px 0;
        color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("💧 Water Potability Prediction App")
st.markdown("Enter the water quality parameters below and check if the water is **Potable (Drinkable)** or not.")

# Feature details for tooltips
feature_info = {
    "pH Value": "pH measures acidity/alkalinity (0–14). Neutral water is around 7. Ideal range: 6.5–8.5.",
    "Hardness": "Concentration of calcium and magnesium salts. Too high = bitter taste, too low = pipe corrosion.",
    "Solids": "Total dissolved solids (TDS). High = salty, unsafe.",
    "Chloramines": "Disinfectant used in water treatment. Safe in small levels, harmful if too much.",
    "Sulfate": "High sulfate = diarrhea + bad taste.",
    "Conductivity": "Water’s ability to conduct electricity. Higher = more salts.",
    "Organic Carbon": "Organic pollutants in water. High = contamination.",
    "Trihalomethanes": "By-products of chlorine. Carcinogenic if high.",
    "Turbidity": "Cloudiness due to particles. High = unsafe."
}

# Function for slider with tooltip
def slider_with_info(label, min_val, max_val, default, step, key):
    col1, col2 = st.columns([5,1])
    with col1:
        val = st.slider(label, min_val, max_val, default, step=step, key=key)
    with col2:
        st.markdown(f'<div class="info-icon" data-tooltip="{feature_info[label]}">i</div>', unsafe_allow_html=True)
    return val

# Layout with sidebar info box
col_left, col_right = st.columns([1,3])

with col_left:
    st.markdown(
        """
        <div class="info-box">
            <h3>Created by Numair Amin</h3>
            <p>⚡ Model Accuracy: 67.3%</p>
            <p>🧠 Algorithm: Random Forest</p>
            <p>⚖️ Balancing: SMOTE</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_right:
    col1, col2, col3 = st.columns(3)
    with col1:
        ph = slider_with_info("pH Value", 0.0, 14.0, 7.0, 0.1, "ph")
    with col2:
        hardness = slider_with_info("Hardness", 0.0, 400.0, 150.0, 0.1, "hardness")
    with col3:
        solids = slider_with_info("Solids", 0.0, 50000.0, 20000.0, 100.0, "solids")

    col1, col2, col3 = st.columns(3)
    with col1:
        chloramines = slider_with_info("Chloramines", 0.0, 15.0, 7.0, 0.1, "chloramines")
    with col2:
        sulfate = slider_with_info("Sulfate", 0.0, 500.0, 300.0, 1.0, "sulfate")
    with col3:
        conductivity = slider_with_info("Conductivity", 0.0, 2000.0, 400.0, 1.0, "conductivity")

    col1, col2, col3 = st.columns(3)
    with col1:
        organic_carbon = slider_with_info("Organic Carbon", 0.0, 50.0, 10.0, 0.1, "organic")
    with col2:
        trihalomethanes = slider_with_info("Trihalomethanes", 0.0, 150.0, 60.0, 0.1, "tri")
    with col3:
        turbidity = slider_with_info("Turbidity", 0.0, 10.0, 4.0, 0.1, "turbidity")

# Prediction Button
if st.button("🔍 Check Potability", use_container_width=True):
    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ The water is **Potable (Safe to Drink).**")
    else:
        st.error("❌ The water is **Not Potable (Unsafe to Drink).**")
