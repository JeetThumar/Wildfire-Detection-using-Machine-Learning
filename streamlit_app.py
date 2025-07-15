import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load trained Random Forest model
model = joblib.load("rf_model.pkl")

# Load sample data for visualization (correlation heatmap)
df = pd.read_csv("synthetic_wildfire_data.csv")

# App Title and Intro
st.set_page_config(page_title="Wildfire Detection App", layout="centered")
st.title("🔥 Wildfire Detection using Machine Learning")
st.markdown("""
Welcome to the **Wildfire Detection System**!  
This app predicts the possibility of wildfire using real-time satellite parameters.  
Enter the environmental conditions in the sidebar and click **Predict** to see results.
""")

# Sidebar Inputs
st.sidebar.header("🌿 Input Satellite Features")

latitude = st.sidebar.slider("Latitude", -90.0, 90.0, 0.0)
longitude = st.sidebar.slider("Longitude", -180.0, 180.0, 0.0)
ir_band_1 = st.sidebar.slider("IR Band 1", 0.0, 150.0, 36.5)
ir_band_2 = st.sidebar.slider("IR Band 2", 0.0, 200.0, 64.6)
ir_band_3 = st.sidebar.slider("IR Band 3", 0.0, 300.0, 97.9)
temperature = st.sidebar.slider("Temperature (K)", 200.0, 1200.0, 577.1)
vegetation_index = st.sidebar.slider("Vegetation Index", -1.0, 1.0, 0.0)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 20.0, 10.0)
wind_direction = st.sidebar.slider("Wind Direction (°)", 0.0, 360.0, 180.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 49.9)
elevation = st.sidebar.slider("Elevation (m)", 0.0, 4000.0, 1978.0)
slope = st.sidebar.slider("Slope (degrees)", 0.0, 90.0, 45.0)
land_use = st.sidebar.selectbox("Land Use Type", options=['urban', 'forest', 'farmland', 'desert', 'grassland'])
distance_to_urban = st.sidebar.slider("Distance to Urban (m)", 0.0, 50000.0, 25000.0)
cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, 50.0)

# Encode land_use
land_use_mapping = {'urban': 0, 'forest': 1, 'farmland': 2, 'desert': 3, 'grassland': 4}
land_use_encoded = land_use_mapping[land_use]

# Create input DataFrame
input_data = pd.DataFrame({
    'latitude': [latitude],
    'longitude': [longitude],
    'ir_band_1': [ir_band_1],
    'ir_band_2': [ir_band_2],
    'ir_band_3': [ir_band_3],
    'temperature': [temperature],
    'vegetation_index': [vegetation_index],
    'wind_speed': [wind_speed],
    'wind_direction': [wind_direction],
    'humidity': [humidity],
    'elevation': [elevation],
    'slope': [slope],
    'land_use': [land_use_encoded],
    'distance_to_urban': [distance_to_urban],
    'cloud_cover': [cloud_cover]
})

# Prediction
if st.button("🔍 Predict Wildfire"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🔥 Fire Detected! (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ No Fire Detected (Risk: {probability*100:.2f}%)")

# EDA Section
st.markdown("---")
st.subheader("📊 Correlation Heatmap of Features")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Developed by [Jeet Thumar](https://github.com/JeetThumar) • Powered by Streamlit")

