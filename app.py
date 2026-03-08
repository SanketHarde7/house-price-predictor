import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏡 California House Price Predictor")
st.write("This app predicts the median house value in California based on various features.")

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'rf_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("Model not found! Please run `python src/train.py` first to train the model.")
else:
    st.sidebar.header("Input Features")
    
    # Feature inputs based on California Housing dataset
    med_inc = st.sidebar.number_input("Median Income (in $10,000s)", value=3.5, min_value=0.0)
    house_age = st.sidebar.number_input("House Age (years)", value=28.0, min_value=1.0)
    ave_rooms = st.sidebar.number_input("Average Rooms per Household", value=5.0, min_value=1.0)
    ave_bedrms = st.sidebar.number_input("Average Bedrooms per Household", value=1.0, min_value=0.1)
    population = st.sidebar.number_input("Population", value=1000.0, min_value=1.0)
    ave_occup = st.sidebar.number_input("Average Occupancy", value=3.0, min_value=1.0)
    latitude = st.sidebar.number_input("Latitude", value=35.0)
    longitude = st.sidebar.number_input("Longitude", value=-119.0)
    
    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])
    
    st.subheader("Your Input Data:")
    st.dataframe(input_data)
    
    if st.button("Predict Price", type="primary"):
        prediction = model.predict(input_data)[0]
        # The target in California Housing is in 100,000s of dollars
        predicted_price = prediction * 100000
        
        st.success(f"### Predicted House Value: ${predicted_price:,.2f}")
