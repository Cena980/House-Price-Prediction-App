import streamlit as st
import pickle
import numpy as np
import pandas as pd
# Streamlit UI
st.title("House price Prediction App")
st.write("Enter house details to predict price category.")
st.write("By Ali Sina Nazari")

# Input fields
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=1)
sqft_living = st.number_input("sqft_living (370-13540)", min_value=370, max_value=13540, value=370)
sqft_lot = st.number_input("sqft_lot (638-1074218)", min_value=638, max_value=1074218, value=56000)
floors = st.selectbox("floors", [1, 2, 3])
waterfront = st.selectbox("WaterFront", [0, 1])
view = st.selectbox("View", [0, 1, 2,3,4])
condition = st.selectbox("Condition", [0, 1, 2,3,4,5])
sqft_basement = st.number_input("sqft_basement (0-4820)", min_value=0, max_value=4820, value=0)
yr_built = st.number_input("year Built (1900-2014)", min_value=1900, max_value=2014, value=2000)
city = st.number_input("city (0-43)", min_value=0, max_value=43, value=12)


with open('knn_house_model.pkl','rb') as f:
    loaded_model=pickle.load(f)
    
with open('knn_house_scaler.pkl','rb') as f:
    loaded_scaler=pickle.load(f)

if st.button("Predict"):
    try:
        # Define column names as they were in training
        feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
                        'view', 'condition', 'sqft_basement', 'yr_built',
                        'city']
        # Convert input to a DataFrame with correct column names
        input_features = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
                                        view, condition, sqft_basement, yr_built,
                                        city]], 
                                       columns=feature_names)
        input_features = loaded_scaler.transform(input_features) 
        
        value=loaded_model.predict(input_features)
        if value == 0:
                st.success("✅ low prices")
        elif value == 1:
                st.success("⚠️ Medium Prices")
        elif value==2:
                st.error("⛔ Slightly High Prices")
        elif value==3:
                st.error("⛔ High Price")
    except Exception as e:
         st.error(f"Error: {e}")