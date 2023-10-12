import streamlit as st
import numpy as np

import pickle
import warnings
warnings.filterwarnings('ignore')

# loading saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Title of App
st.title("Crop Recommendation Web App")

# Create an expander for explanations
with st.expander("ℹ️ Explanations"):
    st.write("This web app recommends crops based on soil and environmental conditions.")
    st.write("Please enter the following information:")
    st.write("- Nitrogen (N)%, Phosphorus (P)%, and Potassium (K)%: These are key soil nutrient levels.")
    st.write("- Temperature: Current temperature in degrees Celsius.")
    st.write("- Humidity: Current humidity percentage.")
    st.write("- pH: Soil acidity or alkalinity level.")
    st.write("- Rainfall: Average annual rainfall in millimeters.")




# Getting Input from users
n = st.number_input('Enter the value of Nitrogen(N)%',  min_value=0.00, max_value=200.00)
p = st.number_input('Enter the value of Phosphorus(P)%', min_value=0.00, max_value=100.00)
k = st.number_input('Enter the value of Potassium(k)%', min_value=0.00, max_value=100.00)
temperature = st.number_input('Enter the value of temperature', min_value=0.00, max_value=50.00)
humidity = st.number_input('Enter the value of humidity', min_value=0.00, max_value=100.00)
ph = st.number_input('Enter the value of ph', min_value=2.00, max_value=10.00)
rainfall = st.number_input('Enter the value of rainfall', min_value=0.00, max_value=300.00)
features = [n, p, k, temperature, humidity, ph, rainfall]
feature_array = np.asarray(features)
feature_array2d = feature_array.reshape(1,-1)
crops = ['Apple', 'Banana', 'Blackgram', 'Chickpea', 'Coconut', 'Coffee',
       'Cotton', 'Grapes', 'Jute', 'Kidneybeans', 'Lentil', 'Maize',
       'Mango', 'Mothbeans', 'Mungbean', 'Muskmelon', 'Orange', 'Papaya',
       'Pigeonpeas', 'Pomegranate', 'Rice', 'Watermelon']

# labels = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

if st.button("Check the Result"):
    crop = loaded_model.predict(feature_array2d)
    st.write("The most suitable crop for given soil:", crops[crop[0]])



