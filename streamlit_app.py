import streamlit as st
import numpy as np

import pickle
import warnings
warnings.filterwarnings('ignore')

# loading saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Title of App
st.title("Crop Recommendation Web App")

# Getting Input from users
n = st.number_input('Enter the value of Nitrogen(N)%')
p = st.number_input('Enter the value of Phosphorus(P)%')
k = st.number_input('Enter the value of Potassium(k)%')
temperature = st.number_input('Enter the value of temperature')
humidity = st.number_input('Enter the value of humidity')
ph = st.number_input('Enter the value of ph')
rainfall = st.number_input('Enter the value of rainfall')
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



