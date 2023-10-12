import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
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
temperature = st.number_input('Enter the of temperature')
humidity = st.number_input('Enter the of humidity')
ph = st.number_input('Enter the value of ph')
rainfall = st.number_input('Enter the value of rainfall')
features = [n, p, k, temperature, humidity, ph, rainfall]
feature_array = np.asarray(features)
feature_array2d = feature_array.reshape(1,-1)
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform([features])
# crop = loaded_model.predict(feature_array2d)
result = ''
if st.button("Check the Result"):
    crop = loaded_model.predict(feature_array2d)
    st.write(crop)


