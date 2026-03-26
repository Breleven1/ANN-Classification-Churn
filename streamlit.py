import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#loading the trained model
model = tf.keras.models.load_model('model.h5')

#loading encoders and scalers
with open('label_encoder_gender.pkl','rb') as file:
    loaded_label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    loaded_onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#streamlit app
st.title("Customer Churn Prediction")

#User inputs
#geography = st.selectbox('Geography', loaded_onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', loaded_label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')

#prepare the input_data
input_data = pd.DataFrame({
    'Gender': [loaded_label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Balance':[balance]
})

#onehot encoding geography column
geo_encoded = loaded_onehot_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = loaded_onehot_encoder_geo.get_feature_names_out(['Geography']))

#Combine one-hot encode columns with input data
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#predict churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write("Customer likely to churn")
else:
    st.write("customer unlikely to churn")