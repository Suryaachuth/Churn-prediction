import pickle 
import pandas as pd
import streamlit as st
from tensorflow import keras
loaded_model = keras.models.load_model('model.h5')
label_model = pickle.load(open('label.pkl', 'rb'))
min_max=pickle.load(open("min_max.pkl","rb"))

st.title('Customer Churn Prediction')
age = st.slider('Age', min_value=18, max_value=100, step=1)
gender = st.radio('Gender', ['Male', 'Female'])
location = st.selectbox('Location', ['Location 1', 'Location 2', 'Location 3', 'Location 4'])
subscription_length_months = st.slider('Subscription Length (Months)', min_value=0, max_value=24, step=1)
monthly_bill = st.number_input('Monthly Bill', min_value=0.0)
total_usage_gb = st.number_input('Total Usage (GB)', min_value=0.0)

gender_encoded = label_model.transform([gender])[0]
location_encoded = label_model.transform([location])[0]
user_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_encoded],
    'Location': [location_encoded],
    'Subscription_Length_Months': [subscription_length_months],
    'Monthly_Bill': [monthly_bill],
    'Total_Usage_GB': [total_usage_gb]
})

user_data[['Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']] = min_max.transform(user_data[['Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']])
if st.button('Predict Churn'):
    # Make a prediction
    prediction = model.predict(user_data)

    if prediction[0] == 0:
        st.write("This customer is not likely to churn.")
    else:
        st.write("This customer is likely to churn.")
