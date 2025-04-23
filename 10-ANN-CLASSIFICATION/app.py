import streamlit as st
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('/Users/adityakumarsingh/python/10-ANN-CLASSIFICATION/model.h5')

# Load the encoders and scaler
with open('/Users/adityakumarsingh/python/10-ANN-CLASSIFICATION/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('/Users/adityakumarsingh/python/10-ANN-CLASSIFICATION/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('/Users/adityakumarsingh/python/10-ANN-CLASSIFICATION/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app

st.title("🔍 ChurnGuard AI — " \
"Predict Bank Customer Churn")

st.markdown("Use this form to enter customer details and predict the likelihood of churn.")

# Group fields into columns for better layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("🌍 Select Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", min_value=18, max_value=92, value=30)
    credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650)
    tenure = st.slider("📆 Tenure (Years with Bank)", 0, 10, value=3)

with col2:
    balance = st.number_input("🏦 Account Balance", min_value=0.0, value=50000.0)
    estimated_salary = st.number_input("💰 Estimated Salary", min_value=0.0, value=60000.0)
    num_of_products = st.slider("📦 Number of Products", 1, 4, value=1)
    has_cr_card = st.radio("💳 Has Credit Card?", [0, 1])
    is_active_member = st.radio("🔄 Is Active Member?", [0, 1])

# Optional: map Yes/No to 1/0
#has_cr_card_val = 1 if has_cr_card == "Yes" else 0
#is_active_member_val = 1 if is_active_member == "Yes" else 0

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
# Scale the input data
input_data_scaled = scaler.transform(input_data)
# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
st.write(f'Churn Probability: {prediction_proba:.2f}')
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
