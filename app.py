import streamlit as st
import pickle
import numpy as np


loaded_model = pickle.load(open('heart_disease_model.sav','rb'))

def heart_disease(input_data):
    inp_data_arr = np.asarray(input_data)
    inp_data_reshaped = inp_data_arr.reshape(1,-1)

    prediction = loaded_model.predict(inp_data_reshaped)

    if prediction[0] == 0:
        return 'The person does not have heart disease'
    else:
        return 'The person has heart disease'

st.title("Heart Disease Prediction Web App")

col1,col2,col3 = st.columns(3)

with col1:
    Age = st.text_input('Age')
with col2:
    Sex = st.text_input('Sex')
with col3:
    Chestpain = st.text_input('Chest Pain Type')
with col1:
    BP = st.text_input('BP')
with col2:
    Cholesterol = st.text_input('Cholesterol')
with col3:
    FBS = st.text_input('FBS over 120')
with col1:
    EKG = st.text_input('EKG results')
with col2:
    MaxHR = st.text_input('Max HR')
with col3:
    Exercise_angina = st.text_input('Exercise angina')
with col1:
    Depression_ST = st.text_input('ST depression')
with col2:
    Slope_ST = st.text_input('Slope of ST')
with col3:
    vessels_fluro = st.text_input('Number of vessels fluro')
with col1:
    Thallium = st.text_input('Thallium')

diagnosis = ''

if st.button('Heart Disease Test Results'):
    diagnosis = heart_disease([int(Age), int(Sex), int(Chestpain), int(BP), int(Cholesterol), int(FBS), int(EKG), int(MaxHR), int(Exercise_angina), float(Depression_ST), int(Slope_ST), int(vessels_fluro), int(Thallium)])

st.success(diagnosis)