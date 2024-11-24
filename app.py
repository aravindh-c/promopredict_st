import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title('Promotion prediction app')

df=pd.read_csv('train.csv')
department= st.selectbox("Department", pd.unique(df["department"]))
region= st.selectbox("Region", pd.unique(df["region"]))
education= st.selectbox("Education", pd.unique(df["education"]))
gender= st.selectbox("Gender", pd.unique(df["gender"]))
recruitment_channel= st.selectbox("Recruitment_channel", pd.unique(df["recruitment_channel"]))
previous_year_rating= st.selectbox("Previous_year_rating", pd.unique(df["previous_year_rating"]))

no_of_trainings = st.number_input('No_of_Trainings')
age = st.number_input('Age')
length_of_service = st.number_input('length_of_service')
kpi_met_80 = st.number_input('kpi_met_80')
avg_training_score = st.number_input('avg_training_score')
awards_won = st.number_input('awards_won')

inputs={
   'department':department,
    'region':region,
    'education':education,
    'gender':gender,
    'recruitment_channel':recruitment_channel,
    'previous_year_rating':previous_year_rating,
    'no_of_trainings':no_of_trainings,
    'age':age,
    'length_of_service':length_of_service,
    'KPIs_met >80%':kpi_met_80,
    'awards_won?':awards_won,
    'avg_training_score':avg_training_score
}

model= joblib.load('promotion_pipeline_2311_model.pkl')

if st.button('Predict'):
    x_input=pd.DataFrame(inputs,index=[0])
    prediction=model.predict(x_input)
    st.write(' Predicted value is ::')
    st.write(prediction)






