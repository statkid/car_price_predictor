import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

# Reopening Files
df = pkl.load(open('df.pkl','rb'))
pipe = pkl.load(open('pipe.pkl','rb'))

# Title and Subheading
st.title('Car Price Predictor')
st.write('Get the Best Price for your Car')

# Taking Inputs from Our User:
company = st.selectbox('Company Name :',df['company'].unique())
model = st.selectbox('Model Name :', df['name'].unique())
year = st.number_input('Enter Year of purchase:')
fuel_type = st.selectbox('Fuel Type :',df['fuel_type'].unique())
kms_driven = st.number_input('Enter Total Kilometers Driven')

if st.button('Predict Price'):

    st.title(
    int(
    pipe.predict(
    pd.DataFrame(
    columns=['name','company','year','kms_driven','fuel_type'],data=np.array([model,company,year,kms_driven,fuel_type]).reshape(1,5)))[0]))



