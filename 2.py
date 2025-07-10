import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load models
model1 = pickle.load(open('best_model_non5.pkl', 'rb'))
model2 = pickle.load(open('best_model_int7.9.pkl', 'rb'))

# Load Scaler object (only once)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title='Evaluating and forecasting undergraduate dropouts',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Page title and extra information
st.markdown("""
    <h1 style='color: darkblue; text-align: center; font-size: 36px;'>Warning forecasts of domestic and international student dropouts</h1>
    <div style='text-align: center; font-size: 18px;'>Developed by Dr. Songbo Wang et al., Hubei University of Technology.</div>
    <div style='text-align: center; font-size: 18px;'>Email: Wangsongbo@hbut.edu.cn</div>
""", unsafe_allow_html=True)

# Button to choose Domestic or International students
student_type = st.sidebar.radio("Select student type", ('Domestic', 'International'))

st.sidebar.subheader('Input features for single prediction')
# Use columns to divide into two parts for left (Domestic) and right (International)
col1, col2 = st.columns(2)

# Initialize feature variables
Marital = Mode = Order = Course = Attendance = Qualification = Mother_Q = Father_Q = Mother_O = Father_O = Displaced = 0
Need = Debtor = Fee = Gender = Scholarship = Age = First = Second = Unemployment = Inflation = GDP = 0

# Domestic student input sliders
if student_type == 'Domestic':
    with col1:
        Marital = st.slider('Marital status', 1, 4, 1)
        Mode = st.slider('Application order', 1, 18, 9)
        Order = st.slider("Application order", 1, 5, 2)
        Course = st.slider('Course type', 1, 17, 10)
        Attendance = st.slider('Daytime/evening attendance', 0, 1, 0)
        Qualification = st.slider("Previous qualification", 1, 14, 7)
        Mother_Q = st.slider("Mother qualification", 1, 28, 14)
        Father_Q = st.slider('Father qualification', 1, 28, 15)
        Mother_O = st.slider('Mother occupation', 1, 25, 18)
        Father_O = st.slider("Father occupation", 1, 26, 20)
        Displaced = st.slider('Displaced', 0, 1, 0)

    with col2:
        Need = st.slider('Educational special need', 0, 1, 0)
        Debtor = st.slider("Debtor", 0, 1, 0)
        Fee = st.slider("Tuition fee", 0, 1, 0)
        Gender = st.slider("Gender", 0, 1, 0)
        Scholarship = st.slider('Scholarship', 0, 1, 0)
        Age = st.slider("Age", 18, 59, 23)
        First = st.slider("1st semester approved course", 0, 18, 9)
        Second = st.slider("2nd semester approved course", 0, 12, 4)
        Unemployment = st.slider("Unemployment rate", 7.6, 16.2, 11.0)
        Inflation = st.slider("Inflation rate", -0.8, 3.7, 1.0)
        GDP = st.slider("GDP", -4.06, 3.51, 1.00)

# International student input sliders
elif student_type == 'International':
    with col1:
        Marital = st.slider('Marital status', 1, 4, 1)
        Mode = st.slider('Application order', 1, 18, 9)
        Order = st.slider("Application order", 1, 5, 2)
        Course = st.slider('Course type', 1, 17, 10)
        Attendance = st.slider('Daytime/evening attendance', 0, 1, 0)
        Qualification = st.slider("Previous qualification", 1, 14, 7)
        Mother_Q = st.slider("Mother qualification", 1, 28, 14)
        Father_Q = st.slider('Father qualification', 1, 28, 15)
        Mother_O = st.slider('Mother occupation', 1, 25, 18)
        Father_O = st.slider("Father occupation", 1, 26, 20)
        Displaced = st.slider('Displaced', 0, 1, 0)

    with col2:
        Need = st.slider('Educational special need', 0, 1, 0)
        Debtor = st.slider("Debtor", 0, 1, 0)
        Fee = st.slider("Tuition fee", 0, 1, 0)
        Gender = st.slider("Gender", 0, 1, 0)
        Scholarship = st.slider('Scholarship', 0, 1, 0)
        Age = st.slider("Age", 18, 59, 23)
        First = st.slider("1st semester approved course", 0, 18, 9)
        Second = st.slider("2nd semester approved course", 0, 12, 4)
        Unemployment = st.slider("Unemployment rate", 7.6, 16.2, 11.0)
        Inflation = st.slider("Inflation rate", -0.8, 3.7, 1.0)
        GDP = st.slider("GDP", -4.06, 3.51, 1.00)

