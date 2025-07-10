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

st.sidebar.subheader('Input features for single prediction')
st.sidebar.markdown('<h2 style="font-size: 24px;">Select student type</h2>', unsafe_allow_html=True)
# Radio button for selecting student type
student_type = st.sidebar.radio("", ('Domestic', 'International'))

# Use columns to divide into two parts
col1, col2 = st.sidebar.columns(2)

with col1:
    # Nationality slider based on the student type
    if student_type == 'Domestic':
        Nationality = 1
    elif student_type == 'International':
        Nationality = st.slider("Nationality", 2, 21, 10)
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

if st.sidebar.button('Predict'):
    # Input features for prediction
    input_features = [[Marital, Mode, Order, Course, Attendance, Qualification, Nationality,
                       Mother_Q, Father_Q, Mother_O, Father_O, Displaced, Need, Debtor,
                       Fee, Gender, Scholarship, Age, First, Second, Unemployment, Inflation, GDP]]
    
    # Create DataFrame
    new_data = pd.DataFrame(input_features, columns=['Marital', 'Mode', 'Order', 'Course', 'Attendance', 'Qualification',
                                                     'Nationality', 'Mother_Q', 'Father_Q', 'Mother_O', 'Father_O',
                                                     'Displaced', 'Need', 'Debtor', 'Fee', 'Gender', 'Scholarship',
                                                     'Age', 'First', 'Second', 'Unemployment', 'Inflation', 'GDP'])

    # Standardize the input features
    feature_values_scaled = scaler.transform(new_data.values)

try:
    col1, col2, col3 = st.columns([5, 1, 5])  # Left column for Domestic, right column for International

    # Domestic Student Prediction (Left Column)
    with col1:
        prediction = model1.predict(feature_values_scaled)
        st.markdown("<h1 style='color: blue; font-size: 30px;'>Domestic students result:</h1>", unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.markdown(f"<h2 style='font-size: 36px; color: green;'> Dropout！</h2>", unsafe_allow_html=True)
        elif prediction[0] == 0:
            st.markdown(f"<h2 style='font-size: 36px; color: green;'>Graduate！</h2>", unsafe_allow_html=True)

        # Display features and values for Domestic students
        rows = []
        for i in range(0, len(new_data.columns), 2):
            feature_name_1 = new_data.columns[i]
            value_1 = new_data.iloc[0, i]
            feature_name_2 = new_data.columns[i+1] if i+1 < len(new_data.columns) else ''
            value_2 = new_data.iloc[0, i+1] if i+1 < len(new_data.columns) else ''
            row = [f'{feature_name_1}: {value_1}', f'{feature_name_2}: {value_2}']
            rows.append(row)

        rows_df = pd.DataFrame(rows, columns=['Line 1', 'Line 2'])
        st.dataframe(rows_df, use_container_width=True)

    with col3:
        # Check if Nationality is within the valid range for international students
        if 2 <= Nationality <= 21:
            # Use model2 for International students
            prediction = model2.predict(feature_values_scaled)
            st.markdown("<h1 style='color: blue; font-size: 30px;'>International students result:</h1>", unsafe_allow_html=True)

            if prediction[0] == 1:
                st.markdown(f"<h2 style='font-size: 36px; color: green;'>Dropout！</h2>", unsafe_allow_html=True)
            elif prediction[0] == 0:
                st.markdown(f"<h2 style='font-size: 36px; color: green;'>Graduate！</h2>", unsafe_allow_html=True)

            # Display features and values for International students
            rows = []
            for i in range(0, len(new_data.columns), 2):
                feature_name_1 = new_data.columns[i]
                value_1 = new_data.iloc[0, i]
                feature_name_2 = new_data.columns[i+1] if i+1 < len(new_data.columns) else ''
                value_2 = new_data.iloc[0, i+1] if i+1 < len(new_data.columns) else ''
                row = [f'{feature_name_1}: {value_1}', f'{feature_name_2}: {value_2}']
                rows.append(row)

            rows_df = pd.DataFrame(rows, columns=['Line 1', 'Line 2'])
            st.dataframe(rows_df, use_container_width=True)

        else:
            # Adjust the Nationality to 10 if it doesn't meet the criteria for international
            Nationality = 10
            # Create the new data again with the modified Nationality
            input_features = [[Marital, Mode, Order, Course, Attendance, Qualification, Nationality,
                               Mother_Q, Father_Q, Mother_O, Father_O, Displaced, Need, Debtor,
                               Fee, Gender, Scholarship, Age, First, Second, Unemployment, Inflation, GDP]]

            # Create DataFrame
            new_data = pd.DataFrame(input_features, columns=['Marital', 'Mode', 'Order', 'Course', 'Attendance', 'Qualification',
                                                             'Nationality', 'Mother_Q', 'Father_Q', 'Mother_O', 'Father_O',
                                                             'Displaced', 'Need', 'Debtor', 'Fee', 'Gender', 'Scholarship',
                                                             'Age', 'First', 'Second', 'Unemployment', 'Inflation', 'GDP'])

            # Standardize the input features
            feature_values_scaled = scaler.transform(new_data.values)
            
            # Perform prediction for the modified Nationality (if needed)
            prediction = model1.predict(feature_values_scaled)
            st.markdown("<h1 style='color: blue; font-size: 30px;'>Domestic students result:</h1>", unsafe_allow_html=True)

            if prediction[0] == 1:
                st.markdown(f"<h2 style='font-size: 36px; color: green;'>Dropout！</h2>", unsafe_allow_html=True)
            elif prediction[0] == 0:
                st.markdown(f"<h2 style='font-size: 36px; color: green;'>Graduate！</h2>", unsafe_allow_html=True)

            # Display features and values for Domestic students
            rows = []
            for i in range(0, len(new_data.columns), 2):
                feature_name_1 = new_data.columns[i]
                value_1 = new_data.iloc[0, i]
                feature_name_2 = new_data.columns[i+1] if i+1 < len(new_data.columns) else ''
                value_2 = new_data.iloc[0, i+1] if i+1 < len(new_data.columns) else ''
                row = [f'{feature_name_1}: {value_1}', f'{feature_name_2}: {value_2}']
                rows.append(row)

            rows_df = pd.DataFrame(rows, columns=['Line 1', 'Line 2'])
            st.dataframe(rows_df, use_container_width=True)

except ValueError as e:
    st.error(f"输入值无效: {e}")
except Exception as e:
    st.error(f"发生错误: {e}")
