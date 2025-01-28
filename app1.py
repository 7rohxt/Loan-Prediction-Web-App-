import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\rk\OneDrive\Desktop\Loan Prediction\dataset\loan_train.csv"
df = pd.read_csv(file_path)

# Preprocessing
le_Gender = LabelEncoder()
le_Married = LabelEncoder()
le_Dependents = LabelEncoder()
le_Education = LabelEncoder()
le_Self_Employed = LabelEncoder()
le_Area = LabelEncoder()

df['Gender_n'] = le_Gender.fit_transform(df['Gender'])
df['Married_n'] = le_Married.fit_transform(df['Married'])
df['Dependents_n'] = le_Dependents.fit_transform(df['Dependents'])
df['Education_n'] = le_Education.fit_transform(df['Education'])
df['Self_Employed_n'] = le_Self_Employed.fit_transform(df['Self_Employed'])
df['Area_n'] = le_Area.fit_transform(df['Area'])

inputs_n = df.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Area', 'Status'], axis='columns')
target = df['Status']

X_train, X_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2, random_state=42)

# Model training
model = tree.DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Streamlit app
st.title("Loan Prediction Web Application")
st.sidebar.header("User Parameters")

# Function to get user input
def get_user_input():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    married = st.sidebar.selectbox("Marital Status", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Number of Dependents", ("0", "1", "2", "3+"))
    education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.sidebar.selectbox("Self-Employed", ("Yes", "No"))
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
    term = st.sidebar.number_input("Loan Term (in months)", min_value=1)
    credit_history = st.sidebar.selectbox("Credit History", (1.0, 0.0))
    area = st.sidebar.selectbox("Area", ("Urban", "Semiurban", "Rural"))
    
    user_input = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Loan_Amount': loan_amount,
        'Term': term,
        'Credit_History': credit_history,
        'Area': area
    }
    return user_input

# Function to encode user input
def encode_user_input(user_input, label_encoders):
    encoded_input = {}
    for col, value in user_input.items():
        if col in label_encoders:
            encoded_input[col + '_n'] = label_encoders[col].transform([value])[0]
        else:
            encoded_input[col] = value
    return encoded_input

user_input = get_user_input()
encoded_input = encode_user_input(user_input, {
    'Gender': le_Gender,
    'Married': le_Married,
    'Dependents': le_Dependents,
    'Education': le_Education,
    'Self_Employed': le_Self_Employed,
    'Area': le_Area,
})

user_df = pd.DataFrame([encoded_input])

model_feature_columns = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Term', 
                         'Credit_History', 'Gender_n', 'Married_n', 'Dependents_n', 
                         'Education_n', 'Self_Employed_n', 'Area_n']

user_df = user_df[model_feature_columns]

if st.sidebar.button("Predict Loan Status"):
    loan_status = model.predict(user_df)
    if loan_status[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Denied")

# Visualizations
st.header("Data Visualizations")

if st.checkbox("Show Gender Distribution by Loan Status"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Gender', hue='Status', ax=ax)
    ax.set_title("Loan Status by Gender", fontsize=16)
    ax.set_xlabel("Gender", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    st.pyplot(fig)

if st.checkbox("Show Credit History Distribution"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Credit_History', hue='Status', ax=ax)
    ax.set_title("Loan Status by Credit History", fontsize=16)
    st.pyplot(fig)

if st.checkbox("Show Dependents Distribution"):
    fig, ax = plt.subplots(figsize=(8, 8))
    dependents_dist = df['Dependents'].value_counts()
    ax.pie(dependents_dist, labels=dependents_dist.index, autopct='%1.1f%%', startangle=140)
    ax.set_title("Distribution of Loan Applicants by Dependents", fontsize=16)
    st.pyplot(fig)

if st.checkbox("Show Correlation Heatmap"):
    continuous_vars = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Credit_History']
    corr_matrix = df[continuous_vars].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix Heatmap", fontsize=16)
    st.pyplot(fig)
