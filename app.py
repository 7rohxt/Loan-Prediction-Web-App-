import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

# Set Streamlit page configuration
st.set_page_config(page_title="Loan Data Analysis", layout="wide")

st.title("Loan Data Analysis and Visualization")

file_path = r"dataset\loan_train.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocessing
inputs = df.drop('Status', axis='columns')
target = df['Status']

# Label encoding
le_Gender = LabelEncoder()
le_Married = LabelEncoder()
le_Dependents = LabelEncoder()
le_Education = LabelEncoder()
le_Self_Employed = LabelEncoder()
le_Area = LabelEncoder()

inputs['Gender_n'] = le_Gender.fit_transform(inputs['Gender'])
inputs['Married_n'] = le_Married.fit_transform(inputs['Married'])
inputs['Dependents_n'] = le_Dependents.fit_transform(inputs['Dependents'])
inputs['Education_n'] = le_Education.fit_transform(inputs['Education'])
inputs['Self_Employed_n'] = le_Self_Employed.fit_transform(inputs['Self_Employed'])
inputs['Area_n'] = le_Area.fit_transform(inputs['Area'])

inputs_n = inputs.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Area'], axis='columns')

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2, random_state=42)

# Drop missing values
df.dropna(inplace=True)

# Train Decision Tree Classifier
model = tree.DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Accuracy scores
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

st.subheader("Model Accuracy")
st.write(f"Training Accuracy: **{round(train_score, 2)}**")
st.write(f"Testing Accuracy: **{round(test_score, 2)}**")

from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Function to get input from the user
def get_user_input():
    # Get user input for each feature
    gender = input("Enter Gender (Male/Female): ")
    married = input("Enter Marital Status (Yes/No): ")
    dependents = input("Enter Number of Dependents: ")
    education = input("Enter Education (Graduate/Not Graduate): ")
    self_employed = input("Enter Self Employment Status (Yes/No): ")
    area = input("Enter Area (Urban/Rural): ")
    applicant_income = float(input("Enter Applicant Income: "))
    coapplicant_income = float(input("Enter Coapplicant Income: "))
    loan_amount = float(input("Enter Loan Amount: "))
    term = int(input("Enter Loan Term (in months): "))  # Term comes after Coapplicant Income
    credit_history = float(input("Enter Credit History (1.0 or 0.0): "))

    # Create a dictionary for the user input
    user_input = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'Area': area,
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Loan_Amount': loan_amount,
        'Term': term,  # Term comes after Coapplicant Income
        'Credit_History': credit_history
    }
    return user_input

# Encode user input based on the fitted LabelEncoders
def encode_user_input(user_input, label_encoders):
    encoded_input = {}
    for col, value in user_input.items():
        if col in label_encoders:
            try:
                # Normalize input to match training categories
                if isinstance(value, str):
                    value = value.strip().capitalize()  # Handle case sensitivity
                # Transform the value
                encoded_input[col + '_n'] = label_encoders[col].transform([value])[0]
            except ValueError:
                # Handle unseen labels gracefully
                st.error(f"Invalid value for {col}: {value}. Please provide a valid input.")
                return None  # Stop processing
        else:
            # For numeric inputs, add them directly
            encoded_input[col] = value
    return encoded_input

# Get user input
user_input = get_user_input()

# Encode the categorical user input
encoded_input = encode_user_input(user_input, {
    'Gender': le_Gender,
    'Married': le_Married,
    'Dependents': le_Dependents,
    'Education': le_Education,
    'Self_Employed': le_Self_Employed,
    'Area': le_Area,
})

# Convert to DataFrame to match the input format
user_df = pd.DataFrame([encoded_input])

# Reorder the user input DataFrame to match the training column order
model_feature_columns = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Term', 
                         'Credit_History', 'Gender_n', 'Married_n', 'Dependents_n', 
                         'Education_n', 'Self_Employed_n', 'Area_n']

# Reorder the user input DataFrame to match the training feature order
user_df = user_df[model_feature_columns]

# Predict loan status using the trained model
loan_status = model.predict(user_df)

# Output the prediction to the user
if loan_status[0] == 1:
    print("Loan Approved")
else:
    print("Loan Denied")
# Visualizations
st.subheader("Visualizations")

# Countplot for Loan Status by Gender
st.write("### Loan Status by Gender")
fig, ax = plt.subplots(figsize=(4, 3))
sns.countplot(data=df, x='Gender', hue='Status', ax=ax)
plt.title('Loan Status by Gender', fontsize=12)
plt.xlabel('Gender', fontsize=6)
plt.ylabel('Count', fontsize=6)
plt.legend(title='Loan Status', loc='upper right')
st.pyplot(fig)

# Countplot for Credit History
st.write("### Loan Status by Credit History")
fig, ax = plt.subplots(figsize=(4, 3))
sns.countplot(data=df, x='Credit_History', hue='Status', ax=ax)
plt.title('Loan Status by Credit History', fontsize=12)
plt.xlabel('Credit History', fontsize=6)
plt.ylabel('Count', fontsize=6)
plt.legend(title='Loan Status', loc='upper right')
st.pyplot(fig)

# Pie chart for Distribution of Dependents
st.write("### Distribution of Loan Applicants by Number of Dependents")
fig, ax = plt.subplots(figsize=(4, 4))
dependents_counts = df['Dependents'].value_counts()
ax.pie(dependents_counts, labels=dependents_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Loan Applicants by Number of Dependents', fontsize=12)
st.pyplot(fig)

# Heatmap for correlation matrix
st.write("### Correlation Matrix Heatmap")
continuous_vars = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Credit_History']
corr_matrix = df[continuous_vars].corr()

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, vmin=-1, vmax=1, ax=ax)
plt.title('Correlation Matrix Heatmap', fontsize=12)
st.pyplot(fig)
