#header files
import os
import pandas as pd
import numpy as np

file_path =r"C:\Users\rk\OneDrive\Desktop\Loan Prediction\dataset\loan_train.csv"

df = pd.read_csv(file_path)
(df.head())
inputs = df.drop('Status',axis='columns')
target = df['Status']
from sklearn.preprocessing import LabelEncoder
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
inputs.head()

inputs_n = inputs.drop(['Gender','Married','Dependents','Education','Self_Employed','Area'],axis='columns')
inputs_n

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2, random_state=42)

df.dropna(inplace=True)

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Training Accuracy:", round(train_score,2))
print("Testing Accuracy:",round(test_score,2))


def get_user_input():
    gender = input("Enter Gender (Male/Female): ")
    married = input("Enter Marital Status (Yes/No): ")
    dependents = input("Enter Number of Dependents: ")
    education = input("Enter Education (Graduate/Not Graduate): ")
    self_employed = input("Enter Self Employment Status (Yes/No): ")
    applicant_income = float(input("Enter Applicant Income: "))
    coapplicant_income = float(input("Enter Coapplicant Income: "))
    loan_amount = float(input("Enter Loan Amount: "))
    term = int(input("Enter Loan Term (in months): "))  # Term comes after Coapplicant Income
    credit_history = float(input("Enter Credit History (1.0 or 0.0): "))
    area = input("Enter Area (Urban/Rural): ")

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


def encode_user_input(user_input, label_encoders):
    encoded_input = {}
    for col, value in user_input.items():
        if col in label_encoders:
            encoded_input[col + '_n'] = label_encoders[col].transform([value])[0]
        else:
            encoded_input[col] = value
    return encoded_input

# Get user input
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

loan_status = model.predict(user_df)

if loan_status[0] == 1:
    print("Loan Approved")
else:
    print("Loan Denied")

#visualizations

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.countplot(data=df, x='Gender', hue='Status')

plt.title('Loan Status by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Loan Status', loc='upper right')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Credit_History', hue='Status')

plt.figure(figsize=(8, 8))
Dependents_n = df['Dependents'].value_counts()
plt.pie(Dependents_n, labels=Dependents_n.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Loan Applicants with respect to Number of Dependents', fontsize=16)
plt.show()

continuous_vars = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Credit_History']
corr_matrix = df[continuous_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True,cmap='coolwarm', fmt='.2f', cbar=True, vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show() 