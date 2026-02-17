Heart Disease Prediction using Machine Learning
Project Overview
This project predicts the presence of heart disease using Machine Learning techniques.
The model is built using Logistic Regression and achieves an accuracy of 84.7%.

The goal of this project is to analyze patient medical data and classify whether a person has heart disease or not.

Dataset Information
The dataset used is the UCI Heart Disease Dataset.

Features include:
Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Resting ECG
Maximum Heart Rate
Exercise Induced Angina
ST Depression
Slope
Number of Major Vessels
Thalassemia
Target (Heart Disease Presence)
Target Values:

0 → No Heart Disease
1 → Heart Disease
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
VS Code
Machine Learning Process
1️ Data Preprocessing
Converted categorical variables using One-Hot Encoding
Handled missing values using Mean Imputation
Converted target variable into binary classification
2️ Model Used
Logistic Regression
3️ Train-Test Split
80% Training Data
20% Testing Data
Model Performance
Accuracy: 84.7%

Confusion Matrix:
Predicted 0	Predicted 1
Actual 0	62	13
Actual 1	15	94
The model correctly classified most patients with strong predictive performance.

How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/yourusername/heart-disease-prediction.git
