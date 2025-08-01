Loan Application Classification Project
A comprehensive machine learning application for classifying loan applications as Approved or Declined using multiple ML algorithms and advanced imbalanced data handling techniques.

🚀 Project Overview
This project implements a complete machine learning pipeline for loan approval prediction, featuring:

Multiple ML Algorithms: Logistic Regression, KNN, Random Forest, XGBoost

Imbalanced Data Handling: SMOTE, Random Undersampling, Combined techniques

Comprehensive Evaluation: Multiple metrics and cross-validation

Interactive Web Application: Streamlit-based user interface

Production-Ready Code: Modular, well-documented, and configurable

📊 Dataset Information
Total Records: ~50,000 loan applications

Features: 14 features (9 numerical, 5 categorical)

Target Variable: Loan_Status (Approved: 40,882, Declined: 8,092)

Imbalance Ratio: ~5:1 (Approved:Declined)

Features
Numerical Features (9):

loan_tenure_months

interest_rate_offered

monthly_income

cibil_score

existing_emis_monthly

debt_to_income_ratio

applicant_age

number_of_dependents

loan_amount_requested

Categorical Features (5):

loan_type: Business Loan, Car Loan, Education Loan, Personal Loan, Home Loan

purpose_of_loan: Medical Emergency, Education, Debt Consolidation, etc.

employment_status: Retired, Unemployed, Self-Employed, Salaried, etc.

property_ownership_status: Rented, Owned, Jointly Owned

gender: Female, Other, Male

🏗️ Project Structure

loan_classification_app/
├── data/
│   ├── raw/                    # Raw data files
│   │   └── loan_data.csv
│   └── processed/              # Processed data files
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data preprocessing pipeline
│   ├── model_training.py       # Model training pipeline
│   ├── model_evaluation.py     # Model evaluation utilities
│   └── utils.py               # Utility functions
├── models/                     # Trained model files
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── best_model.pkl
├── notebooks/                  # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── streamlit_app/             # Streamlit application
│   ├── app.py                 # Main application
│   ├── pages/                 # Application pages
│   │   ├── prediction.py
│   │   ├── model_comparison.py
│   │   └── data_insights.py
│   └── utils/                 # App utilities
│       ├── preprocessing.py
│       └── model_loader.py
├── config/                    # Configuration files
│   └── config.yaml
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── setup.py                  # Package setup
