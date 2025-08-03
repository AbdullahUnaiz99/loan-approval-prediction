# 📦 Loan Application Classification Project

A comprehensive machine learning application to classify loan applications as **Approved** or **Declined**.  
Built with multiple ML algorithms, advanced imbalanced data handling techniques, and an interactive Streamlit web interface.

---

## 🚀 Project Overview

This project implements a **production-ready ML pipeline** for loan approval prediction, featuring:
- **Multiple ML Algorithms**: Logistic Regression, KNN, Random Forest, XGBoost
- **Imbalanced Data Handling**: SMOTE, Random Undersampling, and combined techniques
- **Robust Evaluation**: Cross-validation and multiple metrics
- **Interactive Web App**: Built using Streamlit
- **Modular Codebase**: Clean, documented, and configurable

---

## 📊 Dataset

|                        | Count     |
|------------------------|---------:|
| Total Records          | ~50,000  |
| Approved               | 40,882   |
| Declined               | 8,092    |
| Imbalance Ratio        | ~5:1     |
| Features               | 14 (9 numerical, 5 categorical) |

### ➤ Numerical Features
- `loan_tenure_months`
- `interest_rate_offered`
- `monthly_income`
- `cibil_score`
- `existing_emis_monthly`
- `debt_to_income_ratio`
- `applicant_age`
- `number_of_dependents`
- `loan_amount_requested`

### ➤ Categorical Features
- `loan_type` — Business Loan, Car Loan, Education Loan, Personal Loan, Home Loan
- `purpose_of_loan` — Medical Emergency, Education, Debt Consolidation, etc.
- `employment_status` — Retired, Unemployed, Self-Employed, Salaried, etc.
- `property_ownership_status` — Rented, Owned, Jointly Owned
- `gender` — Female, Other, Male

---

## 🏗️ Project Structure

```plaintext
loan_classification_app/
├── data/
│   ├── raw/                    # Raw data
│   │   └── loan_data.csv
│   └── processed/              # Processed train/test splits
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── src/                        # Core ML pipeline
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── models/                     # Saved trained models
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
├── streamlit_app/              # Streamlit web app
│   ├── app.py
│   ├── pages/
│   │   ├── prediction.py
│   │   ├── model_comparison.py
│   │   └── data_insights.py
│   └── utils/
│       ├── preprocessing.py
│       └── model_loader.py
├── config/                     # Config files
│   └── config.yaml
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── setup.py                    # Package setup

---
## 🚀 Quick Start

### ➤ Run the full machine learning pipeline (preprocessing + training + evaluation)
- python src/main.py --stage all --data data/raw/loan_application.csv

### ➤ Launch the Streamlit web application
- streamlit run streamlit_app/app.py