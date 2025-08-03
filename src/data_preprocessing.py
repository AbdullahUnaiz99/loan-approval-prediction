import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import logging
from src.utils import load_config

class DataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_config(config_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

        # Define your categorical and numerical features exactly
        self.categorical_cols = [
            'loan_type', 'purpose_of_loan', 'employment_status',
            'property_ownership_status', 'gender'
        ]
        self.numerical_cols = [
            'loan_tenure_months', 'interest_rate_offered', 'monthly_income',
            'cibil_score', 'existing_emis_monthly', 'debt_to_income_ratio',
            'applicant_age', 'number_of_dependents', 'loan_amount_requested'
        ]

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        logging.info(f"Raw data shape: {df.shape}")

        # Filter only Approved and Declined for binary classification
        df = df[df['loan_status'].isin(['Approved', 'Declined'])].copy()
        df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Declined': 0})

        logging.info(f"Filtered binary data shape: {df.shape}")
        return df

    def handle_outliers(self, df):
        if not self.config['preprocessing']['handle_outliers']:
            return df

        df_processed = df.copy()
        for col in self.numerical_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

        logging.info("Outliers handled with IQR")
        return df_processed

    def encode_categorical(self, df):
        df_processed = df.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        return df_processed

    def split_and_scale(self, df):
        target_col = self.config['data']['target_column']
        X = df[self.numerical_cols + self.categorical_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['preprocessing']['test_size'],
            random_state=self.config['preprocessing']['random_state'],
            stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        self.feature_names = X.columns.tolist()

        return X_train_scaled, X_test_scaled, y_train, y_test

    def resample_data(self, X_train, y_train, technique):
        if technique == 'none':
            return X_train, y_train

        sampler = None
        if technique == 'smote':
            sampler = SMOTE(random_state=self.config['preprocessing']['random_state'])
        elif technique == 'random_undersampling':
            sampler = RandomUnderSampler(random_state=self.config['preprocessing']['random_state'])
        elif technique == 'smote_enn':
            sampler = SMOTEENN(random_state=self.config['preprocessing']['random_state'])
        else:
            logging.warning(f"Unknown resampling technique: {technique}")
            return X_train, y_train

        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)

    def preprocess_pipeline(self, filepath):
        df = self.load_data(filepath)
        df = self.handle_outliers(df)
        df = self.encode_categorical(df)
        X_train, X_test, y_train, y_test = self.split_and_scale(df)

        processed_data = {}
        for technique in self.config['imbalanced_techniques']:
            X_res, y_res = self.resample_data(X_train, y_train, technique)
            processed_data[technique] = {
                'X_train': X_res,
                'X_test': X_test,
                'y_train': y_res,
                'y_test': y_test,
                'feature_names': self.feature_names
            }

        # Save processed data for training stage
        os.makedirs(self.config['data']['processed_path'], exist_ok=True)
        import joblib
        joblib.dump(processed_data, os.path.join(self.config['data']['processed_path'], 'processed_data.joblib'))
        joblib.dump(self, os.path.join(self.config['data']['processed_path'], 'preprocessor.joblib'))

        logging.info("Preprocessing pipeline completed")
        return processed_data

    def transform_new_data(self, df_new):
        df_processed = df_new.copy()
        # Encode categorical columns based on fitted LabelEncoders
        for col, le in self.label_encoders.items():
            df_processed[col] = le.transform(df_processed[col])

        # Scale numerical columns
        if self.scaler is not None:
            df_processed = pd.DataFrame(
                self.scaler.transform(df_processed),
                columns=df_processed.columns,
                index=df_processed.index
            )
        return df_processed
