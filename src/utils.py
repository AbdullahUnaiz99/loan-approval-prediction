"""
Utility functions for the loan classification project
"""

import os
import logging
import yaml
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('loan_classification.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def create_directories(paths: list) -> None:
    """Create directories if they don't exist"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory ensured: {path}")

def save_model(model, filepath: str) -> None:
    """Save model to file"""
    try:
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(filepath: str):
    """Load model from file"""
    try:
        model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to file"""
    try:
        if filepath.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif filepath.endswith('.csv'):
            pd.DataFrame(results).to_csv(filepath, index=False)
        logging.info(f"Results saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(model, feature_names: list, title: str = "Feature Importance") -> None:
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate memory usage of DataFrame"""
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    return {
        'total_memory_mb': round(memory_usage, 2),
        'rows': len(df),
        'columns': len(df.columns)
    }

