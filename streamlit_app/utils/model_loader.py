"""
Model loading utilities for Streamlit app
"""

import streamlit as st
import joblib
import json
from typing import Dict, Any, Tuple

@st.cache_resource
def load_saved_model():
    """Load the best saved model"""
    try:
        # Load model
        model = joblib.load("models/best_model.joblib")
        
        # Load metadata
        with open("models/best_model_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except FileNotFoundError:
        st.error("Model not found. Please run the training pipeline first.")
        return None, None

@st.cache_data
def load_training_results():
    """Load training results"""
    try:
        results = joblib.load("models/training_results.joblib")
        return results
    except FileNotFoundError:
        st.warning("Training results not found.")
        return None

@st.cache_data
def load_evaluation_results():
    """Load evaluation results"""
    try:
        results = joblib.load("models/evaluation_results.joblib")
        return results
    except FileNotFoundError:
        st.warning("Evaluation results not found.")
        return None

@st.cache_data
def load_performance_summary():
    """Load performance summary"""
    try:
        import pandas as pd
        summary_df = pd.read_csv("models/performance_summary.csv")
        return summary_df
    except FileNotFoundError:
        st.warning("Performance summary not found.")
        return None

def get_model_info(metadata: Dict[str, Any]) -> str:
    """Get formatted model information"""
    if metadata is None:
        return "Model information not available"
    
    info = f"""
    **Best Model:** {metadata['model_name']}
    **Technique:** {metadata['technique']}
    **Test Accuracy:** {metadata['metrics']['accuracy']:.4f}
    **Test F1-Score:** {metadata['metrics']['f1']:.4f}
    **Test Precision:** {metadata['metrics']['precision']:.4f}
    **Test Recall:** {metadata['metrics']['recall']:.4f}
    """
    
    if 'roc_auc' in metadata['metrics']:
        info += f"\n**Test ROC-AUC:** {metadata['metrics']['roc_auc']:.4f}"
    
    return info
