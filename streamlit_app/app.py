"""
Main Streamlit application for loan classification
"""

import streamlit as st
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from streamlit_app.pages.prediction import show_prediction_page
from streamlit_app.pages.model_comparison import show_model_comparison_page
from streamlit_app.pages.data_insights import show_data_insights_page
from streamlit_app.utils.model_loader import load_saved_model, get_model_info

# Page configuration
st.set_page_config(
    page_title="Loan Classification System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}

.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Sidebar navigation
    st.sidebar.title("🏦 Loan Classification System")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "🏠 Home": "home",
        "🔮 Prediction": "prediction", 
        "📊 Model Comparison": "comparison",
        "📈 Data Insights": "insights"
    }
    
    selected_page = st.sidebar.selectbox("Navigate to:", list(pages.keys()))
    
    # Main content based on selection
    if pages[selected_page] == "home":
        show_home_page()
    elif pages[selected_page] == "prediction":
        show_prediction_page()
    elif pages[selected_page] == "comparison":
        show_model_comparison_page()
    elif pages[selected_page] == "insights":
        show_data_insights_page()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This application provides an end-to-end loan classification system with:
    
    - **Instant Predictions**: Get loan approval predictions
    - **Model Comparison**: Compare different ML algorithms
    - **Data Analysis**: Explore dataset insights
    - **Interactive Interface**: User-friendly web interface
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ using Streamlit")

def show_home_page():
    """Display home page"""
    
    # Main header
    st.markdown('<h1 class="main-header">🏦 Loan Classification System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Loan Classification System** - a comprehensive machine learning solution for 
    predicting loan approvals. This system uses advanced ML algorithms and handles imbalanced data 
    to provide accurate and reliable loan approval predictions.
    """)
    
    # Key features
    st.markdown("## 🚀 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔮 Instant Predictions
        - Real-time loan approval predictions
        - Interactive input forms
        - Confidence scores and risk analysis
        - User-friendly interface
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Advanced ML Models
        - Multiple algorithm comparison
        - Imbalanced data handling
        - Hyperparameter optimization
        - Cross-validation
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Comprehensive Analysis
        - Model performance comparison
        - Data exploration tools
        - Interactive visualizations
        - Statistical insights
        """)
    
    # System overview
    st.markdown("## 📋 System Overview")
    
    # Load model info if available
    model, metadata = load_saved_model()
    
    if model is not None and metadata is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🏆 Current Best Model")
            st.success(get_model_info(metadata))
        
        with col2:
            st.markdown("### 📈 Quick Stats")
            
            # Performance metrics
            metrics = metadata['metrics']
            
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            st.metric("F1-Score", f"{metrics['f1']:.1%}")
            st.metric("Precision", f"{metrics['precision']:.1%}")
            
            if 'roc_auc' in metrics:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    else:
        st.warning("""
        ⚠️ **No trained model found!**
        
        Please run the training pipeline first:
        ```
        python src/main.py --stage all --data data/raw/your_data.csv
        ```
        """)
    
    # How to use
    st.markdown("## 🔧 How to Use")
    
    st.markdown("""
    1. **📊 Data Insights**: Upload your dataset to explore patterns and statistics
    2. **🔮 Prediction**: Enter loan application details to get instant approval predictions  
    3. **📈 Model Comparison**: Compare performance of different ML models and techniques
    
    Navigate using the sidebar menu to access different features of the system.
    """)
    
    # Technical details
    with st.expander("🔬 Technical Details"):
        st.markdown("""
        ### Machine Learning Pipeline
        
        **Data Preprocessing:**
        - Missing value imputation (median for numerical, mode for categorical)
        - Outlier detection and capping using IQR method
        - Feature encoding (Label encoding for categorical variables)
        - Feature scaling (StandardScaler)
        
        **Imbalanced Data Handling:**
        - SMOTE (Synthetic Minority Oversampling Technique)
        - Random Undersampling
        - SMOTE + Edited Nearest Neighbours (SMOTE-ENN)
        
        **Machine Learning Models:**
        - Logistic Regression with L1/L2 regularization
        - K-Nearest Neighbors with distance weighting
        - Random Forest with ensemble learning
        - XGBoost with gradient boosting
        
        **Model Evaluation:**
        - 5-fold stratified cross-validation
        - Multiple metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
        - Confusion matrices and ROC curves
        - Feature importance analysis
        """)
    
    # Quick start guide
    st.markdown("## ⚡ Quick Start")
    
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline
python src/main.py --stage all --data data/raw/loan_data.csv

# 3. Launch the web application
streamlit run streamlit_app/app.py
    """, language="bash")

if __name__ == "__main__":
    main()
