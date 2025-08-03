"""
Data insights page for Streamlit app
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def show_data_insights_page():
    """Display data insights page"""
    
    st.title("📈 Data Insights & Analysis")
    st.write("Explore the loan dataset and understand the patterns that influence loan approval decisions.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your loan dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Basic dataset info
            st.subheader("📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Display sample data
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data types and statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Types")
                dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                st.dataframe(dtype_df)
            
            with col2:
                st.subheader("📈 Statistical Summary")
                st.dataframe(df.describe())
            
            # Missing values analysis
            st.subheader("🔍 Missing Values Analysis")
            
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig_missing = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column",
                    labels={'x': 'Columns', 'y': 'Missing Count'}
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")
            
            # Target variable analysis
            if 'Loan_Status' in df.columns:
                st.subheader("🎯 Target Variable Analysis")
                
                target_counts = df['Loan_Status'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(
                        values=target_counts.values,
                        names=target_counts.index,
                        title="Loan Approval Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.write("**Class Distribution:**")
                    for label, count in target_counts.items():
                        percentage = (count / len(df)) * 100
                        st.write(f"- {label}: {count} ({percentage:.1f}%)")
                    
                    # Class imbalance ratio
                    imbalance_ratio = target_counts.min() / target_counts.max()
                    st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}")
                    
                    if imbalance_ratio < 0.5:
                        st.warning("⚠️ Dataset is imbalanced. Consider using balancing techniques.")
                    else:
                        st.success("✅ Dataset is relatively balanced.")
            
            # Numerical features analysis
            st.subheader("📊 Numerical Features Analysis")
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) > 0:
                selected_num_feature = st.selectbox("Select numerical feature to analyze", numerical_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = px.histogram(
                        df,
                        x=selected_num_feature,
                        nbins=30,
                        title=f"Distribution of {selected_num_feature}"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(
                        df,
                        y=selected_num_feature,
                        title=f"Box Plot of {selected_num_feature}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Feature statistics
                feature_stats = df[selected_num_feature].describe()
                st.write(f"**Statistics for {selected_num_feature}:**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{feature_stats['mean']:.2f}")
                with col2:
                    st.metric("Median", f"{feature_stats['50%']:.2f}")
                with col3:
                    st.metric("Std Dev", f"{feature_stats['std']:.2f}")
                with col4:
                    st.metric("Range", f"{feature_stats['max'] - feature_stats['min']:.2f}")
            
            # Categorical features analysis
            st.subheader("📋 Categorical Features Analysis")
            
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if len(categorical_cols) > 0:
                selected_cat_feature = st.selectbox("Select categorical feature to analyze", categorical_cols)
                
                value_counts = df[selected_cat_feature].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig_bar = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {selected_cat_feature}",
                        labels={'x': selected_cat_feature, 'y': 'Count'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig_pie_cat = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Proportion of {selected_cat_feature}"
                    )
                    st.plotly_chart(fig_pie_cat, use_container_width=True)
            
            # Correlation analysis
            if len(numerical_cols) > 1:
                st.subheader("🔗 Correlation Analysis")
                
                corr_matrix = df[numerical_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix of Numerical Features"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # High correlations
                st.write("**High Correlations (|r| > 0.5):**")
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            high_corr_pairs.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': corr_val
                            })
                
                if high_corr_pairs:
                    high_corr_df = pd.DataFrame(high_corr_pairs)
                    st.dataframe(high_corr_df, use_container_width=True)
                else:
                    st.info("No high correlations found between numerical features.")
            
            # Feature vs Target analysis
            if 'Loan_Status' in df.columns:
                st.subheader("🎯 Feature vs Target Analysis")
                
                analysis_type = st.selectbox("Select analysis type", 
                                           ["Numerical vs Target", "Categorical vs Target"])
                
                if analysis_type == "Numerical vs Target" and len(numerical_cols) > 0:
                    selected_feature = st.selectbox("Select numerical feature", numerical_cols)
                    
                    fig_violin = px.violin(
                        df,
                        x='Loan_Status',
                        y=selected_feature,
                        box=True,
                        title=f"{selected_feature} Distribution by Loan Status"
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
                    
                    # Statistical test
                    from scipy.stats import ttest_ind
                    
                    approved = df[df['Loan_Status'] == 'Y'][selected_feature].dropna()
                    not_approved = df[df['Loan_Status'] == 'N'][selected_feature].dropna()
                    
                    if len(approved) > 0 and len(not_approved) > 0:
                        t_stat, p_value = ttest_ind(approved, not_approved)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("T-statistic", f"{t_stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.success("✅ Significant difference between groups")
                        else:
                            st.info("ℹ️ No significant difference between groups")
                
                elif analysis_type == "Categorical vs Target" and len(categorical_cols) > 1:
                    cat_features = [col for col in categorical_cols if col != 'Loan_Status']
                    selected_feature = st.selectbox("Select categorical feature", cat_features)
                    
                    # Cross-tabulation
                    cross_tab = pd.crosstab(df[selected_feature], df['Loan_Status'], normalize='index') * 100
                    
                    fig_stacked = px.bar(
                        cross_tab,
                        title=f"Loan Approval Rate by {selected_feature}",
                        labels={'value': 'Percentage', 'index': selected_feature}
                    )
                    st.plotly_chart(fig_stacked, use_container_width=True)
                    
                    # Display cross-tabulation table
                    st.write("**Cross-tabulation (Percentages):**")
                    st.dataframe(cross_tab.round(2))
            
            # Outlier detection
            st.subheader("🚨 Outlier Detection")
            
            if len(numerical_cols) > 0:
                selected_outlier_feature = st.selectbox("Select feature for outlier analysis", numerical_cols)
                
                # Calculate IQR
                Q1 = df[selected_outlier_feature].quantile(0.25)
                Q3 = df[selected_outlier_feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[selected_outlier_feature] < lower_bound) | 
                             (df[selected_outlier_feature] > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Outliers", len(outliers))
                with col2:
                    st.metric("Outlier Percentage", f"{len(outliers)/len(df)*100:.2f}%")
                with col3:
                    st.metric("IQR", f"{IQR:.2f}")
                
                if len(outliers) > 0:
                    st.write("**Sample Outliers:**")
                    st.dataframe(outliers.head(), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    else:
        # Show sample insights with dummy data
        st.info("Upload a CSV file to see detailed data insights, or check out the sample analysis below.")
        
        # Sample data creation for demonstration
        st.subheader("📊 Sample Data Analysis")
        
        # Create sample loan data
        np.random.seed(42)
        sample_data = {
            'ApplicantIncome': np.random.normal(5000, 2000, 1000),
            'LoanAmount': np.random.normal(150, 50, 1000),
            'Credit_History': np.random.choice([0, 1], 1000, p=[0.2, 0.8]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], 1000),
            'Loan_Status': np.random.choice(['Y', 'N'], 1000, p=[0.7, 0.3])
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Sample visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sample_hist = px.histogram(
                sample_df,
                x='ApplicantIncome',
                title="Sample: Applicant Income Distribution"
            )
            st.plotly_chart(fig_sample_hist, use_container_width=True)
        
        with col2:
            fig_sample_pie = px.pie(
                values=sample_df['Loan_Status'].value_counts().values,
                names=sample_df['Loan_Status'].value_counts().index,
                title="Sample: Loan Approval Distribution"
            )
            st.plotly_chart(fig_sample_pie, use_container_width=True)

if __name__ == "__main__":
    show_data_insights_page()

