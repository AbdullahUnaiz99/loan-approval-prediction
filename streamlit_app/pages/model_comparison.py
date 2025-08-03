"""
Model comparison page for Streamlit app
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_app.utils.model_loader import load_performance_summary, load_training_results, load_evaluation_results

def show_model_comparison_page():
    """Display model comparison page"""
    
    st.title("📊 Model Performance Comparison")
    st.write("Compare the performance of different models and techniques used in the loan classification project.")
    
    # Load data
    summary_df = load_performance_summary()
    training_results = load_training_results()
    evaluation_results = load_evaluation_results()
    
    if summary_df is None:
        st.error("Performance data not found. Please run the training pipeline first.")
        return
    
    # Performance Summary Table
    st.subheader("📋 Performance Summary")
    
    # Highlight best performing models
    styled_df = summary_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Best Model Highlight
    best_model = summary_df.loc[summary_df['F1-Score'].idxmax()]
    
    st.success(f"""
    🏆 **Best Performing Model**
    - **Model:** {best_model['Model']} with {best_model['Technique']}
    - **F1-Score:** {best_model['F1-Score']:.4f}
    - **Accuracy:** {best_model['Accuracy']:.4f}
    - **ROC-AUC:** {best_model['ROC-AUC']:.4f}
    """)
    
    # Visualization Options
    st.subheader("📈 Performance Visualizations")
    
    # Metric selection
    col1, col2 = st.columns(2)
    with col1:
        selected_metric = st.selectbox("Select Metric for Comparison", 
                                     ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC'])
    with col2:
        chart_type = st.selectbox("Chart Type", ['Bar Chart', 'Horizontal Bar', 'Scatter Plot'])
    
    # Create visualizations
    if chart_type == 'Bar Chart':
        fig = px.bar(summary_df.sort_values(selected_metric, ascending=True),
                    x='Model',
                    y=selected_metric,
                    color='Technique',
                    title=f'{selected_metric} Comparison Across Models',
                    text=selected_metric)
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        
    elif chart_type == 'Horizontal Bar':
        fig = px.bar(summary_df.sort_values(selected_metric),
                    x=selected_metric,
                    y='Model',
                    color='Technique',
                    orientation='h',
                    title=f'{selected_metric} Comparison Across Models',
                    text=selected_metric)
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        
    else:  # Scatter Plot
        fig = px.scatter(summary_df,
                        x='F1-Score',
                        y=selected_metric if selected_metric != 'F1-Score' else 'Accuracy',
                        color='Technique',
                        size='ROC-AUC',
                        hover_data=['Model'],
                        title=f'{selected_metric} vs F1-Score')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-metric comparison
    st.subheader("🎯 Multi-Metric Comparison")
    
    # Radar chart for top 5 models
    top_models = summary_df.nlargest(5, 'F1-Score')
    
    fig_radar = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for idx, row in top_models.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=f"{row['Model']} ({row['Technique']})"
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Top 5 Models - Multi-Metric Comparison"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Technique Comparison
    st.subheader("🔄 Imbalancing Technique Analysis")
    
    # Group by technique
    technique_avg = summary_df.groupby('Technique')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].mean()
    
    fig_technique = px.bar(technique_avg.reset_index().melt(id_vars='Technique', var_name='Metric', value_name='Score'),
                          x='Technique',
                          y='Score',
                          color='Metric',
                          barmode='group',
                          title='Average Performance by Imbalancing Technique')
    
    st.plotly_chart(fig_technique, use_container_width=True)
    
    # Model Family Analysis
    st.subheader("🏗️ Model Architecture Analysis")
    
    # Extract model family
    summary_df['Model_Family'] = summary_df['Model'].str.split('_').str[0]
    model_family_avg = summary_df.groupby('Model_Family')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].mean()
    
    fig_family = px.line(model_family_avg.reset_index().melt(id_vars='Model_Family', var_name='Metric', value_name='Score'),
                        x='Model_Family',
                        y='Score',
                        color='Metric',
                        markers=True,
                        title='Performance by Model Architecture')
    
    st.plotly_chart(fig_family, use_container_width=True)
    
    # Statistical Summary
    st.subheader("📊 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Overall Statistics**")
        stats_df = summary_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].describe()
        st.dataframe(stats_df)
    
    with col2:
        st.write("**Best Performance per Metric**")
        best_performers = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            best_idx = summary_df[metric].idxmax()
            best_performers[metric] = f"{summary_df.loc[best_idx, 'Model']} ({summary_df.loc[best_idx, 'Technique']})"
        
        best_df = pd.DataFrame(list(best_performers.items()), columns=['Metric', 'Best Model'])
        st.dataframe(best_df, use_container_width=True)

if __name__ == "__main__":
    show_model_comparison_page()

