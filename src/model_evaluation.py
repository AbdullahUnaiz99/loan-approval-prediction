"""
Model evaluation module for loan classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any, List, Tuple
from src.utils import load_config

class ModelEvaluator:
    """Model evaluation class for loan classification"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             title: str = "Confusion Matrix") -> go.Figure:
        """Create interactive confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       title=title,
                       labels=dict(x="Predicted", y="Actual"))
        
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            font=dict(size=12)
        )
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      title: str = "ROC Curve") -> go.Figure:
        """Create ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.2f})',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve") -> go.Figure:
        """Create Precision-Recall curve plot"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.2f})',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )
        
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                              title: str = "Feature Importance") -> go.Figure:
        """Create feature importance plot"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logging.warning("Model doesn't have feature importance or coefficients")
            return None
        
        # Sort features by importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title=title,
            labels={'importance': 'Importance', 'feature': 'Features'}
        )
        
        fig.update_layout(height=max(400, len(feature_names) * 20))
        
        return fig
    
    def evaluate_single_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                            feature_names: List[str], model_name: str) -> Dict[str, Any]:
        """Comprehensive evaluation of a single model"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create plots
        plots = {}
        
        # Confusion Matrix
        plots['confusion_matrix'] = self.plot_confusion_matrix(
            y_test, y_pred, f"Confusion Matrix - {model_name}"
        )
        
        # ROC Curve
        if y_pred_proba is not None:
            plots['roc_curve'] = self.plot_roc_curve(
                y_test, y_pred_proba, f"ROC Curve - {model_name}"
            )
            
            plots['pr_curve'] = self.plot_precision_recall_curve(
                y_test, y_pred_proba, f"Precision-Recall Curve - {model_name}"
            )
        
        # Feature Importance
        importance_plot = self.plot_feature_importance(
            model, feature_names, f"Feature Importance - {model_name}"
        )
        if importance_plot:
            plots['feature_importance'] = importance_plot
        
        return {
            'classification_report': class_report,
            'plots': plots,
            'predictions': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
    
    def compare_models(self, results: Dict[str, Any]) -> go.Figure:
        """Create model comparison plot"""
        comparison_data = []
        
        for technique_name, models in results.items():
            for model_name, model_data in models.items():
                metrics = model_data['test_metrics']
                comparison_data.append({
                    'Technique': technique_name,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'ROC-AUC': metrics.get('roc_auc', 0)
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Overall Comparison'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        
        for i, metric in enumerate(metrics):
            row, col = positions[i]
            
            for technique in df_comparison['Technique'].unique():
                technique_data = df_comparison[df_comparison['Technique'] == technique]
                
                fig.add_trace(
                    go.Bar(
                        x=technique_data['Model'],
                        y=technique_data[metric],
                        name=f"{technique}" if i == 0 else f"{technique}",
                        showlegend=True if i == 0 else False,
                        legendgroup=technique
                    ),
                    row=row, col=col
                )
        
        # Overall comparison (F1-Score)
        fig.add_trace(
            go.Bar(
                x=[f"{row['Model']} ({row['Technique']})" for _, row in df_comparison.iterrows()],
                y=df_comparison['F1-Score'],
                name='F1-Score',
                showlegend=False
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=800,
            title_text="Model Performance Comparison",
            showlegend=True
        )
        
        return fig
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        evaluation_results = {}
        
        for technique_name, models in results.items():
            technique_results = {}
            data = processed_data[technique_name]
            
            for model_name, model_data in models.items():
                model = model_data['model']
                
                # Comprehensive evaluation
                evaluation = self.evaluate_single_model(
                    model=model,
                    X_test=data['X_test'],
                    y_test=data['y_test'],
                    feature_names=data['feature_names'],
                    model_name=f"{model_name} ({technique_name})"
                )
                
                technique_results[model_name] = {
                    **model_data,
                    'detailed_evaluation': evaluation
                }
            
            evaluation_results[technique_name] = technique_results
        
        # Generate comparison plots
        comparison_plot = self.compare_models(results)
        
        return {
            'detailed_results': evaluation_results,
            'comparison_plot': comparison_plot
        }
    
    def create_performance_dashboard(self, results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create performance dashboard with multiple plots"""
        
        dashboard_plots = {}
        
        # Performance summary bar chart
        summary_data = []
        for technique_name, models in results.items():
            for model_name, model_data in models.items():
                metrics = model_data['test_metrics']
                summary_data.append({
                    'Model': f"{model_name}\n({technique_name})",
                    'F1-Score': metrics['f1'],
                    'Accuracy': metrics['accuracy'],
                    'ROC-AUC': metrics.get('roc_auc', 0)
                })
        
        summary_df = pd.DataFrame(summary_data).sort_values('F1-Score', ascending=True)
        
        # F1-Score comparison
        dashboard_plots['f1_comparison'] = px.bar(
            summary_df,
            x='F1-Score',
            y='Model',
            orientation='h',
            title='F1-Score Comparison Across Models',
            color='F1-Score',
            color_continuous_scale='viridis'
        )
        
        # Multi-metric comparison
        metrics_df = summary_df.melt(
            id_vars=['Model'],
            value_vars=['F1-Score', 'Accuracy', 'ROC-AUC'],
            var_name='Metric',
            value_name='Score'
        )
        
        dashboard_plots['multi_metric'] = px.bar(
            metrics_df,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Multi-Metric Performance Comparison'
        )
        
        dashboard_plots['multi_metric'].update_xaxes(tickangle=45)

        
        return dashboard_plots
