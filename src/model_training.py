"""
Model training module for loan classification
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
from typing import Dict, Any, Tuple, List
from src.utils import load_config, save_model

class ModelTrainer:
    """Model training class for loan classification"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.models = {}
        self.best_models = {}
        self.results = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with their parameter grids"""
        models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': self.config['models']['logistic_regression']['params']
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': self.config['models']['knn']['params']
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': self.config['models']['random_forest']['params']
            },
            'xgboost': {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': self.config['models']['xgboost']['params']
            }
        }
        
        logging.info("Models initialized successfully")
        return models
    
    def train_single_model(self, model_name: str, model_config: Dict[str, Any], 
                          X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """Train a single model with hyperparameter tuning"""
        logging.info(f"Training {model_name}...")
        
        # Setup GridSearchCV
        grid_search = GridSearchCV(
            estimator=model_config['model'],
            param_grid=model_config['params'],
            cv=self.config['evaluation']['cv_folds'],
            scoring=self.config['evaluation']['scoring'],
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logging.info(f"{model_name} training completed. Best CV score: {best_score:.4f}")
        logging.info(f"Best parameters: {best_params}")
        
        results = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'grid_search': grid_search
        }
        
        return best_model, results
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test set"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        return metrics
    
    def train_all_models(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train all models on all imbalanced data techniques"""
        models_config = self.initialize_models()
        all_results = {}
        
        for technique_name, data in processed_data.items():
            logging.info(f"\n=== Training models with technique: {technique_name} ===")
            
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']
            
            technique_results = {}
            
            for model_name, model_config in models_config.items():
                try:
                    # Train model
                    best_model, training_results = self.train_single_model(
                        model_name, model_config, X_train, y_train
                    )
                    
                    # Evaluate model
                    test_metrics = self.evaluate_model(best_model, X_test, y_test)
                    
                    # Store results
                    technique_results[model_name] = {
                        'model': best_model,
                        'training_results': training_results,
                        'test_metrics': test_metrics
                    }
                    
                    logging.info(f"{model_name} - Test F1: {test_metrics['f1']:.4f}")
                    
                except Exception as e:
                    logging.error(f"Error training {model_name} with {technique_name}: {e}")
                    continue
            
            all_results[technique_name] = technique_results
        
        self.results = all_results
        return all_results
    
    def find_best_model(self) -> Tuple[str, str, Any, Dict[str, float]]:
        """Find the best model across all techniques"""
        best_score = 0
        best_model = None
        best_technique = None
        best_model_name = None
        best_metrics = None
        
        for technique_name, models in self.results.items():
            for model_name, model_data in models.items():
                f1_score = model_data['test_metrics']['f1']
                
                if f1_score > best_score:
                    best_score = f1_score
                    best_model = model_data['model']
                    best_technique = technique_name
                    best_model_name = model_name
                    best_metrics = model_data['test_metrics']
        
        logging.info(f"Best model: {best_model_name} with {best_technique} (F1: {best_score:.4f})")
        return best_technique, best_model_name, best_model, best_metrics
    
    def save_best_model(self) -> None:
        """Save the best model to disk"""
        technique, model_name, model, metrics = self.find_best_model()
        
        model_path = self.config['model_storage']['path']
        import os
        os.makedirs(model_path, exist_ok=True)
        
        # Save best model
        best_model_path = f"{model_path}/{self.config['model_storage']['best_model_name']}"
        save_model(model, best_model_path)
        
        # Save model metadata
        metadata = {
            'technique': technique,
            'model_name': model_name,
            'metrics': metrics,
            'model_path': best_model_path
        }
        
        import json
        with open(f"{model_path}/best_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logging.info(f"Best model saved: {model_name} with {technique}")
    
    def generate_performance_summary(self) -> pd.DataFrame:
        """Generate performance summary DataFrame"""
        summary_data = []
        
        for technique_name, models in self.results.items():
            for model_name, model_data in models.items():
                metrics = model_data['test_metrics']
                summary_data.append({
                    'Technique': technique_name,
                    'Model': model_name,
                    'Accuracy': round(metrics['accuracy'], 4),
                    'Precision': round(metrics['precision'], 4),
                    'Recall': round(metrics['recall'], 4),
                    'F1-Score': round(metrics['f1'], 4),
                    'ROC-AUC': round(metrics.get('roc_auc', 0), 4)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('F1-Score', ascending=False)
        
        return summary_df
    
    def cross_validate_best_model(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform cross-validation on the best model"""
        technique, model_name, best_model, _ = self.find_best_model()
        
        # Get data for the best technique
        data = processed_data[technique]
        X_train = data['X_train']
        y_train = data['y_train']
        
        # Perform cross-validation
        cv_scores = {}
        for metric in self.config['evaluation']['metrics']:
            if metric == 'roc_auc' and len(np.unique(y_train)) == 2:
                scores = cross_val_score(best_model, X_train, y_train, 
                                       cv=self.config['evaluation']['cv_folds'], 
                                       scoring=metric)
            else:
                scores = cross_val_score(best_model, X_train, y_train, 
                                       cv=self.config['evaluation']['cv_folds'], 
                                       scoring=metric)
            
            cv_scores[f'{metric}_mean'] = scores.mean()
            cv_scores[f'{metric}_std'] = scores.std()
        
        logging.info("Cross-validation completed for best model")
        return cv_scores
