"""
Main execution script for loan classification project
"""

import argparse
import logging
import os
from src.utils import setup_logging, load_config, create_directories
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

def main():
    """Main execution function"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Loan Classification Pipeline')
    parser.add_argument('--stage', choices=['preprocess', 'train', 'evaluate', 'all'], 
                       default='all', help='Pipeline stage to run')
    parser.add_argument('--data', type=str, required=True, help='Path to input data file')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.info("Starting Loan Classification Pipeline")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    create_directories([
        config['data']['processed_path'],
        config['model_storage']['path'],
        'logs'
    ])
    
    try:
        if args.stage in ['preprocess', 'all']:
            logging.info("=== Data Preprocessing Stage ===")
            preprocessor = DataPreprocessor(args.config)
            processed_data = preprocessor.preprocess_pipeline(args.data)
            logging.info("Data preprocessing completed successfully")
        
        if args.stage in ['train', 'all']:
            logging.info("=== Model Training Stage ===")
            
            # Load processed data if not already loaded
            if args.stage == 'train':
                import joblib
                processed_data = joblib.load(f"{config['data']['processed_path']}/processed_data.joblib")
            
            trainer = ModelTrainer(args.config)
            results = trainer.train_all_models(processed_data)
            
            # Save best model
            trainer.save_best_model()
            
            # Generate performance summary
            summary_df = trainer.generate_performance_summary()
            print("\n=== Performance Summary ===")
            print(summary_df.to_string(index=False))
            
            # Save results
            import joblib
            joblib.dump(results, f"{config['model_storage']['path']}/training_results.joblib")
            summary_df.to_csv(f"{config['model_storage']['path']}/performance_summary.csv", index=False)
            
            logging.info("Model training completed successfully")
        
        if args.stage in ['evaluate', 'all']:
            logging.info("=== Model Evaluation Stage ===")
            
            # Load results if not already loaded
            if args.stage == 'evaluate':
                import joblib
                results = joblib.load(f"{config['model_storage']['path']}/training_results.joblib")
                processed_data = joblib.load(f"{config['data']['processed_path']}/processed_data.joblib")
            
            evaluator = ModelEvaluator(args.config)
            evaluation_results = evaluator.generate_evaluation_report(results, processed_data)
            
            # Create dashboard plots
            dashboard_plots = evaluator.create_performance_dashboard(results)
            
            # Save evaluation results
            import joblib
            joblib.dump(evaluation_results, f"{config['model_storage']['path']}/evaluation_results.joblib")
            
            logging.info("Model evaluation completed successfully")
        
        logging.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()

