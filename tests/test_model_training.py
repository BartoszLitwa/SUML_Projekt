#!/usr/bin/env python3
"""
Test model training functionality
"""
import sys
import os
sys.path.append('Model')

from model_training import ModelTrainer
from data_preprocessing import DataPreprocessor


def test_model_training():
    """Test if model training completes successfully"""
    print('🔄 Testing model training...')
    
    trainer = ModelTrainer()
    preprocessor = DataPreprocessor()
    
    # Get dataset path
    dataset_path = 'Data/01_Raw/oral_cancer_prediction_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f'❌ Dataset not found at {dataset_path}')
        return False
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset_path)
    if X_train is None:
        print('❌ Failed to prepare data')
        return False
    
    # Train models
    best_model, best_model_name = trainer.train_models(X_train, X_test, y_train, y_test)
    if best_model is None:
        print('❌ Model training failed')
        return False
    
    # Save model and preprocessor
    model_path = trainer.save_model(best_model, best_model_name)
    preprocessor.save_preprocessor()
    print(f'✅ Model trained and saved to {model_path}')
    return True


if __name__ == "__main__":
    success = test_model_training()
    if not success:
        sys.exit(1) 