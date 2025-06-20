#!/usr/bin/env python3
"""
Test model training functionality
"""
import sys
import os
sys.path.append('Model')

from model_training import ModelTrainer


def test_model_training():
    """Test if model training completes successfully"""
    print('🔄 Testing model training...')
    
    trainer = ModelTrainer()
    
    # Get dataset path
    dataset_path = 'Data/01_Raw/oral_cancer_prediction_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f'❌ Dataset not found at {dataset_path}')
        return False
    
    # Train and save model
    success = trainer.train_and_evaluate(dataset_path)
    if success:
        print('✅ Model trained successfully for testing')
        return True
    else:
        print('❌ Model training failed')
        return False


if __name__ == "__main__":
    success = test_model_training()
    if not success:
        sys.exit(1) 