#!/usr/bin/env python3
"""
Debug script to test model training with realistic dataset
"""

import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append('.')

try:
    # Test imports
    print("Testing imports...")
    from Model.data_preprocessing import DataPreprocessor
    from Model.model_training import ModelTrainer
    print("✓ Imports successful")
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    dataset_path = 'Data/01_Raw/test_realistic_dataset.csv'
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"✓ Dataset loaded: {df.shape}")
        print(f"✓ Target distribution:\n{df['Oral Cancer (Diagnosis)'].value_counts()}")
        
        # Test preprocessing
        print("\nTesting preprocessing...")
        preprocessor = DataPreprocessor()
        
        # Simple test - just load and clean
        df_clean = preprocessor.clean_data(df)
        print(f"✓ Data cleaned: {df_clean.shape}")
        
        # Test feature engineering
        df_encoded = preprocessor.encode_features(df_clean)
        print(f"✓ Data encoded: {df_encoded.shape}")
        
        df_enhanced = preprocessor.create_feature_interactions(df_encoded)
        print(f"✓ Features enhanced: {df_enhanced.shape}")
        
        # Test feature selection
        X, y = preprocessor.select_features(df_enhanced)
        print(f"✓ Features selected: X={X.shape}, y={len(y)}")
        print(f"✓ Target unique values: {y.unique()}")
        print(f"✓ Target counts: {y.value_counts()}")
        
        print("\n🎉 All preprocessing tests passed!")
        
    else:
        print(f"❌ Dataset not found at: {dataset_path}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug script completed.") 