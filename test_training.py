#!/usr/bin/env python3
"""
Test script to verify model training works
"""

import sys
import os
sys.path.append('.')

def test_model_training():
    try:
        print("🧪 Testing Model Training with Fixed Dataset...")
        
        # Import modules
        from Model.data_preprocessing import DataPreprocessor
        from Model.model_training import ModelTrainer
        
        # Initialize
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer()
        
        # Test dataset path
        dataset_path = 'Data/01_Raw/test_realistic_dataset.csv'
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            return False
        
        print(f"✅ Dataset found: {dataset_path}")
        
        # Test data preparation
        print("\n📊 Testing data preparation...")
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset_path)
        
        if X_train is None:
            print("❌ Data preparation failed")
            return False
            
        print(f"✅ Data prepared successfully:")
        print(f"   Training: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        print(f"   Target train unique: {y_train.unique()}")
        print(f"   Target test unique: {y_test.unique()}")
        
        # Test model training
        print("\n🤖 Testing model training...")
        best_model, best_model_name = trainer.train_models(X_train, X_test, y_train, y_test)
        
        if best_model is None:
            print("❌ Model training failed")
            return False
            
        print(f"✅ Model training successful!")
        print(f"   Best model: {best_model_name}")
        print(f"   Best score: {trainer.best_score:.4f}")
        
        # Test evaluation
        print("\n📈 Testing model evaluation...")
        evaluation_results = trainer.evaluate_model(best_model, X_test, y_test, best_model_name)
        
        print(f"✅ Model evaluation successful!")
        print(f"   Final ROC AUC: {evaluation_results['roc_auc']:.4f}")
        print(f"   Final Accuracy: {evaluation_results['accuracy']:.4f}")
        
        # Test feature importance
        print("\n🔍 Testing feature importance...")
        feature_importance = trainer.get_feature_importance(best_model, preprocessor.feature_columns)
        
        if feature_importance is not None:
            print("✅ Feature importance extracted successfully!")
            print(f"   Top feature: {feature_importance.iloc[0]['feature']}")
        
        # Test model saving
        print("\n💾 Testing model saving...")
        model_path = trainer.save_model(best_model, best_model_name)
        preprocessor.save_preprocessor()
        
        print(f"✅ Model saved successfully to: {model_path}")
        
        print("\n🎉 ALL TESTS PASSED! Model training is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_training()
    sys.exit(0 if success else 1) 