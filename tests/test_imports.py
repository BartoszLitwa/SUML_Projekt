#!/usr/bin/env python3
"""
Test that all required imports work correctly
"""
import sys
sys.path.append('Model')


def test_imports():
    """Test that all required modules can be imported"""
    print('🔍 Testing imports...')
    
    try:
        from data_preprocessing import DataPreprocessor
        print('✅ DataPreprocessor import successful')
    except ImportError as e:
        print(f'❌ DataPreprocessor import error: {e}')
        return False
    
    try:
        from model_training import ModelTrainer
        print('✅ ModelTrainer import successful')
    except ImportError as e:
        print(f'❌ ModelTrainer import error: {e}')
        return False
    
    try:
        from model_inference import OralCancerPredictor
        print('✅ OralCancerPredictor import successful')
    except ImportError as e:
        print(f'❌ OralCancerPredictor import error: {e}')
        return False
    
    print('✅ All imports successful')
    return True


if __name__ == "__main__":
    success = test_imports()
    if not success:
        sys.exit(1) 