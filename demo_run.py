#!/usr/bin/env python3
"""
Simple demo script for the Oral Cancer Risk Prediction System
"""

import os
import sys
import pandas as pd
import numpy as np

# Add Model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

def create_mock_dataset():
    """Create a mock dataset for demonstration purposes"""
    print("Creating mock dataset for demonstration...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create mock data
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Tobacco Use': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'Alcohol Consumption': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'HPV Infection': np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9]),
        'Betel Quid Use': np.random.choice(['Yes', 'No'], n_samples, p=[0.05, 0.95]),
        'Chronic Sun Exposure': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'Poor Oral Hygiene': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'Diet (Fruits & Vegetables Intake)': np.random.choice(['High', 'Moderate', 'Low'], n_samples),
        'Family History of Cancer': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'Compromised Immune System': np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9]),
        'Oral Lesions': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'Unexplained Bleeding': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'Difficulty Swallowing': np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9]),
        'White or Red Patches in Mouth': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
    }
    
    # Create target variable with some logic
    # Higher risk for certain combinations
    risk_factors = (
        (data['Tobacco Use'] == 'Yes').astype(int) * 3 +
        (data['Alcohol Consumption'] == 'Yes').astype(int) * 2 +
        (data['HPV Infection'] == 'Yes').astype(int) * 2 +
        (data['Age'] > 50).astype(int) * 1 +
        (data['Poor Oral Hygiene'] == 'Yes').astype(int) * 1 +
        (data['Family History of Cancer'] == 'Yes').astype(int) * 1 +
        (data['Oral Lesions'] == 'Yes').astype(int) * 2 +
        (data['Diet (Fruits & Vegetables Intake)'] == 'Low').astype(int) * 1
    )
    
    # Add some noise and create binary target
    risk_with_noise = risk_factors + np.random.normal(0, 1, n_samples)
    data['Oral Cancer (Diagnosis)'] = (risk_with_noise > np.percentile(risk_with_noise, 70)).astype(int)
    data['Oral Cancer (Diagnosis)'] = ['Yes' if x == 1 else 'No' for x in data['Oral Cancer (Diagnosis)']]
    
    return pd.DataFrame(data)

def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    try:
        # Test imports
        from data_preprocessing import DataPreprocessor
        from model_inference import OralCancerPredictor
        print("✅ All modules imported successfully")
        
        # Test initialization
        preprocessor = DataPreprocessor()
        predictor = OralCancerPredictor()
        print("✅ Classes initialized successfully")
        
        # Test if dataset exists
        dataset_path = 'Data/01_Raw/oral_cancer_prediction_dataset.csv'
        if os.path.exists(dataset_path):
            print("✅ Dataset found")
        else:
            print("⚠️ Dataset not found - creating mock data")
            df = create_mock_dataset()
            os.makedirs('Data/01_Raw', exist_ok=True)
            df.to_csv(dataset_path, index=False)
            print("✅ Mock dataset created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def print_instructions():
    """Print usage instructions"""
    print("\n=== How to Use ===")
    print("1. Train the model:")
    print("   python Model/model_training.py")
    print("")
    print("2. Run the app:")
    print("   streamlit run App/streamlit_app.py")
    print("")
    print("3. Or use Docker:")
    print("   docker-compose up --build")

def main():
    """Main demo function"""
    print("🦷 Oral Cancer Risk Prediction System - Demo")
    print("=" * 50)
    
    # Test basic functionality
    basic_test_passed = test_basic_functionality()
    
    # Print instructions
    print_instructions()
    
    # Summary
    print("\n=== Summary ===")
    if basic_test_passed:
        print("✅ System is ready to use!")
    else:
        print("❌ Please check the errors above")
    
    print("\n👥 Developed by: Students s25809, s24339, s24784")
    print("📚 Course: SUML 2023/2024")

if __name__ == "__main__":
    main() 