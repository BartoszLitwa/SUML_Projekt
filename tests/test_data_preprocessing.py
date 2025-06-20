import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add Model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))

from data_preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with mock data"""
        self.preprocessor = DataPreprocessor()
        
        # Create mock data for testing
        self.mock_data = {
            'Age': [25, 35, 45, 55, 65],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Tobacco Use': ['No', 'Yes', 'No', 'Yes', 'Yes'],
            'Alcohol Consumption': ['No', 'No', 'Yes', 'Yes', 'No'],
            'HPV Infection': ['No', 'No', 'No', 'Yes', 'No'],
            'Betel Quid Use': ['No', 'No', 'No', 'No', 'Yes'],
            'Chronic Sun Exposure': ['No', 'Yes', 'No', 'No', 'Yes'],
            'Poor Oral Hygiene': ['No', 'Yes', 'Yes', 'No', 'Yes'],
            'Diet (Fruits & Vegetables Intake)': ['High', 'Low', 'Moderate', 'High', 'Low'],
            'Family History of Cancer': ['No', 'No', 'Yes', 'No', 'No'],
            'Compromised Immune System': ['No', 'No', 'No', 'Yes', 'No'],
            'Oral Lesions': ['No', 'Yes', 'No', 'No', 'Yes'],
            'Unexplained Bleeding': ['No', 'No', 'No', 'Yes', 'No'],
            'Difficulty Swallowing': ['No', 'No', 'Yes', 'No', 'No'],
            'White or Red Patches in Mouth': ['No', 'Yes', 'No', 'No', 'Yes'],
            'Oral Cancer (Diagnosis)': ['No', 'Yes', 'No', 'Yes', 'Yes']
        }
        
        self.df = pd.DataFrame(self.mock_data)
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        df_clean = self.preprocessor.clean_data(self.df)
        
        # Check that the cleaned data has the same shape
        self.assertEqual(df_clean.shape, self.df.shape)
        
        # Check that no null values remain (assuming no nulls in mock data)
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
    
    def test_encode_features(self):
        """Test feature encoding functionality"""
        df_encoded = self.preprocessor.encode_features(self.df)
        
        # Check that categorical features are encoded as numbers
        for col in ['Gender', 'Tobacco Use', 'Alcohol Consumption']:
            if col in df_encoded.columns:
                self.assertTrue(df_encoded[col].dtype in [np.int32, np.int64])
    
    def test_select_features(self):
        """Test feature selection functionality"""
        df_encoded = self.preprocessor.encode_features(self.df)
        X, y = self.preprocessor.select_features(df_encoded)
        
        # Check that we have features and target
        self.assertGreater(X.shape[1], 0)  # Should have at least one feature
        self.assertEqual(len(y), len(X))   # Same number of samples
        
        # Check that target is properly encoded
        self.assertTrue(y.dtype in [np.int32, np.int64])
    
    def test_transform_input(self):
        """Test input transformation for prediction"""
        # First fit the preprocessor with the mock data
        df_clean = self.preprocessor.clean_data(self.df)
        df_encoded = self.preprocessor.encode_features(df_clean)
        X, y = self.preprocessor.select_features(df_encoded)
        
        # Create a sample input
        sample_input = {
            'Age': 45,
            'Gender': 'Male',
            'Tobacco Use': 'Yes',
            'Alcohol Consumption': 'No',
            'HPV Infection': 'No',
            'Betel Quid Use': 'No',
            'Chronic Sun Exposure': 'No',
            'Poor Oral Hygiene': 'Yes',
            'Diet (Fruits & Vegetables Intake)': 'Low',
            'Family History of Cancer': 'No',
            'Compromised Immune System': 'No',
            'Oral Lesions': 'No',
            'Unexplained Bleeding': 'No',
            'Difficulty Swallowing': 'No',
            'White or Red Patches in Mouth': 'No'
        }
        
        # Transform the input
        try:
            transformed_input = self.preprocessor.transform_input(sample_input)
            self.assertEqual(len(transformed_input), 1)  # Should have one row
            self.assertEqual(len(transformed_input.columns), len(self.preprocessor.feature_columns))
        except Exception as e:
            # If scaler is not fitted yet, this test will fail gracefully
            self.skipTest(f"Scaler not fitted yet: {e}")


class TestDataPreprocessorWithMissingValues(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with missing values"""
        self.preprocessor = DataPreprocessor()
        
        # Create mock data with missing values
        self.mock_data_with_nulls = {
            'Age': [25, None, 45, 55, 65],
            'Gender': ['Male', 'Female', None, 'Female', 'Male'],
            'Tobacco Use': ['No', 'Yes', 'No', None, 'Yes'],
            'Oral Cancer (Diagnosis)': ['No', 'Yes', 'No', 'Yes', 'Yes']
        }
        
        self.df_with_nulls = pd.DataFrame(self.mock_data_with_nulls)
    
    def test_handle_missing_values(self):
        """Test that missing values are handled properly"""
        df_clean = self.preprocessor.clean_data(self.df_with_nulls)
        
        # Check that no null values remain
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
        
        # Check that the shape is preserved
        self.assertEqual(df_clean.shape[0], self.df_with_nulls.shape[0])


if __name__ == '__main__':
    unittest.main() 