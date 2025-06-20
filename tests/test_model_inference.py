import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add Model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))

from model_inference import OralCancerPredictor


class TestOralCancerPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = OralCancerPredictor()
        
        # Sample input data
        self.sample_input = {
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
    
    def test_initialization(self):
        """Test predictor initialization"""
        self.assertIsNotNone(self.predictor)
        self.assertFalse(self.predictor.is_loaded)
        self.assertIsNone(self.predictor.model)
    
    def test_get_risk_level(self):
        """Test risk level categorization"""
        # Test different risk levels
        self.assertEqual(self.predictor._get_risk_level(10), "Low")
        self.assertEqual(self.predictor._get_risk_level(30), "Moderate")
        self.assertEqual(self.predictor._get_risk_level(60), "High")
        self.assertEqual(self.predictor._get_risk_level(90), "Very High")
    
    def test_get_recommendations(self):
        """Test recommendation generation"""
        recommendations = self.predictor._get_recommendations(self.sample_input, 50)
        
        # Should always have basic recommendations
        self.assertIn("Regular dental check-ups and oral examinations", recommendations)
        self.assertIn("Maintain good oral hygiene", recommendations)
        
        # Should have tobacco-specific recommendations
        tobacco_recs = [rec for rec in recommendations if "tobacco" in rec.lower()]
        self.assertGreater(len(tobacco_recs), 0)
    
    def test_validate_input_format(self):
        """Test input validation"""
        # Test with valid input (will fail if model not loaded, but format is checked first)
        with patch.object(self.predictor, 'is_loaded', True):
            with patch.object(self.predictor, 'feature_columns', list(self.sample_input.keys())):
                is_valid, message = self.predictor.validate_input(self.sample_input)
                self.assertTrue(is_valid)
        
        # Test with invalid input type
        is_valid, message = self.predictor.validate_input("invalid_input")
        self.assertFalse(is_valid)
        self.assertIn("dictionary", message)
    
    @patch('os.path.exists')
    @patch('joblib.load')
    def test_load_model_failure(self, mock_joblib_load, mock_path_exists):
        """Test model loading failure"""
        mock_path_exists.return_value = False
        mock_joblib_load.side_effect = FileNotFoundError("No model files found")
        
        # Mock the preprocessor load method to fail
        with patch.object(self.predictor.preprocessor, 'load_preprocessor', return_value=False):
            result = self.predictor.load_model()
            self.assertFalse(result)
    
    def test_batch_predict_without_model(self):
        """Test batch prediction without loaded model"""
        df = pd.DataFrame([self.sample_input])
        result = self.predictor.batch_predict(df)
        self.assertIsNone(result)


class TestPredictorRecommendations(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for recommendation testing"""
        self.predictor = OralCancerPredictor()
    
    def test_tobacco_recommendations(self):
        """Test tobacco-specific recommendations"""
        input_with_tobacco = {'Tobacco Use': 'Yes'}
        recommendations = self.predictor._get_recommendations(input_with_tobacco, 30)
        
        tobacco_recs = [rec for rec in recommendations if "tobacco" in rec.lower() or "quit" in rec.lower()]
        self.assertGreater(len(tobacco_recs), 0)
    
    def test_alcohol_recommendations(self):
        """Test alcohol-specific recommendations"""
        input_with_alcohol = {'Alcohol Consumption': 'Yes'}
        recommendations = self.predictor._get_recommendations(input_with_alcohol, 30)
        
        alcohol_recs = [rec for rec in recommendations if "alcohol" in rec.lower()]
        self.assertGreater(len(alcohol_recs), 0)
    
    def test_hpv_recommendations(self):
        """Test HPV-specific recommendations"""
        input_with_hpv = {'HPV Infection': 'Yes'}
        recommendations = self.predictor._get_recommendations(input_with_hpv, 30)
        
        hpv_recs = [rec for rec in recommendations if "hpv" in rec.lower()]
        self.assertGreater(len(hpv_recs), 0)
    
    def test_high_risk_recommendations(self):
        """Test high risk specific recommendations"""
        recommendations = self.predictor._get_recommendations({}, 80)  # High risk
        
        # Should include oncologist consultation for high risk
        oncologist_recs = [rec for rec in recommendations if "oncologist" in rec.lower()]
        self.assertGreater(len(oncologist_recs), 0)


if __name__ == '__main__':
    unittest.main() 