import pandas as pd
import numpy as np
import joblib
import os
from data_preprocessing import DataPreprocessor


class OralCancerPredictor:
    def __init__(self, model_dir='Model/artifacts'):
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.feature_columns = []
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            # Load preprocessor
            if not self.preprocessor.load_preprocessor(self.model_dir):
                return False
            
            # Load model metadata to find the best model
            metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                model_name = metadata['model_name']
                model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
            else:
                # Fallback to finding any model file
                model_files = [f for f in os.listdir(self.model_dir) if f.endswith('_model.pkl')]
                if not model_files:
                    raise FileNotFoundError("No model files found")
                model_path = os.path.join(self.model_dir, model_files[0])
            
            # Load the model
            self.model = joblib.load(model_path)
            self.feature_columns = self.preprocessor.feature_columns
            self.is_loaded = True
            
            print(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_risk(self, input_data):
        """Predict oral cancer risk for given input data"""
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        try:
            # Preprocess input data
            processed_data = self.preprocessor.transform_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]
            
            # Calculate risk percentage
            risk_percentage = probability[1] * 100  # Probability of positive class
            
            result = {
                'prediction': int(prediction),
                'risk_percentage': float(risk_percentage),
                'risk_level': self._get_risk_level(risk_percentage),
                'recommendations': self._get_recommendations(input_data, risk_percentage)
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def _get_risk_level(self, risk_percentage):
        """Categorize risk level based on percentage"""
        if risk_percentage < 20:
            return "Low"
        elif risk_percentage < 50:
            return "Moderate"
        elif risk_percentage < 80:
            return "High"
        else:
            return "Very High"
    
    def _get_recommendations(self, input_data, risk_percentage):
        """Generate personalized recommendations based on input data and risk"""
        recommendations = []
        
        # Basic recommendations for all users
        recommendations.append("Regular dental check-ups and oral examinations")
        recommendations.append("Maintain good oral hygiene")
        
        # Risk-specific recommendations
        if risk_percentage > 30:
            recommendations.append("Consider consulting an oncologist for further evaluation")
            recommendations.append("Schedule more frequent oral health screenings")
        
        # Feature-specific recommendations
        if isinstance(input_data, dict):
            if input_data.get('Tobacco Use') == 'Yes':
                recommendations.append("Quit tobacco use - this is the most important risk factor")
                recommendations.append("Consider nicotine replacement therapy or smoking cessation programs")
            
            if input_data.get('Alcohol Consumption') == 'Yes':
                recommendations.append("Limit alcohol consumption")
                recommendations.append("Avoid combining alcohol with tobacco use")
            
            if input_data.get('Poor Oral Hygiene') == 'Yes':
                recommendations.append("Improve oral hygiene practices")
                recommendations.append("Use fluoride toothpaste and mouthwash")
            
            if input_data.get('Diet (Fruits & Vegetables Intake)') == 'Low':
                recommendations.append("Increase intake of fruits and vegetables")
                recommendations.append("Consider antioxidant-rich foods")
            
            if input_data.get('HPV Infection') == 'Yes':
                recommendations.append("Discuss HPV vaccination with your healthcare provider")
                recommendations.append("Regular HPV screening and monitoring")
        
        return recommendations
    
    def get_feature_requirements(self):
        """Get the required features for prediction"""
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        return self.feature_columns
    
    def validate_input(self, input_data):
        """Validate input data format and completeness"""
        if not self.is_loaded:
            if not self.load_model():
                return False, "Model not loaded"
        
        if not isinstance(input_data, dict):
            return False, "Input data must be a dictionary"
        
        missing_features = []
        for feature in self.feature_columns:
            if feature not in input_data:
                missing_features.append(feature)
        
        if missing_features:
            return False, f"Missing features: {missing_features}"
        
        return True, "Input data is valid"
    
    def batch_predict(self, input_dataframe):
        """Make predictions for a batch of input data"""
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        try:
            results = []
            for index, row in input_dataframe.iterrows():
                input_dict = row.to_dict()
                result = self.predict_risk(input_dict)
                if result:
                    result['index'] = index
                    results.append(result)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return None


# Example usage and testing
def test_predictor():
    """Test the predictor with sample data"""
    predictor = OralCancerPredictor()
    
    # Sample input data
    sample_input = {
        'Age': 45,
        'Gender': 'Male',
        'Tobacco Use': 'Yes',
        'Alcohol Consumption': 'Yes',
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
    
    # Make prediction
    result = predictor.predict_risk(sample_input)
    
    if result:
        print("\n=== Prediction Result ===")
        print(f"Prediction: {result['prediction']}")
        print(f"Risk Percentage: {result['risk_percentage']:.2f}%")
        print(f"Risk Level: {result['risk_level']}")
        print("\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")
    else:
        print("Failed to make prediction")


if __name__ == "__main__":
    test_predictor() 