import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Oral Cancer (Diagnosis)'
        
    def load_data(self, file_path):
        """Load the oral cancer dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def explore_data(self, df):
        """Explore the dataset structure"""
        print("\n=== Dataset Information ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"Target distribution:\n{df[self.target_column].value_counts()}")
        
    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove ID column if exists
        if 'ID' in df_clean.columns:
            df_clean = df_clean.drop('ID', axis=1)
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def encode_features(self, df):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        # Define columns that need encoding
        categorical_columns = [
            'Country', 'Gender', 'Tobacco Use', 'Alcohol Consumption', 
            'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure',
            'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)',
            'Family History of Cancer', 'Compromised Immune System',
            'Oral Lesions', 'Unexplained Bleeding', 'Difficulty Swallowing',
            'White or Red Patches in Mouth', 'Treatment Type', 'Early Diagnosis',
            'Oral Cancer (Diagnosis)'
        ]
        
        # Encode categorical features
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def select_features(self, df):
        """Select relevant features for prediction"""
        # Features most relevant for oral cancer prediction
        selected_features = [
            'Age', 'Gender', 'Tobacco Use', 'Alcohol Consumption',
            'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure',
            'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)',
            'Family History of Cancer', 'Compromised Immune System',
            'Oral Lesions', 'Unexplained Bleeding', 'Difficulty Swallowing',
            'White or Red Patches in Mouth'
        ]
        
        # Filter to only include features that exist in the dataset
        available_features = [col for col in selected_features if col in df.columns]
        self.feature_columns = available_features
        
        X = df[available_features]
        y = df[self.target_column]
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data(self, file_path, test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None, None, None, None
        
        # Explore data
        self.explore_data(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode features
        df_encoded = self.encode_features(df_clean)
        
        # Select features
        X, y = self.select_features(df_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"\nTraining set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, save_dir='Model/artifacts'):
        """Save the preprocessor components"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        
        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(save_dir, 'feature_columns.pkl'))
        
        print(f"Preprocessor saved to {save_dir}")
    
    def load_preprocessor(self, save_dir='Model/artifacts'):
        """Load the preprocessor components"""
        try:
            self.label_encoders = joblib.load(os.path.join(save_dir, 'label_encoders.pkl'))
            self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
            self.feature_columns = joblib.load(os.path.join(save_dir, 'feature_columns.pkl'))
            print(f"Preprocessor loaded from {save_dir}")
            return True
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            return False
    
    def transform_input(self, input_data):
        """Transform new input data for prediction"""
        # Create DataFrame if input is dict
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns and col != self.target_column:
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Select only feature columns
        input_df = input_df[self.feature_columns]
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        input_scaled = pd.DataFrame(input_scaled, columns=self.feature_columns)
        
        return input_scaled


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        'Data/01_Raw/oral_cancer_prediction_dataset.csv'
    )
    
    if X_train is not None:
        preprocessor.save_preprocessor()
        print("Data preprocessing completed successfully!") 