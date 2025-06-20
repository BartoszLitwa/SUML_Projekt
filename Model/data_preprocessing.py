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
        
        # Analyze target distribution
        target_counts = df[self.target_column].value_counts()
        print(f"Target distribution:\n{target_counts}")
        
        # Calculate class balance
        total = len(df)
        pos_rate = target_counts.get('Yes', 0) / total
        neg_rate = target_counts.get('No', 0) / total
        print(f"Positive rate: {pos_rate:.3f} ({pos_rate*100:.1f}%)")
        print(f"Negative rate: {neg_rate:.3f} ({neg_rate*100:.1f}%)")
        
        # Analyze key risk factors
        print("\n=== Key Risk Factor Analysis ===")
        risk_factors = ['Tobacco Use', 'Alcohol Consumption', 'HPV Infection']
        for factor in risk_factors:
            if factor in df.columns:
                cross_tab = pd.crosstab(df[factor], df[self.target_column], normalize='index')
                print(f"\n{factor} vs Cancer:")
                print(cross_tab)
        
    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove ID column if exists
        if 'ID' in df_clean.columns:
            df_clean = df_clean.drop('ID', axis=1)
            
        # Clean and standardize target column
        if self.target_column in df_clean.columns:
            # Standardize target values to 'Yes'/'No'
            target_values = df_clean[self.target_column].astype(str).str.strip()
            df_clean[self.target_column] = target_values.replace({
                '1': 'Yes', '0': 'No', 'True': 'Yes', 'False': 'No',
                'yes': 'Yes', 'no': 'No', 'YES': 'Yes', 'NO': 'No'
            })
            
            # Remove any rows with invalid target values
            valid_targets = ['Yes', 'No']
            before_count = len(df_clean)
            df_clean = df_clean[df_clean[self.target_column].isin(valid_targets)]
            after_count = len(df_clean)
            
            if before_count != after_count:
                print(f"⚠️ Removed {before_count - after_count} rows with invalid target values")
            
            print(f"Target value counts after cleaning: {df_clean[self.target_column].value_counts()}")
        
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
    
    def create_feature_interactions(self, df):
        """Create meaningful feature interactions"""
        df_features = df.copy()
        
        print("\n=== Creating Feature Interactions ===")
        
        # High-risk combination: Tobacco + Alcohol
        if 'Tobacco Use' in df.columns and 'Alcohol Consumption' in df.columns:
            df_features['Tobacco_Alcohol_Risk'] = (
                (df['Tobacco Use'] == 1) & (df['Alcohol Consumption'] == 1)
            ).astype(int)
            print("✓ Created Tobacco + Alcohol interaction")
        
        # Age groups (higher risk in older populations)
        if 'Age' in df.columns:
            df_features['Age_High_Risk'] = (df['Age'] > 50).astype(int)
            df_features['Age_Very_High_Risk'] = (df['Age'] > 65).astype(int)
            print("✓ Created Age risk categories")
        
        # Multiple symptoms present
        symptom_cols = ['Oral Lesions', 'Unexplained Bleeding', 'Difficulty Swallowing', 'White or Red Patches in Mouth']
        available_symptoms = [col for col in symptom_cols if col in df.columns]
        if len(available_symptoms) >= 2:
            df_features['Multiple_Symptoms'] = df[available_symptoms].sum(axis=1)
            df_features['Multiple_Symptoms_Binary'] = (df_features['Multiple_Symptoms'] >= 2).astype(int)
            print(f"✓ Created multiple symptoms feature from {len(available_symptoms)} symptoms")
        
        # Combined lifestyle risk
        lifestyle_cols = ['Tobacco Use', 'Alcohol Consumption', 'Poor Oral Hygiene']
        available_lifestyle = [col for col in lifestyle_cols if col in df.columns]
        if len(available_lifestyle) >= 2:
            df_features['Lifestyle_Risk_Score'] = df[available_lifestyle].sum(axis=1)
            print(f"✓ Created lifestyle risk score from {len(available_lifestyle)} factors")
        
        print(f"Original features: {df.shape[1]}, Enhanced features: {df_features.shape[1]}")
        return df_features
    
    def select_features(self, df):
        """Select relevant features for prediction"""
        print("\n=== Feature Selection for Risk Prediction ===")
        
        # RISK FACTORS (things that exist BEFORE cancer diagnosis)
        risk_factors = [
            'Age', 'Gender', 'Tobacco Use', 'Alcohol Consumption',
            'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure',
            'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)',
            'Family History of Cancer', 'Compromised Immune System'
        ]
        
        # EARLY SYMPTOMS (might be present but are early indicators)
        early_symptoms = [
            'Oral Lesions', 'Unexplained Bleeding', 'Difficulty Swallowing',
            'White or Red Patches in Mouth'
        ]
        
        # OUTCOME FEATURES (DO NOT USE - these happen AFTER diagnosis)
        outcome_features = [
            'Tumor Size (cm)', 'Cancer Stage', 'Treatment Type', 
            'Survival Rate (5-Year, %)', 'Cost of Treatment (USD)',
            'Economic Burden (Lost Workdays per Year)', 'Early Diagnosis'
        ]
        
        # ENGINEERED FEATURES (created from feature interactions)
        engineered_features = [
            'Tobacco_Alcohol_Risk', 'Age_High_Risk', 'Age_Very_High_Risk',
            'Multiple_Symptoms', 'Multiple_Symptoms_Binary', 'Lifestyle_Risk_Score'
        ]
        
        # Combine risk factors, early symptoms, and engineered features
        selected_features = risk_factors + early_symptoms + engineered_features
        
        # Check which features are available and log them
        available_features = [col for col in selected_features if col in df.columns]
        missing_features = [col for col in selected_features if col not in df.columns]
        excluded_outcomes = [col for col in outcome_features if col in df.columns]
        
        print(f"Available risk factors: {len(available_features)}")
        print(f"Missing features: {missing_features}")
        print(f"Excluded outcome features: {excluded_outcomes}")
        
        # Ensure we have meaningful features
        if len(available_features) < 5:
            print("⚠️ Warning: Very few features available. Model may not perform well.")
        
        self.feature_columns = available_features
        
        X = df[available_features]
        y = df[self.target_column]
        
        print(f"Final feature set: {available_features}")
        print(f"Feature matrix shape: {X.shape}")
        
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
        
        # Create feature interactions
        df_enhanced = self.create_feature_interactions(df_encoded)
        
        # Select features
        X, y = self.select_features(df_enhanced)
        
        # Split data - handle small datasets
        print(f"\nPreparing train/test split...")
        print(f"Target unique values: {y.unique()}")
        print(f"Target value counts: {y.value_counts()}")
        
        # Check if we can use stratified split
        min_class_size = min(y.value_counts()) if len(y.value_counts()) > 1 else 0
        required_test_samples = max(1, int(len(y) * test_size))
        
        # Adjust test size for very small datasets
        if len(y) < 50:
            test_size = min(0.3, max(0.1, 10/len(y)))  # At least 10% but max 30%
            print(f"⚠️ Small dataset detected. Adjusted test_size to {test_size:.2f}")
            required_test_samples = max(1, int(len(y) * test_size))
        
        if len(y.unique()) > 1 and min_class_size >= 2 and min_class_size >= required_test_samples:
            # Use stratified split for balanced datasets
            print("✓ Using stratified split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            # Use simple split for small/imbalanced datasets
            print(f"⚠️ Using simple split (min_class_size={min_class_size}, required_test={required_test_samples})")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"\nTraining set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, save_dir=None):
        """Save the preprocessor components"""
        if save_dir is None:
            # Get the correct path to artifacts directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(script_dir, 'artifacts')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        
        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(save_dir, 'feature_columns.pkl'))
        
        print(f"Preprocessor saved to {save_dir}")
    
    def load_preprocessor(self, save_dir=None):
        """Load the preprocessor components"""
        if save_dir is None:
            # Get the correct path to artifacts directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(script_dir, 'artifacts')
            
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
        
        # Clean data
        input_df = self.clean_data(input_df)
        
        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns and col != self.target_column:
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Create feature interactions (same as during training)
        input_df = self.create_feature_interactions(input_df)
        
        # Select only feature columns
        input_df = input_df[self.feature_columns]
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        input_scaled = pd.DataFrame(input_scaled, columns=self.feature_columns)
        
        return input_scaled


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    
    # Get the correct path to the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'Data', '01_Raw', 'oral_cancer_prediction_dataset.csv')
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset_path)
    
    if X_train is not None:
        preprocessor.save_preprocessor()
        print("Data preprocessing completed successfully!") 