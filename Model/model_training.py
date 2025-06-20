import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os
from datetime import datetime

from data_preprocessing import DataPreprocessor


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.model_scores = {}
        
    def initialize_models(self):
        """Initialize different ML models to try"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=2000,
                class_weight='balanced'  # Handle class imbalance
            ),
            'random_forest': RandomForestClassifier(
                random_state=42, 
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
        }
    
    def handle_imbalanced_data(self, X_train, y_train):
        """Handle imbalanced dataset using SMOTE"""
        print(f"Original dataset shape: {X_train.shape}")
        
        # Convert target to numeric for SMOTE
        y_numeric = y_train.map({'No': 0, 'Yes': 1})
        print(f"Original class distribution: {np.bincount(y_numeric)}")
        
        # Check if SMOTE is needed and possible
        class_counts = np.bincount(y_numeric)
        min_class_count = min(class_counts)
        
        if min_class_count < 2:
            print("⚠️ Skipping SMOTE: Not enough samples in minority class")
            return X_train, y_numeric
        
        if len(class_counts) < 2:
            print("⚠️ Skipping SMOTE: Only one class present")
            return X_train, y_numeric
        
        try:
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_numeric)
            
            print(f"Resampled dataset shape: {X_resampled.shape}")
            print(f"Resampled class distribution: {np.bincount(y_resampled)}")
            
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️ SMOTE failed: {e}")
            print("Using original dataset without resampling")
            return X_train, y_numeric
    
    def train_models(self, X_train, X_test, y_train, y_test, use_smote=True):
        """Train multiple models and compare performance"""
        self.initialize_models()
        
        # Handle imbalanced data if requested
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_imbalanced_data(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Data quality analysis
        print("\n=== Data Quality Analysis ===")
        print(f"Training set shape: {X_train.shape}")
        print(f"Feature distribution (first few features):")
        for col in X_train.columns[:5]:
            unique_vals = X_train[col].nunique()
            print(f"  {col}: {unique_vals} unique values")
            if unique_vals <= 10:
                print(f"    Values: {sorted(X_train[col].unique())}")
        
        # Check for class balance
        y_counts = y_train.value_counts()
        print(f"\nClass distribution:")
        for class_val, count in y_counts.items():
            print(f"  {class_val}: {count} ({count/len(y_train)*100:.1f}%)")
        
        print("\n=== Training Models ===")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train_balanced, y_train_balanced)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Convert y_test to numeric if needed
                y_test_numeric = y_test.map({'No': 0, 'Yes': 1}) if y_test.dtype == 'object' else y_test
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_numeric, y_pred)
                roc_auc = roc_auc_score(y_test_numeric, y_pred_proba)
                
                # Store model performance
                self.model_scores[name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'model': model
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
                
                # Cross-validation - adjust CV folds for small datasets
                n_samples = len(X_train_balanced)
                cv_folds = min(5, max(2, n_samples // 10))  # Use fewer folds for small datasets
                
                if cv_folds >= 2:
                    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv_folds, scoring='roc_auc')
                    print(f"{name} - CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f}) (CV={cv_folds})")
                else:
                    print(f"{name} - Skipping CV (dataset too small: {n_samples} samples)")
                
                # Update best model
                if roc_auc > self.best_score:
                    self.best_score = roc_auc
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        print(f"\nBest model: {self.best_model_name} with ROC AUC: {self.best_score:.4f}")
        return self.best_model, self.best_model_name
    
    def hyperparameter_tuning(self, X_train, y_train, model_name=None):
        """Simple hyperparameter tuning for the best model"""
        if model_name is None:
            model_name = self.best_model_name
        
        print(f"\n=== Using default parameters for {model_name} ===")
        print("Skipping hyperparameter tuning to keep it simple for students")
        
        return self.best_model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Best Model"):
        """Detailed evaluation of the model"""
        print(f"\n=== Detailed Evaluation for {model_name} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Convert y_test to numeric if needed
        y_test_numeric = y_test.map({'No': 0, 'Yes': 1}) if y_test.dtype == 'object' else y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_numeric, y_pred)
        roc_auc = roc_auc_score(y_test_numeric, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_numeric, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_numeric, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test_numeric, y_pred),
            'confusion_matrix': cm
        }
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance from the model"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance:")
            print(importance_df.head(10))
            
            return importance_df
        else:
            print("Model doesn't have feature importance attribute")
            return None
    
    def save_model(self, model=None, model_name=None, save_dir=None):
        """Save the trained model"""
        if model is None:
            model = self.best_model
        if model_name is None:
            model_name = self.best_model_name
        if save_dir is None:
            # Get the correct path to artifacts directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(script_dir, 'artifacts')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f'{model_name}_model.pkl')
        joblib.dump(model, model_path)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'score': self.best_score,
            'timestamp': datetime.now().isoformat(),
            'model_scores': self.model_scores
        }
        
        metadata_path = os.path.join(save_dir, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, model, X):
        """Make predictions with the model"""
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return predictions, probabilities


def main():
    """Main training pipeline"""
    print("Starting Oral Cancer Risk Prediction Model Training...")
    
    # Initialize preprocessor and trainer
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # Get the correct path to the dataset (relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Use original dataset but with fixes to handle the data issues
    dataset_path = os.path.join(project_root, 'Data', '01_Raw', 'oral_cancer_prediction_dataset.csv')
    print("Using original dataset with improved preprocessing")
    
    print(f"Looking for dataset at: {dataset_path}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset_path)
    
    if X_train is None:
        print("Failed to prepare data. Exiting...")
        return
    
    # Train models
    best_model, best_model_name = trainer.train_models(X_train, X_test, y_train, y_test)
    
    # Hyperparameter tuning
    tuned_model = trainer.hyperparameter_tuning(X_train, y_train)
    
    # Evaluate the best model
    evaluation_results = trainer.evaluate_model(tuned_model, X_test, y_test, best_model_name)
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance(tuned_model, preprocessor.feature_columns)
    
    # Save model and preprocessor
    model_path = trainer.save_model(tuned_model, best_model_name)
    preprocessor.save_preprocessor()
    
    print("\nModel training completed successfully!")
    print(f"Best model: {best_model_name}")
    print(f"ROC AUC Score: {trainer.best_score:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main() 