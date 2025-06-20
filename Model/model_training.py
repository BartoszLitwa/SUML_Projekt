import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'svm': SVC(random_state=42, probability=True)
        }
    
    def handle_imbalanced_data(self, X_train, y_train):
        """Handle imbalanced dataset using SMOTE"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Original dataset shape: {X_train.shape}")
        print(f"Resampled dataset shape: {X_resampled.shape}")
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_models(self, X_train, X_test, y_train, y_test, use_smote=True):
        """Train multiple models and compare performance"""
        self.initialize_models()
        
        # Handle imbalanced data if requested
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_imbalanced_data(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        print("\n=== Training Models ===")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train_balanced, y_train_balanced)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Store model performance
                self.model_scores[name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'model': model
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
                print(f"{name} - CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
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
        """Perform hyperparameter tuning for the best model"""
        if model_name is None:
            model_name = self.best_model_name
        
        print(f"\n=== Hyperparameter Tuning for {model_name} ===")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        }
        
        if model_name in param_grids:
            # Get the base model
            base_model = self.models[model_name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, 
                param_grids[model_name], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Update the best model
            self.best_model = grid_search.best_estimator_
            self.best_score = grid_search.best_score_
            
            return grid_search.best_estimator_
        else:
            print(f"No hyperparameter tuning defined for {model_name}")
            return self.best_model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Best Model"):
        """Detailed evaluation of the model"""
        print(f"\n=== Detailed Evaluation for {model_name} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
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
    
    def save_model(self, model=None, model_name=None, save_dir='Model/artifacts'):
        """Save the trained model"""
        if model is None:
            model = self.best_model
        if model_name is None:
            model_name = self.best_model_name
        
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
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        'Data/01_Raw/oral_cancer_prediction_dataset.csv'
    )
    
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