import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_realistic_oral_cancer_dataset(n_samples=10000, random_state=42):
    """
    Create a realistic synthetic oral cancer dataset with proper medical correlations
    """
    np.random.seed(random_state)
    
    print("Creating realistic oral cancer dataset...")
    
    # Generate base demographics
    data = {}
    
    # Age: Higher risk with age (especially 40+)
    data['Age'] = np.random.normal(45, 15, n_samples)
    data['Age'] = np.clip(data['Age'], 18, 85).astype(int)
    
    # Gender: Slightly higher risk for males
    data['Gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    
    # Major Risk Factors with realistic correlations
    
    # Tobacco Use: Strong risk factor
    # Age correlation: older people more likely to be long-term users
    tobacco_prob = 0.3 + (data['Age'] - 40) * 0.01  # Base 30%, increases with age
    tobacco_prob = np.clip(tobacco_prob, 0.1, 0.7)
    data['Tobacco Use'] = np.random.binomial(1, tobacco_prob, n_samples)
    
    # Alcohol Consumption: Moderate risk factor, correlated with tobacco
    alcohol_base_prob = 0.4
    # People who use tobacco are more likely to drink
    alcohol_prob = np.where(data['Tobacco Use'] == 1, 
                          alcohol_base_prob + 0.3, 
                          alcohol_base_prob)
    data['Alcohol Consumption'] = np.random.binomial(1, alcohol_prob, n_samples)
    
    # HPV Infection: Significant risk factor, especially in younger people
    hpv_prob = 0.15 - (data['Age'] - 30) * 0.002  # Higher in younger people
    hpv_prob = np.clip(hpv_prob, 0.05, 0.25)
    data['HPV Infection'] = np.random.binomial(1, hpv_prob, n_samples)
    
    # Poor Oral Hygiene: Moderate risk factor
    hygiene_prob = 0.25 + (data['Tobacco Use'] * 0.2)  # Smokers have worse hygiene
    data['Poor Oral Hygiene'] = np.random.binomial(1, hygiene_prob, n_samples)
    
    # Family History: Genetic factor
    data['Family History of Cancer'] = np.random.binomial(1, 0.15, n_samples)
    
    # Diet: Protective factor
    diet_prob = 0.7 - (data['Tobacco Use'] * 0.2)  # Smokers eat less fruits/veggies
    data['Diet (Fruits & Vegetables Intake)'] = np.random.binomial(1, diet_prob, n_samples)
    
    # Other risk factors
    data['Betel Quid Use'] = np.random.binomial(1, 0.05, n_samples)  # Low prevalence
    data['Chronic Sun Exposure'] = np.random.binomial(1, 0.3, n_samples)
    data['Compromised Immune System'] = np.random.binomial(1, 0.1, n_samples)
    
    # Symptoms (early indicators)
    base_symptom_prob = 0.1
    
    # Symptoms are more likely in high-risk individuals
    high_risk_multiplier = (
        (data['Age'] > 50).astype(int) * 1.5 +
        data['Tobacco Use'] * 2.0 +
        data['Alcohol Consumption'] * 1.5 +
        data['HPV Infection'] * 1.8 +
        data['Poor Oral Hygiene'] * 1.3 +
        data['Family History of Cancer'] * 1.4
    )
    
    symptom_prob = base_symptom_prob * (1 + high_risk_multiplier * 0.2)
    symptom_prob = np.clip(symptom_prob, 0.02, 0.8)
    
    data['Oral Lesions'] = np.random.binomial(1, symptom_prob, n_samples)
    data['Unexplained Bleeding'] = np.random.binomial(1, symptom_prob * 0.7, n_samples)
    data['Difficulty Swallowing'] = np.random.binomial(1, symptom_prob * 0.5, n_samples)
    data['White or Red Patches in Mouth'] = np.random.binomial(1, symptom_prob * 0.8, n_samples)
    
    # Calculate Cancer Risk with realistic medical correlations
    
    # Base risk increases with age
    base_risk = 0.01 + (data['Age'] - 40) * 0.001  # 1% base risk, increases with age
    base_risk = np.clip(base_risk, 0.005, 0.05)
    
    # Risk multipliers based on medical literature
    risk_multipliers = (
        data['Tobacco Use'] * 8.0 +          # Tobacco: 8x increased risk
        data['Alcohol Consumption'] * 3.0 +   # Alcohol: 3x increased risk
        data['HPV Infection'] * 5.0 +         # HPV: 5x increased risk
        data['Poor Oral Hygiene'] * 2.0 +     # Poor hygiene: 2x increased risk
        data['Family History of Cancer'] * 2.5 + # Family history: 2.5x increased risk
        data['Betel Quid Use'] * 6.0 +        # Betel quid: 6x increased risk
        data['Chronic Sun Exposure'] * 1.5 +   # Sun exposure: 1.5x increased risk
        data['Compromised Immune System'] * 2.0 + # Immune: 2x increased risk
        data['Oral Lesions'] * 4.0 +          # Lesions: 4x increased risk
        data['Unexplained Bleeding'] * 3.0 +   # Bleeding: 3x increased risk
        data['Difficulty Swallowing'] * 3.5 +  # Swallowing: 3.5x increased risk
        data['White or Red Patches in Mouth'] * 3.0  # Patches: 3x increased risk
    )
    
    # Protective factors
    protective_factors = (
        data['Diet (Fruits & Vegetables Intake)'] * 0.5  # Good diet reduces risk by 50%
    )
    
    # Calculate final cancer probability
    cancer_probability = base_risk * (1 + risk_multipliers) * (1 - protective_factors)
    cancer_probability = np.clip(cancer_probability, 0.001, 0.95)
    
    # Generate cancer diagnosis based on calculated probability
    data['Oral Cancer (Diagnosis)'] = np.random.binomial(1, cancer_probability, n_samples)
    
    # Convert to string labels
    data['Oral Cancer (Diagnosis)'] = ['Yes' if x == 1 else 'No' for x in data['Oral Cancer (Diagnosis)']]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add ID column
    df['ID'] = range(1, n_samples + 1)
    
    # Reorder columns to match original dataset structure
    column_order = [
        'ID', 'Age', 'Gender', 'Tobacco Use', 'Alcohol Consumption', 
        'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure', 
        'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)', 
        'Family History of Cancer', 'Compromised Immune System', 
        'Oral Lesions', 'Unexplained Bleeding', 'Difficulty Swallowing', 
        'White or Red Patches in Mouth', 'Oral Cancer (Diagnosis)'
    ]
    
    df = df[column_order]
    
    # Print dataset statistics
    print("\n=== Realistic Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Cancer cases: {sum(df['Oral Cancer (Diagnosis)'] == 'Yes')} ({sum(df['Oral Cancer (Diagnosis)'] == 'Yes')/len(df)*100:.1f}%)")
    print(f"Healthy cases: {sum(df['Oral Cancer (Diagnosis)'] == 'No')} ({sum(df['Oral Cancer (Diagnosis)'] == 'No')/len(df)*100:.1f}%)")
    
    # Show risk factor correlations
    print("\n=== Risk Factor Analysis ===")
    cancer_yes = df[df['Oral Cancer (Diagnosis)'] == 'Yes']
    cancer_no = df[df['Oral Cancer (Diagnosis)'] == 'No']
    
    risk_factors = ['Tobacco Use', 'Alcohol Consumption', 'HPV Infection', 'Poor Oral Hygiene']
    for factor in risk_factors:
        yes_rate = cancer_yes[factor].mean()
        no_rate = cancer_no[factor].mean()
        print(f"{factor}: Cancer={yes_rate:.3f}, No Cancer={no_rate:.3f}, Ratio={yes_rate/no_rate:.2f}x")
    
    return df

def main():
    """Create and save realistic dataset"""
    # Create realistic dataset
    df = create_realistic_oral_cancer_dataset(n_samples=10000, random_state=42)
    
    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, 'Data', '01_Raw', 'realistic_oral_cancer_dataset.csv')
    
    df.to_csv(output_path, index=False)
    print(f"\nRealistic dataset saved to: {output_path}")
    
    # Also create a backup of original
    original_path = os.path.join(project_root, 'Data', '01_Raw', 'oral_cancer_prediction_dataset.csv')
    backup_path = os.path.join(project_root, 'Data', '01_Raw', 'original_oral_cancer_dataset_backup.csv')
    
    if os.path.exists(original_path):
        import shutil
        shutil.copy2(original_path, backup_path)
        print(f"Original dataset backed up to: {backup_path}")
        
        # Replace original with realistic dataset
        shutil.copy2(output_path, original_path)
        print(f"Replaced original dataset with realistic version")

if __name__ == "__main__":
    main() 