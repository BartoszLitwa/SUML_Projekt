#!/usr/bin/env python3
"""
Test model consistency and non-randomness
"""
import sys
import statistics
sys.path.append('Model')

from model_inference import OralCancerPredictor


def get_test_inputs():
    """Get test inputs for consistency testing"""
    return {
        'moderate_risk': {
            'Age': 45,
            'Gender': 'Male',
            'Tobacco Use': 1,
            'Alcohol Consumption': 1,
            'HPV Infection': 0,
            'Betel Quid Use': 0,
            'Chronic Sun Exposure': 0,
            'Poor Oral Hygiene': 1,
            'Diet (Fruits & Vegetables Intake)': 0,
            'Family History of Cancer': 0,
            'Compromised Immune System': 0,
            'Oral Lesions': 0,
            'Unexplained Bleeding': 0,
            'Difficulty Swallowing': 0,
            'White or Red Patches in Mouth': 0
        },
        'low_risk': {
            'Age': 25,
            'Gender': 'Female',
            'Tobacco Use': 0,
            'Alcohol Consumption': 0,
            'HPV Infection': 0,
            'Betel Quid Use': 0,
            'Chronic Sun Exposure': 0,
            'Poor Oral Hygiene': 0,
            'Diet (Fruits & Vegetables Intake)': 1,
            'Family History of Cancer': 0,
            'Compromised Immune System': 0,
            'Oral Lesions': 0,
            'Unexplained Bleeding': 0,
            'Difficulty Swallowing': 0,
            'White or Red Patches in Mouth': 0
        },
        'high_risk': {
            'Age': 65,
            'Gender': 'Male',
            'Tobacco Use': 1,
            'Alcohol Consumption': 1,
            'HPV Infection': 1,
            'Betel Quid Use': 0,
            'Chronic Sun Exposure': 0,
            'Poor Oral Hygiene': 1,
            'Diet (Fruits & Vegetables Intake)': 0,
            'Family History of Cancer': 1,
            'Compromised Immune System': 0,
            'Oral Lesions': 1,
            'Unexplained Bleeding': 1,
            'Difficulty Swallowing': 0,
            'White or Red Patches in Mouth': 1
        }
    }


def test_consistency():
    """Test that same input produces consistent results"""
    print('🔄 Testing model consistency (non-randomness)...')
    
    predictor = OralCancerPredictor()
    test_inputs = get_test_inputs()
    
    # Test same input multiple times - should give consistent results
    test_input = test_inputs['moderate_risk']
    
    # Run same prediction 5 times
    predictions = []
    for i in range(5):
        result = predictor.predict_risk(test_input)
        if result:
            predictions.append(result['risk_percentage'])
    
    if len(predictions) != 5:
        print('❌ Some predictions failed')
        return False
    
    # Check consistency
    std_dev = statistics.stdev(predictions) if len(predictions) > 1 else 0
    print(f'   Predictions: {[f"{p:.1f}%" for p in predictions]}')
    print(f'   Standard deviation: {std_dev:.3f}')
    
    if std_dev > 0.1:  # Allow tiny numerical differences
        print(f'❌ Model is inconsistent (std_dev={std_dev:.3f} > 0.1)')
        return False
    else:
        print('✅ Model predictions are consistent')
        return True


def test_risk_ordering():
    """Test that risk ordering makes sense"""
    print('\n🔍 Testing risk ordering...')
    
    predictor = OralCancerPredictor()
    test_inputs = get_test_inputs()
    
    # Get predictions for different risk levels
    low_result = predictor.predict_risk(test_inputs['low_risk'])
    high_result = predictor.predict_risk(test_inputs['high_risk'])
    
    if not low_result or not high_result:
        print('❌ Risk comparison predictions failed')
        return False
    
    low_risk = low_result['risk_percentage']
    high_risk = high_result['risk_percentage']
    
    print(f'   Low risk case: {low_risk:.1f}%')
    print(f'   High risk case: {high_risk:.1f}%')
    print(f'   Difference: {high_risk - low_risk:.1f}%')
    
    if high_risk <= low_risk:
        print('❌ Model failed to distinguish high vs low risk')
        return False
    elif (high_risk - low_risk) < 10.0:
        print('⚠️  Small difference between high and low risk - model may need improvement')
        return True  # Warning but not failure
    else:
        print('✅ Model correctly orders risk levels')
        return True


def test_model_consistency():
    """Run all consistency tests"""
    print('🧪 Testing model consistency and non-randomness...')
    
    # Test 1: Consistency
    consistency_passed = test_consistency()
    
    # Test 2: Risk ordering
    ordering_passed = test_risk_ordering()
    
    # Overall result
    if consistency_passed and ordering_passed:
        print('\n✅ All consistency tests passed')
        return True
    else:
        print('\n❌ Some consistency tests failed')
        return False


if __name__ == "__main__":
    success = test_model_consistency()
    if not success:
        sys.exit(1) 