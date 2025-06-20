#!/usr/bin/env python3
"""
Test model accuracy with known risk scenarios
"""
import sys
sys.path.append('Model')

from model_inference import OralCancerPredictor


def get_test_cases():
    """Define test cases with expected risk levels"""
    return [
        # Very High Risk Cases
        {
            'data': {
                'Age': 65,
                'Gender': 'Male',
                'Tobacco Use': 1,
                'Alcohol Consumption': 1,
                'HPV Infection': 1,
                'Betel Quid Use': 1,
                'Chronic Sun Exposure': 1,
                'Poor Oral Hygiene': 1,
                'Diet (Fruits & Vegetables Intake)': 0,
                'Family History of Cancer': 1,
                'Compromised Immune System': 1,
                'Oral Lesions': 1,
                'Unexplained Bleeding': 1,
                'Difficulty Swallowing': 1,
                'White or Red Patches in Mouth': 1
            },
            'expected_min_risk': 60.0,  # Should be high risk
            'description': 'Very high risk: elderly, multiple risk factors and symptoms'
        },
        # High Risk Cases  
        {
            'data': {
                'Age': 55,
                'Gender': 'Male',
                'Tobacco Use': 1,
                'Alcohol Consumption': 1,
                'HPV Infection': 0,
                'Betel Quid Use': 0,
                'Chronic Sun Exposure': 0,
                'Poor Oral Hygiene': 1,
                'Diet (Fruits & Vegetables Intake)': 0,
                'Family History of Cancer': 1,
                'Compromised Immune System': 0,
                'Oral Lesions': 1,
                'Unexplained Bleeding': 0,
                'Difficulty Swallowing': 0,
                'White or Red Patches in Mouth': 1
            },
            'expected_min_risk': 30.0,  # Should be moderate-high risk
            'description': 'High risk: tobacco+alcohol, family history, some symptoms'
        },
        # Low Risk Cases
        {
            'data': {
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
            'expected_max_risk': 30.0,  # Should be low risk
            'description': 'Low risk: young, no risk factors, good diet'
        },
        {
            'data': {
                'Age': 30,
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
            'expected_max_risk': 25.0,  # Should be very low risk
            'description': 'Very low risk: young adult, no risk factors'
        }
    ]


def test_model_accuracy():
    """Test model accuracy with known risk scenarios"""
    print('🧪 Testing model accuracy with known risk scenarios...')
    
    predictor = OralCancerPredictor()
    if not predictor.load_model():
        print('❌ Failed to load model')
        return False
    
    test_cases = get_test_cases()
    results = []
    failed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f'\n📋 Test {i}: {test_case["description"]}')
        
        result = predictor.predict_risk(test_case['data'])
        if not result:
            print(f'❌ Test {i} FAILED: Prediction returned None')
            failed_tests += 1
            continue
        
        risk_percentage = result['risk_percentage']
        risk_level = result['risk_level']
        
        print(f'   Predicted: {risk_percentage:.1f}% ({risk_level})')
        
        # Check expectations
        passed = True
        if 'expected_min_risk' in test_case:
            if risk_percentage < test_case['expected_min_risk']:
                print(f'   ❌ Expected >= {test_case["expected_min_risk"]}%, got {risk_percentage:.1f}%')
                passed = False
            else:
                print(f'   ✅ Risk >= {test_case["expected_min_risk"]}% as expected')
        
        if 'expected_max_risk' in test_case:
            if risk_percentage > test_case['expected_max_risk']:
                print(f'   ❌ Expected <= {test_case["expected_max_risk"]}%, got {risk_percentage:.1f}%')
                passed = False
            else:
                print(f'   ✅ Risk <= {test_case["expected_max_risk"]}% as expected')
        
        if not passed:
            failed_tests += 1
        
        results.append({
            'test': i,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'passed': passed
        })
    
    # Summary
    print(f'\n📊 Test Summary:')
    print(f'   Total tests: {len(test_cases)}')
    print(f'   Passed: {len(test_cases) - failed_tests}')
    print(f'   Failed: {failed_tests}')
    
    if failed_tests > 0:
        print(f'\n❌ {failed_tests} accuracy tests failed - model may not be performing well')
        return False
    else:
        print(f'\n✅ All accuracy tests passed - model shows reasonable discrimination')
        return True


if __name__ == "__main__":
    success = test_model_accuracy()
    if not success:
        sys.exit(1) 