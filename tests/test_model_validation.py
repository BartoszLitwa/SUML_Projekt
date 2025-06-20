#!/usr/bin/env python3
"""
Test edge cases and input validation
"""
import sys
sys.path.append('Model')

from model_inference import OralCancerPredictor


def get_complete_input():
    """Get a complete valid input for testing"""
    return {
        'Age': 45,
        'Gender': 'Male',
        'Tobacco Use': 1,
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
    }


def test_input_validation():
    """Test input validation logic"""
    print('🧪 Testing input validation...')
    
    predictor = OralCancerPredictor()
    
    print('\n1. Testing input validation:')
    
    # Test missing required field
    incomplete_input = {
        'Age': 45,
        'Gender': 'Male',
        'Tobacco Use': 1
        # Missing other required fields
    }
    
    is_valid, message = predictor.validate_input(incomplete_input)
    if is_valid:
        print('❌ Validation should fail for incomplete input')
        return False
    else:
        print('✅ Correctly rejected incomplete input')
    
    # Test valid complete input
    complete_input = get_complete_input()
    
    is_valid, message = predictor.validate_input(complete_input)
    if not is_valid:
        print(f'❌ Valid input rejected: {message}')
        return False
    else:
        print('✅ Correctly accepted valid input')
    
    return True


def test_edge_cases():
    """Test edge cases with different age values"""
    print('\n2. Testing edge cases:')
    
    predictor = OralCancerPredictor()
    complete_input = get_complete_input()
    
    edge_cases = [
        {'Age': 18, 'description': 'Minimum age'},
        {'Age': 80, 'description': 'High age'},
        {'Age': 100, 'description': 'Maximum age'}
    ]
    
    for edge_case in edge_cases:
        test_input = complete_input.copy()
        test_input['Age'] = edge_case['Age']
        
        result = predictor.predict_risk(test_input)
        if not result:
            print(f'❌ Failed to predict for {edge_case["description"]} (age {edge_case["Age"]})')
            return False
        
        # Risk percentage should be between 0 and 100
        risk = result['risk_percentage']
        if not (0 <= risk <= 100):
            print(f'❌ Invalid risk percentage {risk}% for {edge_case["description"]}')
            return False
        
        print(f'✅ {edge_case["description"]} (age {edge_case["Age"]}): {risk:.1f}%')
    
    return True


def test_output_format():
    """Test that output format is correct"""
    print('\n3. Testing output format:')
    
    predictor = OralCancerPredictor()
    test_input = get_complete_input()
    
    result = predictor.predict_risk(test_input)
    if not result:
        print('❌ Failed to get prediction result')
        return False
    
    # Check required fields
    required_fields = ['risk_percentage', 'risk_level', 'recommendations']
    for field in required_fields:
        if field not in result:
            print(f'❌ Missing required field: {field}')
            return False
    
    # Check field types
    if not isinstance(result['risk_percentage'], (int, float)):
        print('❌ risk_percentage should be numeric')
        return False
    
    if not isinstance(result['risk_level'], str):
        print('❌ risk_level should be string')
        return False
    
    if not isinstance(result['recommendations'], list):
        print('❌ recommendations should be list')
        return False
    
    # Check risk_level values
    valid_risk_levels = ['Low', 'Moderate', 'High', 'Very High']
    if result['risk_level'] not in valid_risk_levels:
        print(f'❌ Invalid risk level: {result["risk_level"]}')
        return False
    
    print('✅ Output format is correct')
    return True


def test_model_validation():
    """Run all validation tests"""
    print('🧪 Testing edge cases and input validation...')
    
    # Test 1: Input validation
    validation_passed = test_input_validation()
    
    # Test 2: Edge cases
    edge_cases_passed = test_edge_cases()
    
    # Test 3: Output format
    output_format_passed = test_output_format()
    
    # Overall result
    if validation_passed and edge_cases_passed and output_format_passed:
        print('\n✅ All validation tests passed')
        return True
    else:
        print('\n❌ Some validation tests failed')
        return False


if __name__ == "__main__":
    success = test_model_validation()
    if not success:
        sys.exit(1) 