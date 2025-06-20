# Model Testing Suite

This directory contains comprehensive tests for the Oral Cancer Risk Prediction model to ensure it performs significantly better than random chance and handles edge cases correctly.

## Test Files

### `test_imports.py`
- **Purpose:** Verify all required modules can be imported correctly
- **Tests:** DataPreprocessor, ModelTrainer, OralCancerPredictor imports

### `test_model_training.py`
- **Purpose:** Verify model training process works correctly
- **Tests:** Dataset loading, training completion, model artifact creation

### `test_model_accuracy.py`
- **Purpose:** Test model accuracy with known risk scenarios
- **Tests:** 
  - Very high-risk cases (≥60% expected risk)
  - High-risk cases (≥30% expected risk)  
  - Low-risk cases (≤25-30% expected risk)
- **Scenarios:**
  - Elderly with multiple risk factors
  - Tobacco + alcohol users with symptoms
  - Young people with no risk factors

### `test_model_consistency.py`
- **Purpose:** Verify model predictions are consistent and non-random
- **Tests:**
  - Same input produces identical results (consistency)
  - High-risk cases score higher than low-risk cases (ordering)
  - Meaningful discrimination between risk levels (>10% difference)

### `test_model_validation.py`
- **Purpose:** Test edge cases and input validation
- **Tests:**
  - Input validation (rejects incomplete data)
  - Edge cases (ages 18, 80, 100)
  - Output format validation
  - Risk percentage bounds (0-100%)

## Running Tests

### Individual Tests
```bash
# Run specific test
python tests/test_model_accuracy.py
python tests/test_model_consistency.py
python tests/test_model_validation.py
```

### All Tests
```bash
# Run comprehensive test suite
python tests/run_all_tests.py
```

### CI/CD
Tests are automatically run in GitHub Actions on push/PR to main/develop branches.

## Test Criteria

### Model Performance Requirements
- **Better than random:** High-risk cases must score significantly higher than low-risk cases
- **Consistency:** Same input must produce identical results (not random)
- **Medical validity:** Risk scores must align with known medical risk factors
- **Robustness:** Must handle edge cases and validate inputs properly

### Success Criteria
- ✅ Very high-risk cases: ≥60% risk score
- ✅ High-risk cases: ≥30% risk score  
- ✅ Low-risk cases: ≤25-30% risk score
- ✅ Risk ordering: High > Low with ≥10% difference
- ✅ Consistency: Standard deviation < 0.1% across repeated predictions
- ✅ Input validation: Rejects incomplete/invalid data
- ✅ Output format: Valid risk percentages (0-100%) and risk levels

### Test Data
Test cases are designed based on medical knowledge:

**High Risk Factors:**
- Age > 50 (especially > 65)
- Tobacco use + Alcohol consumption
- HPV infection
- Poor oral hygiene  
- Family history of cancer
- Oral lesions and symptoms

**Low Risk Profile:**
- Young age (18-30)
- No tobacco/alcohol use
- Good diet and oral hygiene
- No family history
- No symptoms

## Failure Investigation

If tests fail, check:

1. **Training Data:** Ensure dataset is properly formatted and accessible
2. **Model Performance:** Review training metrics and feature importance
3. **Feature Engineering:** Verify engineered features are created correctly
4. **Data Preprocessing:** Check label encoding and scaling
5. **Test Expectations:** Adjust thresholds if medically justified

## Adding New Tests

To add new test scenarios:

1. Add test case to appropriate test file
2. Define expected risk ranges based on medical literature
3. Update documentation
4. Ensure tests are deterministic and reproducible 