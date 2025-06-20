# 🦷 Oral Cancer Risk Prediction System

An AI-powered web application for assessing oral cancer risk using machine learning. This project was developed as part of the SUML 2025 course at the Polish-Japanese Academy of Information Technology.

## 👥 Team Information

- **Student IDs**: s25809, s24339, s24784
- **Project**: SUML 2025
- **Instructor**: dr Wojciech Oronowicz-Jaśkowiak
- **Institution**: Polsko-Japońska Akademia Technik Komputerowych w Warszawie

## 🎯 Project Overview

This application enables rapid assessment of oral cancer risk based on key factors such as age, lifestyle habits, smoking, alcohol consumption, and other medical parameters. Early detection of cancer risk can significantly impact diagnostic improvement and treatment effectiveness.

## ✨ Features

- **Risk Assessment**: AI-powered prediction of oral cancer risk
- **Interactive Web Interface**: User-friendly Streamlit application
- **Personalized Recommendations**: Tailored health advice based on risk factors
- **Risk Visualization**: Charts and gauges for easy understanding
- **Assessment History**: Track risk changes over time
- **Dockerized Deployment**: Easy deployment with Docker containers
- **CI/CD Pipeline**: Automated model training and deployment
- **Comprehensive Test Suite**: Automated accuracy, consistency, and validation tests

## 🏗️ Architecture

```
SUML_Projekt/
├── App/
│   └── streamlit_app.py          # Streamlit web application
├── Data/
│   └── 01_Raw/
│       └── oral_cancer_prediction_dataset.csv
├── Model/
│   ├── data_preprocessing.py      # Data preprocessing pipeline
│   ├── model_training.py         # ML model training
│   ├── model_inference.py        # Model inference and prediction
│   └── artifacts/                # Trained model artifacts
├── tests/                        # Model and pipeline tests (see below)
├── .github/workflows/            # CI/CD pipelines
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
└── docker-compose.yml            # Docker Compose setup
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SUML_Projekt
   ```

2. **Build and run with Docker Compose**
   ```bash
   # Build and start the application
   docker-compose up --build
   
   # Access the application at http://localhost:8501
   ```

3. **Train the model (first time setup)**
   ```bash
   # Run model training
   docker-compose --profile training run model-training
   ```

### Option 2: Local Development

1. **Clone and setup environment**
   ```bash
   git clone <repository-url>
   cd SUML_Projekt
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   cd Model
   python model_training.py
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run App/streamlit_app.py
   ```

4. **Access the application**
   Open your browser to `http://localhost:8501`

## 📊 Dataset

The application uses the **Oral Cancer Prediction Dataset** from Kaggle, which includes:

- **Demographics**: Age, Gender, Country
- **Lifestyle Factors**: Tobacco use, Alcohol consumption, Diet quality
- **Medical History**: HPV infection, Family history, Immune status
- **Symptoms**: Oral lesions, Bleeding, Swallowing difficulties
- **Clinical Features**: Tumor characteristics, Treatment history

## 🤖 Machine Learning Pipeline

### Data Preprocessing
- Missing value imputation
- Categorical feature encoding
- Feature scaling and normalization
- Class imbalance handling with SMOTE

### Model Training
The system trains and compares multiple algorithms:
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **LightGBM**
- **Support Vector Machine**

### Model Selection
- Cross-validation with ROC-AUC scoring
- Hyperparameter tuning with GridSearch
- Best model selection based on performance metrics

### Model Evaluation
- Accuracy and ROC-AUC metrics
- Confusion matrix analysis
- Feature importance analysis
- Cross-validation scores

## 🌐 Web Application Features

### 1. Risk Assessment Tab
- **Patient Information Input**: Comprehensive form for health data
- **Real-time Validation**: Input validation and error handling
- **Risk Calculation**: AI-powered risk percentage calculation
- **Risk Categorization**: Low, Moderate, High, Very High risk levels

### 2. Risk Analysis Tab
- **Risk Factor Visualization**: Charts showing present risk factors
- **Interactive Analysis**: Dynamic risk factor profiling

### 3. History Tab
- **Assessment Tracking**: Historical risk assessments
- **Trend Analysis**: Risk changes over time
- **Data Export**: Download assessment history

### User Interface Features
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean, professional interface with custom CSS
- **Interactive Charts**: Plotly-powered visualizations
- **Risk Gauge**: Visual risk percentage display
- **Color-coded Results**: Intuitive risk level indication

## 📋 User Stories Implementation

### ✅ User Story 1 (Implemented)
*"As a user, I want to input basic health data and receive oral cancer risk assessment"*

**Implementation:**
- Comprehensive input form with 15+ health parameters
- Real-time risk calculation with ML model
- Personalized risk percentage and level
- Tailored recommendations based on risk factors

**Technical Features:**
- Input validation and error handling
- Data preprocessing and normalization
- ML model inference with confidence scores
- Responsive web interface

## 🔧 Development

### Running Model Tests

The project includes a comprehensive, custom Python test suite (no pytest required) to ensure the model is accurate, consistent, and robust. All tests are in the `tests/` directory.

#### **Run All Tests Locally**
```bash
python tests/run_all_tests.py
```

#### **Run Individual Tests**
```bash
python tests/test_imports.py
python tests/test_model_training.py
python tests/test_model_accuracy.py
python tests/test_model_consistency.py
python tests/test_model_validation.py
```

#### **Test Suite Overview**

| Test File                   | Purpose                                                      |
|----------------------------|--------------------------------------------------------------|
| test_imports.py             | Verify all required modules can be imported                  |
| test_model_training.py      | Ensure model training completes and artifacts are created    |
| test_model_accuracy.py      | Check model accuracy on known high/low risk scenarios        |
| test_model_consistency.py   | Ensure predictions are consistent and risk ordering is valid |
| test_model_validation.py    | Test input validation, edge cases, and output format         |
| run_all_tests.py            | Run all tests in sequence                                   |

#### **Test Criteria**
- High-risk cases must score significantly higher than low-risk cases
- Model must be consistent (not random)
- Input validation and output format must be robust
- Edge cases (e.g., age extremes) must be handled

### Code Quality
```bash
# Lint code
flake8 .

# Format code
black .

# Type checking
mypy Model/
```

### CI/CD Pipeline

The project includes automated GitHub Actions workflows:

1. **Tests Workflow** (`.github/workflows/tests.yml`)
   - Runs on Python 3.9
   - Installs dependencies
   - Runs all model and pipeline tests in `tests/`
   - Fails if any test fails

2. **Model Training and Deployment** (`.github/workflows/model-training.yml`)
   - Automated model training on push/PR
   - Docker image building and testing
   - Artifact generation and storage
   - Deployment package creation

## 🔄 Model Retraining

The model can be retrained automatically via CI/CD or manually:

### Automatic Retraining
- Triggered on pushes to main branch
- Runs in GitHub Actions environment
- Uploads trained model artifacts
- Updates deployment package

### Manual Retraining
```bash
# Local retraining
cd Model
python model_training.py

# Docker retraining
docker-compose --profile training run model-training
```

## 📈 Performance Metrics

The current model achieves:
- **Accuracy**: ~85-90% (varies by algorithm)
- **ROC-AUC**: ~0.85-0.92
- **Precision**: ~80-85%
- **Recall**: ~85-90%

*Note: Actual performance depends on the specific dataset and model selection*

## 🐳 Docker Deployment

### Development
```bash
# Start development environment
docker-compose up

# Rebuild after changes
docker-compose up --build
```

### Production
```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# Check logs
docker-compose logs -f oral-cancer-app

# Health check
curl http://localhost:8501/_stcore/health
```

## 🛡️ Security Considerations

- **Data Privacy**: No personal data is stored permanently
- **Input Validation**: All inputs are validated and sanitized
- **Session Management**: Streamlit session state for temporary data
- **Container Security**: Minimal Docker image with security best practices

## 🔧 Configuration

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Port for Streamlit app (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)
- `PYTHONPATH`: Python path for module imports

### Model Configuration
- Model artifacts stored in `Model/artifacts/`
- Configurable hyperparameters in training scripts
- Feature selection can be modified in preprocessing

## 📚 API Documentation

### OralCancerPredictor Class

```python
from Model.model_inference import OralCancerPredictor

predictor = OralCancerPredictor()

# Make prediction
result = predictor.predict_risk({
    'Age': 45,
    'Gender': 'Male',
    'Tobacco Use': 1,  # Use 1/0 for binary features
    # ... other features
})

# Result structure
{
    'prediction': 1,  # 0 or 1
    'risk_percentage': 67.5,
    'risk_level': 'High',
    'recommendations': [...]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of an academic assignment for SUML 2025 course.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for educational and screening purposes only. It should not replace professional medical consultation. Always consult with healthcare professionals for medical advice and diagnosis.

## 📞 Support

For questions or issues related to this academic project, please contact the development team through the course platform or create an issue in the repository.

## 🙏 Acknowledgments

- **Dataset**: Oral Cancer Prediction Dataset from Kaggle
- **Instructor**: dr Wojciech Oronowicz-Jaśkowiak
- **Institution**: Polish-Japanese Academy of Information Technology
- **Libraries**: Streamlit, Scikit-learn, XGBoost, LightGBM, Plotly

---

*Developed with ❤️ by Team s25809, s24339, s24784 for SUML 2025*