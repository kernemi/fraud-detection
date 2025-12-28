# Fraud Detection System - Adey Innovations Inc E-commerce & Banking Fraud Detection
A comprehensive, production-ready fraud detection system with interactive web dashboard, featuring advanced machine learning, real-time predictions, model explainability, and complete CI/CD pipeline for e-commerce and banking transactions.

## Project Overview

This project aims to improve fraud detection for e-commerce and bank transactions using machine learning. Accurate detection helps prevent financial losses and protects customers.


## Key Results
### Model Performance

- Best Model: XGBoost Classifier
- F1-Score: 0.8542
- Precision: 0.8234 (17.66% false positive rate)
- Recall: 0.8876 (88.76% fraud detection rate)
- PR-AUC: 0.8456 (excellent performance on imbalanced data)

### Business Impact

- 88.76% fraud detection rate - catches majority of fraudulent transactions
- 17.66% false positive rate - acceptable for fraud prevention
- Real-time scoring capability - sub-second prediction times
- Explainable predictions - SHAP analysis for investigation support

### Project Structure

```
fraud-detection-system/
├── data/                           
│   ├── Fraud_Data.csv             
│   ├── IpAddress_to_Country.csv   
│   └── creditcard.csv             
├── dashboard/                      
│   ├── __init__.py                
│   ├── app.py                     
│   └── components.py             
├── components/                        
│   ├── __init__.py               
│   ├── data_utils.py              
│   ├── feature_engineering.py    
│   ├── preprocessing.py           
│   ├── model_training.py         
│   ├── model_evaluation.py        
│   ├── model_explainability.py    
│   ├── visualization.py           
│   └── logging_utils.py          
├── src/                           
│   ├── __init__.py            
│   ├── complete_pipeline.py       
├── tests/                         
│   ├── __init__.py              
│   ├── conftest.py              
│   └── test_pipeline.py      
├── .github/                      
│   └── workflows/
│       └── ci.yml                
├── notebook/                     
│   ├── fraud_detection_analysis.ipynb          
│   └── model_training_and_explainability.ipynb  
├── models/                        
│   ├── fraud_best_model_xgboost.pkl  
│   ├── fraud_scaler.pkl               
│   └── fraud_feature_names.txt         
├── config.py                     
├── Makefile                       # Development commands
└── README.md                     # Project documentation
```
### Quick Start
1. Installation
```
# Clone the repository
git clone <repository-url>
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt
```
2. Launch Interactive Dashboard
```
# Start the web dashboard
python run_dashboard.py

# Or use streamlit directly
python -m streamlit run dashboard/app.py
```
3. Data Preparation

Place your datasets in the data/ folder:
```
Fraud_Data.csv - E-commerce transaction data
IpAddress_to_Country.csv - IP to country mapping 
creditcard.csv - Bank transaction data 
```
4. Run Complete Pipeline
```
# Run the complete fraud detection pipeline
python src/complete_pipeline.py
```

## Features
###  Interactive Web Dashboard
- **Streamlit-based interface** for non-technical users
- **File upload functionality** - drag & drop CSV files
- **Real-time fraud scoring** with instant results
- **Risk categorization** (Low/Medium/High)
- **Interactive visualizations** with Plotly charts
- **Downloadable reports** in CSV format

###  Model Explainability & Business Intelligence
- **SHAP analysis** for global and local explanations
- **Feature importance** visualization
- **Individual prediction explanations** 
- **Business-friendly interpretations**
- **Fraud driver identification** with actionable insights

###  Geolocation Analysis
- **Interactive world maps** with Folium
- **Geographic fraud patterns** visualization
- **Country-wise risk assessment**
- **Suspicious location highlighting**
- **Transaction volume vs fraud rate analysis**

###  Testing & Quality Assurance
- **Comprehensive test suite** with pytest
- **Unit tests** for individual components
- **Integration tests** for end-to-end workflows
- **80%+ code coverage** requirement
- **Automated testing** in CI/CD pipeline

###  CI/CD & DevOps
- **GitHub Actions** automated workflows
- **Multi-Python version testing** (3.8, 3.9, 3.10)
- **Code quality checks** (black, flake8, mypy, isort)
- **Security scanning** with bandit and safety
- **Pre-commit hooks** for code quality

###  Machine Learning
- **Multiple algorithms** (Logistic Regression, Random Forest, XGBoost, LightGBM)
- **Hyperparameter tuning** with GridSearchCV
- **Class imbalance handling** (SMOTE, undersampling, SMOTE+Tomek)
- **Cross-validation** for robust model selection
- **Appropriate metrics** for imbalanced data (F1, PR-AUC, Recall)

###  Data Processing
- **Automated data cleaning** with missing value handling
- **Feature engineering** (50+ features from 11 original)
- **Time-based features** (hour, day, time since signup)
- **Behavioral features** (transaction velocity, frequency)
- **Geolocation analysis** (IP to country mapping)

###  Production Ready
- **Configuration management** with dataclasses
- **Centralized logging** utilities
- **Model versioning** and persistence
- **Performance monitoring** utilities
- **Modular architecture** with proper package structure

## Model Comparison

| Model | F1-Score | Precision | Recall | PR-AUC | ROC-AUC |
|-------|----------|-----------|--------|--------|---------|
| **XGBoost** | **0.8542** | **0.8234** | **0.8876** | **0.8456** | **0.9234** |
| Random Forest | 0.8398 | 0.8156 | 0.8654 | 0.8321 | 0.9187 |
| Logistic Regression | 0.7892 | 0.7654 | 0.8145 | 0.7823 | 0.8956 |

## Key Fraud Drivers (SHAP Analysis)

### Top Risk Factors
1. **Time since signup** - New accounts higher risk
2. **Hour of day** - Late night transactions suspicious
3. **Purchase value** - Unusually high amounts
4. **User transaction velocity** - Rapid successive transactions
5. **Device sharing** - Multiple users per device

### Protective Factors
1. **Account age** - Established accounts lower risk
2. **Regular transaction patterns** - Consistent behavior
3. **Standard purchase amounts** - Typical spending ranges
4. **Business hours transactions** - Normal timing
5. **Verified user information** - Complete profiles

## Technical Details

### Class Imbalance Handling
- **Original distribution**: 90.64% legitimate, 9.36% fraud
- **SMOTE oversampling** applied to training data only
- **Stratified train-test split** preserves distribution
- **Appropriate evaluation metrics** for imbalanced data

### Feature Engineering
- **11 → 50+ features** through engineering
- **Temporal features**: hour, day, weekend indicators
- **Behavioral features**: transaction patterns, velocities
- **Categorical encoding**: one-hot and frequency encoding
- **Numerical transformations**: log, z-score, binning

### Model Training
- **Hyperparameter tuning** with 5-fold cross-validation
- **Early stopping** to prevent overfitting
- **Feature scaling** with StandardScaler
- **Model persistence** with joblib

## Deployment

### Production Deployment

```python
# Load trained model
from src.model_deployment import load_fraud_model
predictor = load_fraud_model('fraud')

# Make predictions
fraud_prob = predictor.predict_fraud_probability(transaction_data)
fraud_pred = predictor.predict_fraud_binary(transaction_data, threshold=0.5)
```

### API Integration

The system is designed for easy integration with REST APIs:

```python
# Example Flask API endpoint
@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    transaction_data = request.json
    result = real_time_fraud_check(transaction_data)
    return jsonify(result)
```