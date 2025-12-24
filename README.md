# RetailGenius Customer Churn Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org/)

An end-to-end Machine Learning project for predicting customer churn in e-commerce, built with production-ready best practices.

## 📋 Project Overview

This project implements a customer churn prediction system for **RetailGenius**, a fictional e-commerce company. The goal is to identify customers at risk of churning and enable proactive retention strategies.

### Key Features

- **Modular Pipeline**: Separate scripts for data preparation, feature engineering, training, and inference
- **MLflow Integration**: Experiment tracking, model versioning, and serving
- **Explainable AI**: SHAP-based model interpretability
- **Production Ready**: PEP8 compliant, documented, and reproducible

## 🏗️ Project Structure

```
retailgenius-churn-prediction/
├── data/
│   ├── raw/                 # Original immutable data
│   ├── processed/           # Final datasets for modeling
│   ├── interim/             # Intermediate transformed data
│   └── external/            # External data sources
├── docs/                    # Sphinx documentation
├── models/                  # Trained models and artifacts
├── mlruns/                  # MLflow tracking data
├── notebooks/               # Jupyter notebooks for exploration
├── references/              # Data dictionaries and manuals
├── reports/
│   └── figures/             # Generated visualizations
├── src/
│   ├── data/                # Data processing scripts
│   │   ├── __init__.py
│   │   └── data_preparation.py
│   ├── features/            # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/              # Model training and inference
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── inference.py
│   └── visualization/       # Plotting utilities
│       ├── __init__.py
│       └── plots.py
├── requirements.txt         # Project dependencies
├── setup.py                 # Package installation
├── pyproject.toml           # Project configuration
├── Makefile                 # Convenience commands
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/EbotFabien/retailgenius-churn-prediction.git
cd retailgenius-churn-prediction
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Dataset

Download the E-Commerce Customer Churn dataset from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) and place it in `data/raw/`.

### 4. Run the Pipeline

```bash
# Run complete pipeline
make all

# Or run individual steps:
make data        # Data preparation
make features    # Feature engineering
make train       # Model training
make inference   # Run predictions
```

### 5. View MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000 in your browser
```

## 📊 Pipeline Steps

### Step 1: Data Preparation
```bash
python -m src.data.data_preparation
```
- Loads raw data
- Handles missing values
- Basic cleaning and validation

### Step 2: Feature Engineering
```bash
python -m src.features.feature_engineering
```
- Creates derived features
- Encodes categorical variables
- Scales numerical features

### Step 3: Model Training
```bash
python -m src.models.train
```
- Trains multiple models (Random Forest, XGBoost, LightGBM)
- Tracks experiments with MLflow
- Registers best model

### Step 4: Inference
```bash
python -m src.models.inference
```
- Loads registered model
- Generates predictions
- Outputs churn probabilities

## 🔬 Explainable AI (SHAP)

The project includes comprehensive SHAP analysis:

```bash
python -m src.visualization.shap_analysis
```

Generated visualizations:
- Summary plots
- Waterfall plots
- Force plots
- Beeswarm plots
- Dependence plots

## 📈 MLflow Features

- **Experiment Tracking**: Parameters, metrics, and artifacts
- **Model Registry**: Version control for models
- **Model Serving**: Local inference server

### Start Serving

```bash
mlflow models serve -m "models:/ChurnModel/Production" -p 5001
```

### Make Predictions

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"feature1": value1, ...}]}'
```

## 📖 Documentation

Generate documentation with Sphinx:

```bash
cd docs
make html
# Open docs/_build/html/index.html
```

## 🧪 Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 👥 Team

- EPITA International Programs - AI Project Methodology 2025-2026

## 📄 License

This project is for educational purposes as part of EPITA coursework.

## 📚 References

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [E-Commerce Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
