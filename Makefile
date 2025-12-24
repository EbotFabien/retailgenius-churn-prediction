.PHONY: all clean data features train inference serve docs lint format test help

# Python interpreter
PYTHON = python

# Directories
SRC_DIR = src
DATA_DIR = data
MODEL_DIR = models
DOCS_DIR = docs

# Default target
all: data features train
	@echo "Pipeline completed successfully!"

# Help target
help:
	@echo "RetailGenius Churn Prediction - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make all        - Run complete ML pipeline"
	@echo "  make data       - Run data preparation"
	@echo "  make features   - Run feature engineering"
	@echo "  make train      - Train models with MLflow tracking"
	@echo "  make inference  - Run model inference"
	@echo "  make shap       - Generate SHAP explanations"
	@echo ""
	@echo "MLflow Commands:"
	@echo "  make mlflow-ui  - Start MLflow UI on port 5000"
	@echo "  make serve      - Serve model on port 5001"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint       - Run linters (flake8, pylint)"
	@echo "  make format     - Format code with black"
	@echo "  make test       - Run tests"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs       - Generate Sphinx documentation"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean      - Clean generated files"
	@echo "  make install    - Install dependencies"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Data preparation
data:
	@echo "Running data preparation..."
	$(PYTHON) -m src.data.data_preparation
	@echo "Data preparation completed!"

# Feature engineering
features: data
	@echo "Running feature engineering..."
	$(PYTHON) -m src.features.feature_engineering
	@echo "Feature engineering completed!"

# Model training
train: features
	@echo "Training models with MLflow tracking..."
	$(PYTHON) -m src.models.train
	@echo "Model training completed!"

# Model inference
inference:
	@echo "Running inference..."
	$(PYTHON) -m src.models.inference
	@echo "Inference completed!"

# SHAP analysis
shap:
	@echo "Generating SHAP explanations..."
	$(PYTHON) -m src.visualization.shap_analysis
	@echo "SHAP analysis completed!"

# Start MLflow UI
mlflow-ui:
	@echo "Starting MLflow UI on http://localhost:5000"
	mlflow ui --port 5000

# Serve model
serve:
	@echo "Starting model serving on http://localhost:5001"
	mlflow models serve -m "models:/ChurnModel/Production" -p 5001 --no-conda

# Code formatting
format:
	@echo "Formatting code with black..."
	black $(SRC_DIR)/
	@echo "Code formatted!"

# Linting
lint:
	@echo "Running flake8..."
	flake8 $(SRC_DIR)/
	@echo "Running pylint..."
	pylint $(SRC_DIR)/
	@echo "Linting completed!"

# Type checking
typecheck:
	@echo "Running mypy..."
	mypy $(SRC_DIR)/

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v
	@echo "Tests completed!"

# Generate documentation
docs:
	@echo "Generating documentation..."
	cd $(DOCS_DIR) && make html
	@echo "Documentation generated in $(DOCS_DIR)/_build/html/"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(SRC_DIR)/**/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/interim/*
	rm -rf $(MODEL_DIR)/*
	rm -rf reports/figures/*
	rm -rf $(DOCS_DIR)/_build
	@echo "Cleaned!"

# Deep clean (including MLflow)
deep-clean: clean
	@echo "Deep cleaning (including MLflow runs)..."
	rm -rf mlruns/*
	@echo "Deep clean completed!"
