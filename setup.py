"""Setup script for RetailGenius Churn Prediction package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="retailgenius-churn-prediction",
    version="1.0.0",
    author="EPITA AI PM Team",
    author_email="team@epita.fr",
    description="Customer Churn Prediction for RetailGenius E-commerce",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/retailgenius-churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pylint>=3.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "churn-train=src.models.train:run_training_pipeline",
            "churn-predict=src.models.inference:run_inference",
            "churn-shap=src.visualization.shap_analysis:run_shap_analysis",
        ],
    },
)
