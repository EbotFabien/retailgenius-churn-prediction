RetailGenius Churn Prediction Documentation
============================================

Welcome to the documentation for the RetailGenius Customer Churn Prediction project.

This project implements an end-to-end machine learning pipeline for predicting
customer churn in e-commerce, featuring:

* **MLflow Integration**: Experiment tracking, model versioning, and serving
* **Explainable AI**: SHAP-based model interpretability
* **Production Ready**: PEP8 compliant, documented, and reproducible

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api/modules

Installation
------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/YOUR_USERNAME/retailgenius-churn-prediction.git
   cd retailgenius-churn-prediction

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

Quick Start
-----------

.. code-block:: bash

   # Run complete pipeline
   make all

   # View MLflow UI
   mlflow ui --port 5000

API Reference
-------------

Data Module
^^^^^^^^^^^
.. automodule:: src.data.data_preparation
   :members:

Features Module
^^^^^^^^^^^^^^^
.. automodule:: src.features.feature_engineering
   :members:

Models Module
^^^^^^^^^^^^^
.. automodule:: src.models.train
   :members:

.. automodule:: src.models.inference
   :members:

Visualization Module
^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.visualization.shap_analysis
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
