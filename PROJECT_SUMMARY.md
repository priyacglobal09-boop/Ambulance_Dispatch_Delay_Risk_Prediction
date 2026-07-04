# Project Summary

## Objective

Build an ambulance dispatch delay risk prediction system that estimates whether a response case has elevated delay risk based on incident, route, weather, crew, and zone context.

## Delivered Components

- Synthetic dataset with 2000 emergency response records
- Preprocessing pipeline with validation, scaling, one-hot encoding, and ordinal encoding
- Three model approaches:
  - Logistic regression baseline
  - FCNN with integer encoded categorical features
  - FCNN with embedding layers
- Training orchestration script
- Saved preprocessing and model artifacts
- Evaluation report and visualizations
- Streamlit web application
- Jupyter notebook for exploration
- Setup and deliverables documentation

## Current Performance

The current saved report shows the FCNN embedding model has the highest accuracy among the saved artifacts at `78.00%`. Logistic regression has the highest saved AUC-ROC at `82.47%`.

Model metrics are stored in `reports/model_comparison.json`.

## Important Caveat

The project uses synthetic data. Results demonstrate modeling workflow and application integration, not validated operational performance in a real emergency dispatch environment.
