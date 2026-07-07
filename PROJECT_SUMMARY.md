# Project Summary

## Objective

Build an ambulance dispatch delay risk prediction system that estimates whether a response case has elevated delay risk based on incident, route, weather, crew, and zone context.

## Delivered Components

- Kaggle IERAD import support with schema normalization
- Synthetic fallback dataset with 2000 emergency response records
- Preprocessing pipeline with validation, scaling, one-hot encoding, and ordinal encoding
- Three model approaches:
  - Logistic regression baseline
  - FCNN with integer encoded categorical features
  - FCNN with embedding layers
- Training orchestration script
- Saved preprocessing and model artifacts
- Evaluation report, tuned thresholds, feature importance, and visualizations
- Streamlit web application
- Jupyter notebook for exploration
- Setup and deliverables documentation

## Current Performance

The current saved report shows the FCNN embedding model has the highest accuracy among the saved artifacts at `98.66%` and the highest AUC-ROC at `99.95%`.

Model metrics are stored in `reports/model_comparison.json`.

## Important Caveat

The Kaggle IERAD source does not include a direct binary `delay_risk` column, so the project derives the target from operational dispatch conditions. Results demonstrate modeling workflow and application integration, not validated operational performance in a real emergency dispatch environment.
