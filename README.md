# Ambulance Dispatch Delay Risk Prediction

AI-powered ambulance dispatch delay risk prediction system using three machine learning approaches:

- Logistic regression baseline with one-hot encoded categorical features
- Fully connected neural network with integer encoded categorical features
- Fully connected neural network with embedding layers for categorical features

The project includes Kaggle IERAD import support, a synthetic fallback dataset, reusable preprocessing pipeline, model training orchestration, generated model artifacts, evaluation reports, visualizations, and a Streamlit application for interactive predictions.

## Project Structure

```text
Ambulance_Dispatch_Delay_Risk_Prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── create_synthetic_data.py
│   └── emergency_response_data.csv
├── models/
│   ├── data_pipeline.pkl
│   ├── fcnn_embedding_model.pt
│   ├── fcnn_integer_model.pt
│   └── linear_regression_model.pkl
├── notebooks/
│   └── exploration_modeling.ipynb
├── reports/
│   ├── figures/
│   └── model_comparison.json
├── src/
│   ├── main.py
│   ├── data_ingestion.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
├── DELIVERABLES.md
├── PROJECT_SUMMARY.md
├── QUICKSTART.md
├── README.md
└── requirements.txt
```

## Current Model Results

The current generated report in `reports/model_comparison.json` contains:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Regression | 97.57% | 96.18% | 96.90% | 96.54% | 99.79% |
| FCNN Integer Encoding | 98.36% | 97.35% | 97.98% | 97.66% | 99.91% |
| FCNN Embedding Layers | 98.66% | 98.46% | 97.70% | 98.08% | 99.95% |

These numbers are generated from a 100,000-row stratified sample of the Kaggle IERAD dataset normalized into the project schema. Re-running `src/main.py` can produce slightly different neural network results unless all random seeds and runtime libraries are held constant.

The training pipeline also tunes model-specific decision thresholds on the validation split and saves linear-model feature importances to `reports/feature_importance_linear.json`.

## Kaggle IERAD Dataset

The project supports the Kaggle Integrated Emergency Response Dataset (IERAD):

```text
https://www.kaggle.com/datasets/datasetengineer/integrated-emergency-response-dataset-ierad
```

Kaggle usually requires a local API token at `~/.kaggle/kaggle.json` for CLI downloads. After configuring Kaggle credentials, run:

```bash
/home/priya_paul/.venv/bin/pip install -r requirements.txt
/home/priya_paul/.venv/bin/python src/main.py --data-source ierad --download-ierad
```

If you already downloaded the dataset from Kaggle, train from the local CSV or ZIP:

```bash
/home/priya_paul/.venv/bin/python src/main.py --data-source ierad --ierad-input /path/to/ierad.csv
```

`src/data_ingestion.py` normalizes IERAD-style columns into the project schema used by the models and Streamlit app.

IERAD does not include a direct binary `delay_risk` column, so the importer derives one from operational dispatch conditions such as distance, priority, traffic, weather, injuries, ambulance speed, fuel level, hospital capacity, and region. If a future dataset includes an explicit delay label, the importer will use that label directly.

## Quick Start

From the repository root:

```bash
/home/priya_paul/.venv/bin/pip install -r requirements.txt
/home/priya_paul/.venv/bin/python src/main.py --data-source current
/home/priya_paul/.venv/bin/streamlit run app/streamlit_app.py
```

The Streamlit app loads the saved preprocessing pipeline and model artifacts from `models/`.

## Main Workflows

### Train all models

```bash
/home/priya_paul/.venv/bin/python src/main.py
```

This trains on the current `data/emergency_response_data.csv`, fits preprocessing, trains the three models, saves model artifacts, and writes figures plus `reports/model_comparison.json`.

### Regenerate synthetic fallback data

```bash
/home/priya_paul/.venv/bin/python src/main.py --data-source synthetic
```

### Launch the Streamlit app

```bash
/home/priya_paul/.venv/bin/streamlit run app/streamlit_app.py
```

The app includes:

- Overview page with project status
- Model comparison dashboard
- Decision threshold and feature importance views
- Single-case prediction form
- Per-prediction driver explanations
- Batch CSV prediction workflow
- Batch risk distribution summaries
- About page with model details

### Open the notebook

```bash
cd notebooks
/home/priya_paul/.venv/bin/jupyter notebook exploration_modeling.ipynb
```

## Input Features

Categorical inputs:

- `priority_level`
- `incident_type`
- `time_of_day`
- `weather_conditions`
- `day_of_week`
- `traffic_density`
- `dispatch_zone`

Numerical inputs:

- `distance_to_scene`
- `crew_experience_years`
- `ambulance_age_years`
- `temperature`
- `historical_zone_delay_rate`
- `caller_stress_score`

Target:

- `delay_risk`, where `1` means elevated delay risk and `0` means lower delay risk.

## Notes

This system is a decision-support prototype trained on Kaggle IERAD-derived data with an engineered delay-risk target. It is not a replacement for emergency dispatch policy, live routing systems, clinical triage, or public-safety operating procedures.
