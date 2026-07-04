# Ambulance Dispatch Delay Risk Prediction

AI-powered ambulance dispatch delay risk prediction system using three machine learning approaches:

- Logistic regression baseline with one-hot encoded categorical features
- Fully connected neural network with integer encoded categorical features
- Fully connected neural network with embedding layers for categorical features

The project includes a synthetic emergency response dataset, reusable preprocessing pipeline, model training orchestration, generated model artifacts, evaluation reports, visualizations, and a Streamlit application for interactive predictions.

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
| Linear Regression | 76.67% | 78.47% | 86.77% | 82.41% | 82.47% |
| FCNN Integer Encoding | 75.67% | 78.43% | 84.66% | 81.42% | 80.85% |
| FCNN Embedding Layers | 78.00% | 80.90% | 85.19% | 82.99% | 81.72% |

These numbers are generated from the checked-in synthetic dataset and saved artifacts. Re-running `src/main.py` can produce slightly different neural network results unless all random seeds and runtime libraries are held constant.

## Quick Start

From the repository root:

```bash
/home/priya_paul/.venv/bin/pip install -r requirements.txt
/home/priya_paul/.venv/bin/python src/main.py
/home/priya_paul/.venv/bin/streamlit run app/streamlit_app.py
```

The Streamlit app loads the saved preprocessing pipeline and model artifacts from `models/`.

## Main Workflows

### Train all models

```bash
/home/priya_paul/.venv/bin/python src/main.py
```

This regenerates the synthetic dataset, fits preprocessing, trains the three models, saves model artifacts, and writes figures plus `reports/model_comparison.json`.

### Launch the Streamlit app

```bash
/home/priya_paul/.venv/bin/streamlit run app/streamlit_app.py
```

The app includes:

- Overview page with project status
- Model comparison dashboard
- Single-case prediction form
- Batch CSV prediction workflow
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

This system is a decision-support prototype trained on synthetic data. It is not a replacement for emergency dispatch policy, live routing systems, clinical triage, or public-safety operating procedures.
