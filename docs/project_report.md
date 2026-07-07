# Ambulance Dispatch Delay Risk Prediction System

## 1. Project Overview

Ambulance dispatch delays can affect emergency response outcomes, especially when incidents occur in high-traffic zones, difficult weather, distant locations, or areas with limited responder availability. This project builds a decision-support prototype that predicts whether an ambulance response case has elevated delay risk.

The system accepts emergency response attributes, processes them through a machine learning workflow, and returns a delay-risk prediction with operational recommendations. The goal is to help dispatchers and emergency response planners identify delay-prone cases or zones and take faster mitigation action.

## 2. Problem Statement

Build a working solution for an Ambulance Dispatch Delay Risk Prediction System that:

- Uses emergency response, dispatch, location, weather, and unit-related attributes.
- Predicts delay risk for single or batch emergency cases.
- Explains the result in simple terms.
- Provides actionable recommendations for dispatchers or public-safety decision-makers.
- Includes validation through model metrics and scenario checks.

## 3. Users and Real-World Impact

Primary users include:

- Emergency dispatchers who need quick risk awareness before assigning or monitoring an ambulance.
- Public-safety operations managers who review delay-prone areas and resource allocation.
- Data analysts who evaluate response-time patterns across zones, traffic levels, and incident types.

The system can support decisions such as:

- Assigning the closest suitable unit.
- Alerting backup or auxiliary units for high-risk cases.
- Monitoring high-risk dispatches more closely.
- Reviewing zones with repeated delay risk.
- Planning operational improvements based on model insights.

This is a decision-support prototype, not a replacement for emergency dispatch policy, clinical triage, live routing systems, or trained human judgment.

## 4. Dataset and Source Material

The project is designed around the Kaggle Integrated Emergency Response Dataset (IERAD):

https://www.kaggle.com/datasets/datasetengineer/integrated-emergency-response-dataset-ierad

The repository includes a normalized emergency response CSV at `data/emergency_response_data.csv`. It also includes a synthetic data generator at `data/create_synthetic_data.py` so the workflow can still run if Kaggle access is not available.

IERAD does not provide a direct binary `delay_risk` target in the same schema used by this project. The importer therefore creates a project-ready schema and derives a delay-risk target from operational factors such as distance, priority, traffic, weather, caller stress, unit-related fields, and region patterns. This engineered target is documented as a limitation because it demonstrates the ML workflow but does not prove real operational performance.

## 5. Input and Output Fields

Input features used by the model:

Categorical features:

- `priority_level`
- `incident_type`
- `time_of_day`
- `weather_conditions`
- `day_of_week`
- `traffic_density`
- `dispatch_zone`

Numerical features:

- `distance_to_scene`
- `crew_experience_years`
- `ambulance_age_years`
- `temperature`
- `historical_zone_delay_rate`
- `caller_stress_score`

Target/output:

- `delay_risk`: `1` means elevated delay risk, and `0` means lower delay risk.

The Streamlit application also outputs:

- Delay probability.
- Risk band: low, moderate, or high.
- Binary delay prediction.
- Prediction drivers.
- Dispatcher-oriented recommendations.

## 6. System Workflow

The project follows this workflow:

1. Load IERAD-compatible or synthetic emergency response data.
2. Normalize source columns into the project schema.
3. Validate required columns and remove incomplete records.
4. Split data into train, validation, and test sets.
5. Fit preprocessing transformers on training data only.
6. Train multiple machine learning models.
7. Tune model-specific decision thresholds on validation data.
8. Evaluate on the test set using classification metrics.
9. Save models, preprocessing pipeline, metrics, and plots.
10. Serve predictions and recommendations through a Streamlit application.

Main code locations:

- `src/data_ingestion.py`: IERAD import and schema normalization.
- `src/preprocessing.py`: validation, train/validation/test split, scaling, one-hot encoding, and ordinal encoding.
- `src/models.py`: logistic regression and neural network model definitions.
- `src/main.py`: training orchestration, threshold tuning, evaluation, and artifact saving.
- `src/utils.py`: plots, reports, and operational recommendations.
- `app/streamlit_app.py`: interactive dashboard and prediction interface.

## 7. AI / ML Component

The project uses supervised machine learning for delay-risk classification.

Three approaches are implemented:

1. Logistic regression baseline using one-hot encoded categorical features.
2. Fully connected neural network using integer-encoded categorical features.
3. Fully connected neural network using embedding layers for categorical features.

The AI component is useful because dispatch delay risk is influenced by several interacting factors. For example, distance alone may not be enough to indicate risk, but distance combined with traffic, weather, time of day, priority level, and zone history can produce a much stronger risk signal.

The system also includes:

- Decision-threshold tuning to convert probabilities into practical risk flags.
- Feature-importance reporting for the logistic regression baseline.
- Counterfactual-style prediction driver explanations in the app.
- Rule-based operational recommendations based on risk probability and input conditions.

## 8. Prototype / Application Working

The Streamlit app can be launched with:

```bash
/home/priya_paul/.venv/bin/streamlit run app/streamlit_app.py
```

The app includes these screens:

- Overview: project status, dataset preview, and delay-risk summaries.
- Model Dashboard: model metrics, thresholds, feature importance, and generated figures.
- Single Prediction: user enters one emergency case and receives risk output.
- Batch Prediction: user uploads a CSV and downloads prediction results.
- About: model purpose, repository paths, and responsible-use notes.

For a single prediction, the user enters incident priority, incident type, time, weather, traffic, dispatch zone, distance, crew experience, ambulance age, temperature, historical zone delay rate, and caller stress score. The app then returns the delay probability, risk label, binary prediction, important drivers, and recommendations such as checking route alternatives, alerting backup responders, or using active monitoring for high-risk cases.

## 9. Results and Evaluation

The saved model comparison report is stored at `reports/model_comparison.json`. The current saved results are:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 97.57% | 96.18% | 96.90% | 96.54% | 99.79% |
| FCNN Integer Encoding | 98.36% | 97.35% | 97.98% | 97.66% | 99.91% |
| FCNN Embedding Layers | 98.66% | 98.46% | 97.70% | 98.08% | 99.95% |

The FCNN with embedding layers currently performs best among the saved artifacts. The repository also includes confusion matrices, ROC curve comparison, model metric comparison, and neural network training-history plots under `reports/figures/`.

Validation methods used:

- Train/validation/test split.
- Accuracy, precision, recall, F1-score, and AUC-ROC.
- Confusion matrix plots.
- ROC curve comparison.
- Decision-threshold tuning on validation data.
- Smoke test of saved model artifacts through the prediction path.

## 10. Operational Recommendations

The project does not stop at model output. It translates predictions into next-step suggestions. Examples include:

- For low risk: proceed with routine dispatch protocol.
- For high traffic: use live routing and active siren priority.
- For long distance: alert auxiliary zone or backup responders.
- For hazardous weather: warn the crew about road conditions.
- For high-priority cases: consider cross-dispatching first responders if ambulance delay is likely.
- For high caller stress: keep the line open and provide pre-arrival instructions.

These recommendations are intended as decision-support suggestions and must be reviewed within real emergency service protocols.

## 11. Limitations and Responsible Use

Important limitations:

- The delay-risk target may be engineered when the source data does not provide a direct operational delay label.
- Current performance metrics should not be interpreted as validated real-world dispatch accuracy.
- The model does not use live GPS, live hospital capacity, real-time traffic feeds, or current ambulance availability.
- Local emergency response policies, clinical triage rules, and dispatcher judgment must always take priority.
- Bias or data quality issues may affect results if historical records overrepresent certain zones, incident types, or operating conditions.

Responsible use guidance:

- Use the system as a planning and risk-awareness tool.
- Validate on local real-world dispatch data before operational use.
- Review predictions with emergency response experts.
- Monitor false positives and false negatives carefully.
- Do not use the model as the sole basis for emergency dispatch decisions.

## 12. Future Improvements

Possible improvements:

- Use true response-time or arrival-delay labels if available.
- Add regression modeling for estimated response time in minutes.
- Integrate real-time traffic, GPS, and ambulance availability feeds.
- Add geospatial hotspot maps for delay-prone zones.
- Improve explanations with SHAP or another model-interpretability method.
- Add automated tests for preprocessing, inference, and batch upload workflows.
- Add screenshots and a recorded demo video for final submission.

## 13. Conclusion

This project delivers a working ambulance dispatch delay-risk prediction prototype with data preparation, preprocessing, machine learning models, evaluation reports, visualizations, and an interactive Streamlit app. It satisfies the core technical requirements for an AIML emergency response project and provides actionable output for users. Before real operational use, it must be validated on trusted local emergency response data and reviewed under public-safety governance.
