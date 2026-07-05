import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import load_pytorch_model_checkpoint  # noqa: E402
from preprocessing import AmbulanceDataPipeline  # noqa: E402
from utils import get_actionable_recommendations  # noqa: E402
from data_ingestion import convert_ierad_to_project_schema  # noqa: E402


MODEL_OPTIONS = {
    "FCNN with Embeddings": "embedding_fcnn",
    "FCNN with Integer Encoding": "integer_fcnn",
    "Logistic Regression": "linear",
}

MODEL_DISPLAY_NAMES = {
    "linear_regression": "Logistic Regression",
    "fcnn_integer": "FCNN with Integer Encoding",
    "fcnn_embedding": "FCNN with Embeddings",
}


st.set_page_config(
    page_title="Ambulance Delay Risk",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> AmbulanceDataPipeline:
    pipeline_path = REPO_ROOT / "models" / "data_pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing preprocessing pipeline: {pipeline_path}")
    return AmbulanceDataPipeline.load_pipeline(str(pipeline_path))


@st.cache_resource(show_spinner=False)
def load_models():
    pipeline = load_pipeline()
    model_dir = REPO_ROOT / "models"

    linear_path = model_dir / "linear_regression_model.pkl"
    integer_path = model_dir / "fcnn_integer_model.pt"
    embedding_path = model_dir / "fcnn_embedding_model.pt"

    missing = [str(path) for path in [linear_path, integer_path, embedding_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    with open(linear_path, "rb") as f:
        linear_model = pickle.load(f)

    integer_model = load_pytorch_model_checkpoint(str(integer_path))
    embedding_model = load_pytorch_model_checkpoint(
        str(embedding_path),
        categorical_dims=pipeline.categorical_dims,
    )

    return {
        "linear": linear_model,
        "integer_fcnn": integer_model,
        "embedding_fcnn": embedding_model,
    }


@st.cache_data(show_spinner=False)
def load_metrics():
    metrics_path = REPO_ROOT / "reports" / "model_comparison.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance():
    importance_path = REPO_ROOT / "reports" / "feature_importance_linear.json"
    if not importance_path.exists():
        return pd.DataFrame()
    return pd.read_json(importance_path)


@st.cache_data(show_spinner=False)
def load_dataset_preview():
    data_path = REPO_ROOT / "data" / "emergency_response_data.csv"
    if not data_path.exists():
        return None
    return pd.read_csv(data_path)


def model_threshold(selected_model: str, metrics: dict | None) -> float:
    if not metrics:
        return 0.5
    thresholds = metrics.get("decision_thresholds", {})
    return float(thresholds.get(selected_model, {}).get("threshold", 0.5))


def risk_label(probability: float) -> str:
    if probability >= 0.7:
        return "High delay risk"
    if probability >= 0.3:
        return "Moderate delay risk"
    return "Low delay risk"


def risk_color(probability: float) -> str:
    if probability >= 0.7:
        return "#b91c1c"
    if probability >= 0.3:
        return "#b45309"
    return "#047857"


def predict_one(raw_input: dict, selected_model: str, pipeline: AmbulanceDataPipeline, models: dict, threshold=0.5):
    if selected_model == "linear":
        features = pipeline.transform_single(raw_input, "linear")
    elif selected_model == "integer_fcnn":
        features = pipeline.transform_single(raw_input, "integer_fcnn")
    elif selected_model == "embedding_fcnn":
        features = pipeline.transform_single(raw_input, "embedding_fcnn")
    else:
        raise ValueError(f"Unknown model key: {selected_model}")

    probability = float(models[selected_model].predict_proba(features)[0])
    prediction = int(probability >= threshold)
    return probability, prediction


def predict_batch(df: pd.DataFrame, selected_model: str, pipeline: AmbulanceDataPipeline, models: dict, threshold=0.5):
    required_cols = pipeline.categorical_cols + pipeline.numerical_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        converted = convert_ierad_to_project_schema(df)
        missing_after_conversion = [col for col in required_cols if col not in converted.columns]
        if missing_after_conversion:
            raise ValueError("Uploaded CSV is missing required columns: " + ", ".join(missing))
        df = converted

    if selected_model == "linear":
        features = pipeline.transform_linear(df)
    else:
        features = pipeline.transform_integer(df)
    if isinstance(features, tuple):
        features = features[0]

    probabilities = models[selected_model].predict_proba(features)
    output = df.copy()
    output["delay_probability"] = probabilities
    output["delay_prediction"] = (probabilities >= threshold).astype(int)
    output["risk_band"] = [risk_label(float(p)) for p in probabilities]
    return output


def reference_input(pipeline: AmbulanceDataPipeline):
    preview = load_dataset_preview()
    if preview is None or preview.empty:
        return default_input(pipeline)

    reference = {}
    for col in pipeline.categorical_cols:
        reference[col] = preview[col].mode().iloc[0]
    for col in pipeline.numerical_cols:
        reference[col] = float(preview[col].median())
    return reference


def explain_prediction(raw_input: dict, selected_model: str, pipeline: AmbulanceDataPipeline, models: dict):
    base_probability, _ = predict_one(raw_input, selected_model, pipeline, models)
    baseline = reference_input(pipeline)
    rows = []
    for feature, baseline_value in baseline.items():
        if raw_input.get(feature) == baseline_value:
            continue
        counterfactual = dict(raw_input)
        counterfactual[feature] = baseline_value
        counterfactual_probability, _ = predict_one(counterfactual, selected_model, pipeline, models)
        rows.append(
            {
                "Feature": feature,
                "Current": raw_input.get(feature),
                "Reference": baseline_value,
                "Impact": base_probability - counterfactual_probability,
            }
        )
    return pd.DataFrame(rows).assign(AbsImpact=lambda df: df["Impact"].abs()).sort_values("AbsImpact", ascending=False).drop(columns=["AbsImpact"]).head(8)


def format_metrics_table(metrics: dict):
    rows = []
    for key, values in metrics["model_performance"].items():
        row = {"Model": MODEL_DISPLAY_NAMES.get(key, key)}
        row.update({metric: f"{value:.2%}" for metric, value in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def sidebar_navigation():
    st.sidebar.title("Ambulance Delay Risk")
    return st.sidebar.radio(
        "Navigation",
        ["Overview", "Model Dashboard", "Single Prediction", "Batch Prediction", "About"],
        label_visibility="collapsed",
    )


def overview_page(metrics):
    st.title("Ambulance Dispatch Delay Risk Prediction")
    st.caption("Decision-support prototype for estimating elevated dispatch delay risk.")

    stats = metrics.get("dataset_stats", {}) if metrics else {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{stats.get('total_records', 0):,}")
    col2.metric("Train", f"{stats.get('train_size', 0):,}")
    col3.metric("Validation", f"{stats.get('val_size', 0):,}")
    col4.metric("Test", f"{stats.get('test_size', 0):,}")
    if stats.get("data_source"):
        st.caption(f"Current artifact data source: {stats['data_source']}")

    st.subheader("System Components")
    st.write(
        "The project contains Kaggle IERAD import support, synthetic fallback data generation, "
        "preprocessing, three trained models, evaluation reports, visualizations, and this "
        "Streamlit interface."
    )

    preview = load_dataset_preview()
    if preview is not None:
        st.subheader("Dataset Explorer")
        filters = st.columns(4)
        region = filters[0].multiselect("Region", sorted(preview["dispatch_zone"].dropna().unique()))
        traffic = filters[1].multiselect("Traffic", sorted(preview["traffic_density"].dropna().unique()))
        weather = filters[2].multiselect("Weather", sorted(preview["weather_conditions"].dropna().unique()))
        incident = filters[3].multiselect("Incident", sorted(preview["incident_type"].dropna().unique()))

        filtered = preview.copy()
        if region:
            filtered = filtered[filtered["dispatch_zone"].isin(region)]
        if traffic:
            filtered = filtered[filtered["traffic_density"].isin(traffic)]
        if weather:
            filtered = filtered[filtered["weather_conditions"].isin(weather)]
        if incident:
            filtered = filtered[filtered["incident_type"].isin(incident)]

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Filtered Rows", f"{len(filtered):,}")
        kpi2.metric("Delay Risk Rate", f"{filtered['delay_risk'].mean():.2%}" if len(filtered) else "0.00%")
        kpi3.metric("Median Distance", f"{filtered['distance_to_scene'].median():.1f}" if len(filtered) else "0.0")

        chart_left, chart_right = st.columns(2)
        if len(filtered):
            chart_left.bar_chart(filtered.groupby("dispatch_zone")["delay_risk"].mean())
            chart_right.bar_chart(filtered.groupby("traffic_density")["delay_risk"].mean())
        st.dataframe(filtered.head(100), width="stretch")


def dashboard_page(metrics):
    st.title("Model Dashboard")

    if not metrics:
        st.warning("No saved metrics report found. Run `python src/main.py` to generate it.")
        return

    metrics_df = format_metrics_table(metrics)
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    thresholds = metrics.get("decision_thresholds", {})
    if thresholds:
        threshold_rows = [
            {"Model": label, "Threshold": values.get("threshold"), "Validation Score": values.get("score")}
            for key, values in thresholds.items()
            for label in [MODEL_OPTIONS and next((name for name, model_key in MODEL_OPTIONS.items() if model_key == key), key)]
        ]
        st.subheader("Decision Thresholds")
        st.dataframe(pd.DataFrame(threshold_rows), width="stretch", hide_index=True)

    chart_df = metrics_df.set_index("Model")[["Accuracy", "AUC-ROC"]].apply(
        lambda col: col.str.rstrip("%").astype(float) / 100.0
    )
    st.bar_chart(chart_df, width="stretch")

    importance = load_feature_importance()
    if not importance.empty:
        st.subheader("Linear Model Feature Importance")
        importance_chart = importance.head(15).set_index("feature")["absolute_importance"]
        st.bar_chart(importance_chart, width="stretch")

    st.subheader("Generated Figures")
    figure_paths = [
        ("Model Metrics", REPO_ROOT / "reports" / "figures" / "metrics_comparison_bar.png"),
        ("ROC Curves", REPO_ROOT / "reports" / "figures" / "roc_curve_comparison.png"),
        ("Linear Confusion Matrix", REPO_ROOT / "reports" / "figures" / "confusion_matrix_linear.png"),
        ("Integer FCNN Confusion Matrix", REPO_ROOT / "reports" / "figures" / "confusion_matrix_fcnn_integer.png"),
        ("Embedding FCNN Confusion Matrix", REPO_ROOT / "reports" / "figures" / "confusion_matrix_fcnn_embedding.png"),
    ]

    cols = st.columns(2)
    for idx, (caption, path) in enumerate(figure_paths):
        if path.exists():
            cols[idx % 2].image(str(path), caption=caption, width="stretch")


def default_input(pipeline: AmbulanceDataPipeline):
    return {
        "priority_level": pipeline.category_mappings["priority_level"][0],
        "incident_type": pipeline.category_mappings["incident_type"][0],
        "time_of_day": pipeline.category_mappings["time_of_day"][0],
        "weather_conditions": pipeline.category_mappings["weather_conditions"][0],
        "day_of_week": pipeline.category_mappings["day_of_week"][0],
        "traffic_density": pipeline.category_mappings["traffic_density"][0],
        "dispatch_zone": pipeline.category_mappings["dispatch_zone"][0],
        "distance_to_scene": 5.0,
        "crew_experience_years": 5,
        "ambulance_age_years": 4,
        "temperature": 70.0,
        "historical_zone_delay_rate": 0.20,
        "caller_stress_score": 5,
    }


def single_prediction_page(pipeline, models, metrics):
    st.title("Single Prediction")

    selected_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    selected_model = MODEL_OPTIONS[selected_label]
    threshold = model_threshold(selected_model, metrics)
    st.caption(f"Decision threshold: {threshold:.2f}")
    defaults = default_input(pipeline)

    left, right = st.columns(2)
    with left:
        raw_input = {
            "priority_level": st.selectbox("Priority level", pipeline.category_mappings["priority_level"], index=0),
            "incident_type": st.selectbox("Incident type", pipeline.category_mappings["incident_type"], index=0),
            "time_of_day": st.selectbox("Time of day", pipeline.category_mappings["time_of_day"], index=0),
            "weather_conditions": st.selectbox("Weather conditions", pipeline.category_mappings["weather_conditions"], index=0),
            "day_of_week": st.selectbox("Day of week", pipeline.category_mappings["day_of_week"], index=0),
            "traffic_density": st.selectbox("Traffic density", pipeline.category_mappings["traffic_density"], index=0),
            "dispatch_zone": st.selectbox("Dispatch zone", pipeline.category_mappings["dispatch_zone"], index=0),
        }

    with right:
        raw_input.update(
            {
                "distance_to_scene": st.number_input("Distance to scene", min_value=0.0, max_value=50.0, value=defaults["distance_to_scene"], step=0.1),
                "crew_experience_years": st.number_input("Crew experience years", min_value=0, max_value=40, value=defaults["crew_experience_years"], step=1),
                "ambulance_age_years": st.number_input("Ambulance age years", min_value=0, max_value=30, value=defaults["ambulance_age_years"], step=1),
                "temperature": st.number_input("Temperature", min_value=-30.0, max_value=130.0, value=defaults["temperature"], step=1.0),
                "historical_zone_delay_rate": st.slider("Historical zone delay rate", min_value=0.0, max_value=1.0, value=defaults["historical_zone_delay_rate"], step=0.01),
                "caller_stress_score": st.slider("Caller stress score", min_value=1, max_value=10, value=defaults["caller_stress_score"], step=1),
            }
        )

    if st.button("Predict delay risk", type="primary"):
        probability, prediction = predict_one(raw_input, selected_model, pipeline, models, threshold=threshold)
        color = risk_color(probability)
        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; padding: 1rem; background: #f8fafc;">
                <h3 style="margin: 0; color: {color};">{risk_label(probability)}</h3>
                <p style="font-size: 1.25rem; margin: 0.5rem 0 0;">Delay probability: <strong>{probability:.2%}</strong></p>
                <p style="margin: 0.25rem 0 0;">Binary prediction: <strong>{prediction}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Operational Recommendations")
        for recommendation in get_actionable_recommendations(raw_input, probability):
            st.write(recommendation)

        explanation = explain_prediction(raw_input, selected_model, pipeline, models)
        if not explanation.empty:
            st.subheader("Prediction Drivers")
            st.dataframe(explanation, width="stretch", hide_index=True)


def batch_prediction_page(pipeline, models, metrics):
    st.title("Batch Prediction")
    selected_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), key="batch_model")
    selected_model = MODEL_OPTIONS[selected_label]
    threshold = model_threshold(selected_model, metrics)
    st.caption(f"Decision threshold: {threshold:.2f}")

    st.write(
        "Upload a CSV with the same input columns as `data/emergency_response_data.csv`, "
        "or upload a raw IERAD-style CSV and the app will attempt to normalize it."
    )
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file is None:
        preview = load_dataset_preview()
        if preview is not None:
            st.caption("Example input format")
            st.dataframe(preview.drop(columns=["delay_risk"], errors="ignore").head(5), width="stretch")
        return

    df = pd.read_csv(uploaded_file)
    try:
        predictions = predict_batch(df, selected_model, pipeline, models, threshold=threshold)
    except Exception as exc:
        st.error(str(exc))
        return

    st.success(f"Generated predictions for {len(predictions):,} rows.")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Predicted High Risk", f"{predictions['delay_prediction'].mean():.2%}")
    summary_cols[1].metric("Average Probability", f"{predictions['delay_probability'].mean():.2%}")
    summary_cols[2].metric("Rows", f"{len(predictions):,}")
    st.bar_chart(predictions["risk_band"].value_counts(), width="stretch")
    st.dataframe(predictions.head(100), width="stretch")
    st.download_button(
        "Download predictions",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="ambulance_delay_predictions.csv",
        mime="text/csv",
    )


def about_page():
    st.title("About")
    st.write(
        "This application demonstrates a complete machine learning workflow for ambulance "
        "dispatch delay risk prediction using Kaggle IERAD-compatible data or synthetic "
        "fallback emergency response data."
    )
    st.write(
        "The models are intended for workflow demonstration and decision-support prototyping. "
        "They should not be used as operational emergency dispatch policy without validation "
        "on real local data, stakeholder review, and safety governance."
    )

    st.subheader("Repository Paths")
    st.code(
        "\n".join(
            [
                f"Repository: {REPO_ROOT}",
                f"Data: {REPO_ROOT / 'data'}",
                f"Models: {REPO_ROOT / 'models'}",
                f"Reports: {REPO_ROOT / 'reports'}",
            ]
        )
    )


def main():
    try:
        pipeline = load_pipeline()
        models = load_models()
    except Exception as exc:
        st.error(str(exc))
        st.info("Run `/home/priya_paul/.venv/bin/python src/main.py` from the repository root to regenerate artifacts.")
        st.stop()

    metrics = load_metrics()
    page = sidebar_navigation()

    if page == "Overview":
        overview_page(metrics)
    elif page == "Model Dashboard":
        dashboard_page(metrics)
    elif page == "Single Prediction":
        single_prediction_page(pipeline, models, metrics)
    elif page == "Batch Prediction":
        batch_prediction_page(pipeline, models, metrics)
    else:
        about_page()


if __name__ == "__main__":
    main()
