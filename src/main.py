import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Configure pathing dynamically so imports succeed from any execution directory
repo_root = "/home/priya_paul/.venv/repos/Ambulance_Dispatch_Delay_Risk_Prediction"
if repo_root not in sys.path:
    sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "src"))
sys.path.append(os.path.join(repo_root, "data"))

from preprocessing import AmbulanceDataPipeline
from models import (
    EmergencyDataset,
    LogisticRegressionBaseline,
    FCNNIntegerNet,
    FCNNEmbeddingNet,
    PyTorchModelTrainer
)
import utils

# Metrics calculators
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, probas) -> dict:
    """Calculates five standard classification metrics."""
    preds = (probas >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        auc_score = roc_auc_score(y_true, probas)
    except ValueError:
        auc_score = 0.5
        
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC-ROC": auc_score
    }

def df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Formats a pandas DataFrame as a Markdown table without requiring external tabulate dependency."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join([str(val) for val in row]) + " |")
    return "\n".join([header, divider] + rows)

def main():
    print("======================================================================")
    print("🚑 STARTING AMBULANCE DISPATCH DELAY RISK PREDICTION TRAINING FLOW 🚑")
    print("======================================================================\n")
    
    # -------------------------------------------------------------
    # Step 1: Ensure Directories Exist & Load Data
    # -------------------------------------------------------------
    data_dir = os.path.join(repo_root, "data")
    models_dir = os.path.join(repo_root, "models")
    reports_dir = os.path.join(repo_root, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    
    for d in [data_dir, models_dir, reports_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)
        
    data_file = os.path.join(data_dir, "emergency_response_data.csv")
    
    # Force re-generation of advanced synthetic data to capture complex non-linear structures
    print("Forcing advanced synthetic data generation...")
    from create_synthetic_data import generate_synthetic_data
    df = generate_synthetic_data(2000, seed=42)
    df.to_csv(data_file, index=False)
    print(f"Generated 2000 records successfully at: {data_file}\n")
        
    # Instantiate Pipeline
    pipeline = AmbulanceDataPipeline()
    df_raw = pipeline.load_and_validate_data(data_file)
    print(f"Successfully loaded raw dataset of {len(df_raw)} records.")
    
    # Stratified Train/Val/Test Split
    train_df, val_df, test_df = pipeline.split_dataset(df_raw)
    print(f"Data Splits: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}\n")
    
    # Fit Pipeline on Train
    print("Fitting preprocessing transformers on train dataframe...")
    pipeline.fit(train_df)
    pipeline_path = os.path.join(models_dir, "data_pipeline.pkl")
    pipeline.save_pipeline(pipeline_path)
    print(f"Data preprocessing pipeline saved to: {pipeline_path}\n")
    
    # -------------------------------------------------------------
    # Step 2: Transform Features for All Three Approaches
    # -------------------------------------------------------------
    print("Preprocessing datasets for all three ML modeling branches...")
    
    # Branch 1: One-Hot Encoding for Linear Regression
    X_train_lin, y_train_lin = pipeline.transform_linear(train_df)
    X_val_lin, y_val_lin = pipeline.transform_linear(val_df)
    X_test_lin, y_test_lin = pipeline.transform_linear(test_df)
    
    # Branch 2: Integer Encoding for FCNN Integer and FCNN Embedding
    X_train_int, y_train_int = pipeline.transform_integer(train_df)
    X_val_int, y_val_int = pipeline.transform_integer(val_df)
    X_test_int, y_test_int = pipeline.transform_integer(test_df)
    
    print(f"  • One-Hot (Linear) feature space dimensions: {X_train_lin.shape[1]}")
    print(f"  • Integer (FCNN) feature space dimensions: {X_train_int.shape[1]}\n")
    
    # -------------------------------------------------------------
    # Step 3: Model 1 Training - Linear Regression (Logistic Regression)
    # -------------------------------------------------------------
    print("-------------------------------------------------------------")
    print("🤖 Model 1: Training Logistic Regression Baseline...")
    print("-------------------------------------------------------------")
    lr_model = LogisticRegressionBaseline(C=0.2, random_state=42)
    lr_model.fit(X_train_lin, y_train_lin)
    
    # Save Model
    lr_model_path = os.path.join(models_dir, "linear_regression_model.pkl")
    with open(lr_model_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"Logistic Regression baseline saved to: {lr_model_path}")
    
    # Evaluate
    probas_lr = lr_model.predict_proba(X_test_lin)
    metrics_lr = calculate_metrics(y_test_lin, probas_lr)
    print(f"Linear Regression Baseline - Test Accuracy: {metrics_lr['Accuracy']:.2%}, AUC-ROC: {metrics_lr['AUC-ROC']:.2%}\n")
    
    # -------------------------------------------------------------
    # Step 4: Model 2 Training - FCNN with Integer Encoding (3-Layer Neural Net)
    # -------------------------------------------------------------
    print("-------------------------------------------------------------")
    print("🤖 Model 2: Training FCNN with Integer Encoding (3-Layer Net)...")
    print("-------------------------------------------------------------")
    
    # Build DataLoaders
    train_dataset_int = EmergencyDataset(X_train_int, y_train_int)
    val_dataset_int = EmergencyDataset(X_val_int, y_val_int)
    
    train_loader_int = DataLoader(train_dataset_int, batch_size=64, shuffle=True)
    val_loader_int = DataLoader(val_dataset_int, batch_size=128, shuffle=False)
    
    int_net = FCNNIntegerNet(input_dim=13, hidden_dim1=64, hidden_dim2=32)
    trainer_int = PyTorchModelTrainer(int_net, "integer_fcnn", lr=0.002)
    
    history_int = trainer_int.fit_loader(
        train_loader_int, 
        val_loader_int, 
        epochs=40, 
        early_stopping_patience=8
    )
    
    # Save training history plot
    utils.plot_training_history(
        history_int, 
        "FCNN Integer Encoding Training Curve", 
        os.path.join(figures_dir, "history_fcnn_integer.png")
    )
    
    # Save Model Checkpoint
    int_net_path = os.path.join(models_dir, "fcnn_integer_model.pt")
    trainer_int.save_model(int_net_path)
    print(f"FCNN Integer Encoding model saved to: {int_net_path}")
    
    # Evaluate
    probas_int = trainer_int.predict_proba(X_test_int)
    metrics_int = calculate_metrics(y_test_int, probas_int)
    print(f"FCNN Integer Baseline - Test Accuracy: {metrics_int['Accuracy']:.2%}, AUC-ROC: {metrics_int['AUC-ROC']:.2%}\n")
    
    # -------------------------------------------------------------
    # Step 5: Model 3 Training - FCNN with Embedding Layers (Advanced DL model)
    # -------------------------------------------------------------
    print("-------------------------------------------------------------")
    print("🤖 Model 3: Training FCNN with Category Embedding Layers [BEST]...")
    print("-------------------------------------------------------------")
    
    # DataLoaders (reusing integer encoding features since categorical elements are integer indexes)
    train_dataset_embed = EmergencyDataset(X_train_int, y_train_int)
    val_dataset_embed = EmergencyDataset(X_val_int, y_val_int)
    
    train_loader_embed = DataLoader(train_dataset_embed, batch_size=64, shuffle=True)
    val_loader_embed = DataLoader(val_dataset_embed, batch_size=128, shuffle=False)
    
    embed_net = FCNNEmbeddingNet(
        categorical_dims=pipeline.categorical_dims, 
        num_continuous_features=6,
        hidden_dim1=64,
        hidden_dim2=32
    )
    # Attach properties to model state
    embed_net.categorical_dims = pipeline.categorical_dims
    embed_net.cat_cols_order = pipeline.categorical_cols
    
    trainer_embed = PyTorchModelTrainer(embed_net, "embedding_fcnn", lr=0.002)
    
    history_embed = trainer_embed.fit_loader(
        train_loader_embed, 
        val_loader_embed, 
        epochs=50, 
        early_stopping_patience=10
    )
    
    # Save training history plot
    utils.plot_training_history(
        history_embed, 
        "FCNN Embedding Layers Training Curve", 
        os.path.join(figures_dir, "history_fcnn_embedding.png")
    )
    
    # Save Model Checkpoint
    embed_net_path = os.path.join(models_dir, "fcnn_embedding_model.pt")
    trainer_embed.save_model(embed_net_path)
    print(f"FCNN Embedding model saved to: {embed_net_path}")
    
    # Evaluate
    probas_embed = trainer_embed.predict_proba(X_test_int)
    metrics_embed = calculate_metrics(y_test_int, probas_embed)
    print(f"FCNN Category Embedding - Test Accuracy: {metrics_embed['Accuracy']:.2%}, AUC-ROC: {metrics_embed['AUC-ROC']:.2%}\n")
    
    # -------------------------------------------------------------
    # Step 6: Create Comparative Dashboard Visualizations & Reports
    # -------------------------------------------------------------
    print("-------------------------------------------------------------")
    print("📊 Generating Visualizations and Comparative Reports...")
    print("-------------------------------------------------------------")
    
    # Build comparison dataframe
    metrics_comparison = {
        "Model": [
            "Linear Regression (Logistic)",
            "FCNN (Integer)",
            "FCNN (Embedding)"
        ],
        "Accuracy": [
            f"{metrics_lr['Accuracy']:.2%}",
            f"{metrics_int['Accuracy']:.2%}",
            f"{metrics_embed['Accuracy']:.2%}"
        ],
        "Precision": [
            f"{metrics_lr['Precision']:.2%}",
            f"{metrics_int['Precision']:.2%}",
            f"{metrics_embed['Precision']:.2%}"
        ],
        "Recall": [
            f"{metrics_lr['Recall']:.2%}",
            f"{metrics_int['Recall']:.2%}",
            f"{metrics_embed['Recall']:.2%}"
        ],
        "F1-Score": [
            f"{metrics_lr['F1-Score']:.2%}",
            f"{metrics_int['F1-Score']:.2%}",
            f"{metrics_embed['F1-Score']:.2%}"
        ],
        "AUC-ROC": [
            f"{metrics_lr['AUC-ROC']:.2%}",
            f"{metrics_int['AUC-ROC']:.2%}",
            f"{metrics_embed['AUC-ROC']:.2%}"
        ]
    }
    
    df_compare = pd.DataFrame(metrics_comparison)
    print("\nCROSS-MODEL PERFORMANCE RESULTS:")
    print(df_to_markdown_simple(df_compare))
    print("\n")
    
    # Generate Confusion Matrices
    utils.plot_confusion_matrix(
        y_test_lin, 
        lr_model.predict(X_test_lin), 
        "Linear Regression (Logistic) Confusion Matrix", 
        os.path.join(figures_dir, "confusion_matrix_linear.png")
    )
    utils.plot_confusion_matrix(
        y_test_int, 
        trainer_int.predict(X_test_int), 
        "FCNN (Integer Encoding) Confusion Matrix", 
        os.path.join(figures_dir, "confusion_matrix_fcnn_integer.png")
    )
    utils.plot_confusion_matrix(
        y_test_int, 
        trainer_embed.predict(X_test_int), 
        "FCNN (Embedding Layers) Confusion Matrix", 
        os.path.join(figures_dir, "confusion_matrix_fcnn_embedding.png")
    )
    
    # Generate combined ROC Plot
    models_dict = {
        "Linear Regression (Logistic)": lr_model,
        "FCNN (Integer)": trainer_int,
        "FCNN (Embedding)": trainer_embed
    }
    X_test_dict = {
        "Linear Regression (Logistic)": X_test_lin,
        "FCNN (Integer)": X_test_int,
        "FCNN (Embedding)": X_test_int
    }
    utils.plot_roc_curves(
        models_dict, 
        X_test_dict, 
        y_test_lin, 
        os.path.join(figures_dir, "roc_curve_comparison.png")
    )
    
    # Generate bar chart performance comparison
    utils.plot_model_comparison(
        df_compare, 
        os.path.join(figures_dir, "metrics_comparison_bar.png")
    )
    
    # Save results as JSON
    results_json = {
        "dataset_stats": {
            "total_records": len(df_raw),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "overall_delay_risk_rate": float(df_raw['delay_risk'].mean())
        },
        "model_performance": {
            "linear_regression": {k: float(v) for k, v in metrics_lr.items()},
            "fcnn_integer": {k: float(v) for k, v in metrics_int.items()},
            "fcnn_embedding": {k: float(v) for k, v in metrics_embed.items()}
        }
    }
    
    utils.save_evaluation_report(
        results_json, 
        os.path.join(reports_dir, "model_comparison.json")
    )
    
    print("\n🎉 SUCCESS! All models trained and comparison visualizations generated!")
    print("You can run 'streamlit run app/streamlit_app.py' to launch the web interface.")
    print("======================================================================")

if __name__ == '__main__':
    main()
