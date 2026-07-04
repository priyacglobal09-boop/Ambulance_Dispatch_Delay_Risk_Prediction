import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Set style for professional-looking plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150
})

def plot_training_history(history: dict, title: str, save_path: str):
    """
    Plots training and validation loss and accuracy over epochs.
    
    Parameters:
        history (dict): Dictionary with train_loss, val_loss, train_acc, val_acc lists.
        title (str): Title for the figure.
        save_path (str): File path to save the generated figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss subplot
    ax1.plot(epochs, history["train_loss"], 'b-o', label='Training Loss', linewidth=1.5, markersize=4)
    ax1.plot(epochs, history["val_loss"], 'r--s', label='Validation Loss', linewidth=1.5, markersize=4)
    ax1.set_title('BCE Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy subplot
    ax2.plot(epochs, history["train_acc"], 'b-o', label='Training Accuracy', linewidth=1.5, markersize=4)
    ax2.plot(epochs, history["val_acc"], 'r--s', label='Validation Accuracy', linewidth=1.5, markersize=4)
    ax2.set_title('Accuracy Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str):
    """Plots and saves a formatted confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    
    # Normalized confusion matrix for annotations in percentage
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = [f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt="", 
        cmap='Blues', 
        cbar=True,
        xticklabels=['Low Risk (0)', 'High Risk (1)'],
        yticklabels=['Low Risk (0)', 'High Risk (1)']
    )
    
    plt.title(title, pad=15)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_roc_curves(models_dict: dict, X_test_dict: dict, y_test: np.ndarray, save_path: str):
    """
    Plots ROC curves for all models on a single graph for comparison.
    
    Parameters:
        models_dict (dict): Dictionary of model name keys to trained model objects.
        X_test_dict (dict): Dictionary of model name keys to respective preprocessed test sets.
        y_test (np.ndarray): True target labels for the test set.
        save_path (str): File path to save the generated ROC curve plot.
    """
    plt.figure(figsize=(8, 7))
    
    colors = {
        'Linear Regression (Logistic)': '#1f77b4',
        'FCNN (Integer)': '#ff7f0e',
        'FCNN (Embedding)': '#2ca02c'
    }
    
    # Baseline random model line
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guessing (AUC = 0.50)')
    
    for model_name, model in models_dict.items():
        X_test = X_test_dict[model_name]
        probas = model.predict_proba(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        
        color = colors.get(model_name, '#7f7f7f')
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_model_comparison(metrics_df: pd.DataFrame, save_path: str):
    """
    Generates a bar chart comparison of key metrics (Accuracy and AUC-ROC) across models.
    """
    # Reshape for seaborn barplot
    metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
    
    # Filter for standard metrics
    metrics_melted = metrics_melted[metrics_melted['Metric'].isin(['Accuracy', 'AUC-ROC'])]
    
    # Convert string percentages like "82.34%" to float 0.8234 for plotting
    metrics_melted['Value'] = metrics_melted['Value'].apply(
        lambda x: float(x.replace('%', '')) / 100.0 if isinstance(x, str) and '%' in x else float(x)
    )
    
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=metrics_melted, 
        x='Model', 
        y='Value', 
        hue='Metric', 
        palette=['#4c72b0', '#55a868']
    )
    
    plt.title('Model Performance Metrics Comparison', pad=20)
    plt.xlabel('Model Type')
    plt.ylabel('Score (0.0 - 1.0)')
    plt.ylim([0.0, 1.05])
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0: # Avoid empty category bars
            ax.annotate(f'{height:.2%}',
                        (p.get_x() + p.get_width() / 2., height + 0.01),
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
                        
    plt.legend(loc='lower left')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def save_evaluation_report(metrics_dict: dict, save_path: str):
    """Saves complete evaluation metrics list to a formatted JSON report."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print(f"Detailed performance report written to {save_path}")

def get_actionable_recommendations(inputs: dict, delay_probability: float) -> list:
    """
    Analyzes prediction inputs and risk score to generate context-specific, 
    actionable operational recommendations for ambulance dispatchers.
    """
    recommendations = []
    
    if delay_probability < 0.3:
        recommendations.append("✅ Routine Dispatch: Standard protocol. Proceed with regular dispatch workflows.")
        recommendations.append("✓ Dispatch within 90 seconds. Inform crew of clear weather and low estimated delays.")
        return recommendations
        
    # High Risk cases (> 30%)
    if delay_probability >= 0.7:
        recommendations.append("🚨 CRITICAL DELAY WARNING: Urgent intervention required to mitigate severe dispatch lag.")
    else:
        recommendations.append("⚠️ MODERATE DELAY RISK: Implement active mitigation procedures.")
        
    # Distance criteria
    if inputs.get('distance_to_scene', 0) > 8.0:
        recommendations.append(f"• Scene is distant ({inputs['distance_to_scene']} miles). Alert nearest auxiliary zone to standby or consider air/mutual aid dispatch.")
        
    # Traffic criteria
    if inputs.get('traffic_density') == 'High':
        recommendations.append("• High traffic detected. Auto-route the ambulance using GPS live routing with active siren priority.")
        
    # Crew experience criteria
    if inputs.get('crew_experience_years', 10) <= 3:
        recommendations.append(f"• Crew experience is low ({inputs['crew_experience_years']} years). Assign an experienced mentor dispatcher to actively assist and monitor transit.")
        
    # Weather conditions
    if inputs.get('weather_conditions') in ['Snowy', 'Foggy']:
        recommendations.append(f"• Hazardous weather ({inputs['weather_conditions']}). Warn crew of road conditions. Advise slower, safer speeds if priority allows.")
        
    # Caller stress score
    if inputs.get('caller_stress_score', 5) >= 8:
        recommendations.append(f"• High caller stress/hysteria ({inputs['caller_stress_score']}/10). Initiate immediate dispatcher-assisted first aid and run standard pre-arrival instructions (PAI). Keep lines open.")
        
    # Priority level
    if inputs.get('priority_level') in ['Critical', 'High']:
        recommendations.append(f"• Case is a '{inputs['priority_level']}' priority. Cross-dispatch fire or first-responder police if the primary ambulance is delayed.")
        
    # Default recommendations
    if len(recommendations) <= 1:
        recommendations.append("• General optimization: Review route alternatives and check zone responder availability.")
        
    return recommendations

if __name__ == '__main__':
    print("Testing Utils Module...")
    # Test recommendations
    inputs = {
        'distance_to_scene': 11.2,
        'traffic_density': 'High',
        'crew_experience_years': 2,
        'weather_conditions': 'Snowy',
        'caller_stress_score': 9,
        'priority_level': 'Critical'
    }
    recs = get_actionable_recommendations(inputs, 0.85)
    print("\nActionable Recommendations Example:")
    for r in recs:
        print(r)
    print("\nUtils module tests passed!")
