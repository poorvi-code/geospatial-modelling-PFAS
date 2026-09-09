import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Paths
ROOT = Path(".")
MODEL_PATH = ROOT / "outputs" / "models" / "lgbm_calibrated.pkl"
DATA_PATH = ROOT / "dataset" / "pfas_golden.parquet"
OUT_DIR = ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_plots():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run 'python main.py' first to train the model!")
        return

    print("Loading data and model...")
    df = pd.read_parquet(DATA_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Prepare features, matching the training pipeline's valid-label filter.
    from implementation.train import FEATURE_COLS
    labels = pd.to_numeric(df["above_100_ng_l"], errors="coerce")
    valid_labels = labels.notna()
    X = df.loc[valid_labels, FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(-1).values
    y = labels.loc[valid_labels].astype(int).values
    
    # Predict
    print("Generating predictions...")
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # 1. Confusion Matrix
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Risk'], yticklabels=['Safe', 'Risk'])
    plt.title('Actual Project Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    print("Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSUCCESS! Your result images are saved in: {OUT_DIR.absolute()}")

if __name__ == "__main__":
    generate_plots()
