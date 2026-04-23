import json
import logging
import pickle
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import ADASYN
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "dataset" / "pfas_golden.parquet"
MODELS_DIR  = ROOT / "outputs" / "models"

FEATURE_COLS = [
    # Substance
    "substance_ord", "is_long_chain", "carbon_chain_length", "is_sulfonyl",
    # Media
    "is_aquatic", "is_soil_based", "is_wastewater",
    # Temporal
    "year_normalized", "is_post_2018", "month",
    # Spatial lag
    "spatial_density_50km", "mean_log_value_50km", "nearest_training_point_km",
    # Source proximity
    "dist_to_airport_km",
]

TARGET_CLF = "above_100_ng_l"
TARGET_REG = "log_value"
GROUP_COL  = "spatial_block_id"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _cv_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        "pr_auc":    float(average_precision_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "brier":     float(brier_score_loss(y_true, y_prob)),
    }


def _spatial_cv(clf_factory, X, y, groups, n_splits=5, use_adasyn=True, label=""):
    """Run spatial block GroupKFold CV, return per-fold metrics."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    pbar = tqdm(enumerate(gkf.split(X, y, groups), 1), total=n_splits, desc=f"  {label} CV", leave=False)
    for fold, (tr_idx, val_idx) in pbar:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # ADASYN resampling on training fold only
        if use_adasyn and y_tr.sum() > 10 and (y_tr == 0).sum() > y_tr.sum():
            try:
                ada = ADASYN(sampling_strategy=0.3, random_state=42, n_neighbors=5)
                X_tr, y_tr = ada.fit_resample(X_tr, y_tr)
            except Exception:
                pass  # fall back to imbalanced data if ADASYN fails

        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_val)[:, 1]
        m = _cv_metrics(y_val, y_prob)
        m["fold"] = fold
        fold_metrics.append(m)
        pbar.set_postfix({"auc": f"{m['roc_auc']:.3f}"})

    return fold_metrics


# ---------------------------------------------------------------------------
# Optuna objective — LightGBM classification
# ---------------------------------------------------------------------------

def _lgbm_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
        "class_weight":     "balanced",
        "random_state":     42,
        "verbose":          -1,
        "n_jobs":           -1,
    }
    clf = LGBMClassifier(**params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
    )
    y_prob = clf.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_prob)


def _tune_lgbm(X, y, groups, n_trials=80):
    """Optuna-tune LightGBM on one held-out spatial fold."""
    gkf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(iter(gkf.split(X, y, groups)))

    # ADASYN on tuning fold
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    try:
        ada = ADASYN(sampling_strategy=0.3, random_state=42)
        X_tr, y_tr = ada.fit_resample(X_tr, y_tr)
    except Exception:
        pass

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    
    with tqdm(total=n_trials, desc="  Optuna tuning", leave=False) as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"best_pr": f"{study.best_value:.3f}"})
            
        study.optimize(
            lambda trial: _lgbm_objective(trial, X_tr, y_tr, X[val_idx], y[val_idx]),
            n_trials=n_trials,
            callbacks=[callback]
        )
        
    log.info(f"Best Optuna PR-AUC: {study.best_value:.4f}")
    return study.best_params


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {GOLDEN_PATH}. Run clean.py first."
        )

    log.info("Loading golden dataset ...")
    df = pd.read_parquet(GOLDEN_PATH)

    # Feature matrix for classification
    clf_df = df.dropna(subset=[TARGET_CLF]).copy()
    X      = clf_df[FEATURE_COLS].fillna(-1).values
    y      = clf_df[TARGET_CLF].values.astype(int)
    groups = clf_df[GROUP_COL].values if GROUP_COL in clf_df.columns else np.arange(len(clf_df))

    log.info(f"Classification dataset: {X.shape}  positive rate: {y.mean()*100:.1f}%")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("PFAS_Risk_Modelling")

    with mlflow.start_run(run_name=f"training_{int(time.time())}"):
        mlflow.log_param("n_samples",    len(X))
        mlflow.log_param("n_features",   len(FEATURE_COLS))

        # ---------------------------------------------------------------
        # 1. Optuna hyperparameter tuning
        # ---------------------------------------------------------------
        log.info("Step 1/4 — Optuna tuning LightGBM (80 trials) ...")
        best_params = _tune_lgbm(X, y, groups, n_trials=80)
        best_params.update({"class_weight": "balanced", "random_state": 42,
                            "verbose": -1, "n_jobs": -1})

        # ---------------------------------------------------------------
        # 2. Spatial CV — LightGBM
        # ---------------------------------------------------------------
        log.info("Step 2/4 — Spatial Block CV (LightGBM) ...")
        lgbm_factory = lambda: LGBMClassifier(**best_params)
        lgbm_cv = _spatial_cv(lgbm_factory, X, y, groups, use_adasyn=True, label="LGBM")
        lgbm_mean_auc = np.mean([m["roc_auc"] for m in lgbm_cv])
        log.info(f"  LightGBM CV  ROC-AUC={lgbm_mean_auc:.4f}")

        # ---------------------------------------------------------------
        # 3. Spatial CV — XGBoost & Random Forest
        # ---------------------------------------------------------------
        log.info("Step 3/4 — Baselines (XGBoost, Random Forest) ...")

        pos_w = float((y == 0).sum()) / max((y == 1).sum(), 1)
        xgb_factory = lambda: XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            scale_pos_weight=pos_w, random_state=42,
            eval_metric="aucpr", verbosity=0, n_jobs=-1,
        )
        xgb_cv = _spatial_cv(xgb_factory, X, y, groups, use_adasyn=False, label="XGB")
        xgb_auc = np.mean([m["roc_auc"] for m in xgb_cv])

        rf_factory = lambda: RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        rf_cv = _spatial_cv(rf_factory, X, y, groups, use_adasyn=False, label="RF")
        rf_auc = np.mean([m["roc_auc"] for m in rf_cv])
        log.info(f"  XGBoost={xgb_auc:.4f}  RF={rf_auc:.4f}")

        # ---------------------------------------------------------------
        # 4. Final training on full data + calibration
        # ---------------------------------------------------------------
        log.info("Step 4/4 — Final training on full dataset ...")

        # ADASYN full-data
        X_full, y_full = X, y
        try:
            ada = ADASYN(sampling_strategy=0.3, random_state=42)
            X_full, y_full = ada.fit_resample(X_full, y_full)
        except Exception:
            pass

        lgbm_final = LGBMClassifier(**best_params)
        lgbm_final.fit(X_full, y_full)

        # Isotonic calibration
        gkf = GroupKFold(n_splits=5)
        cal_tr, cal_val = next(iter(gkf.split(X, y, groups)))
        
        lgbm_final_for_cal = LGBMClassifier(**best_params)
        lgbm_final_for_cal.fit(X[cal_tr], y[cal_tr])
        
        # In newer scikit-learn, 'prefit' is replaced by wrapping in FrozenEstimator
        lgbm_cal = CalibratedClassifierCV(
            FrozenEstimator(lgbm_final_for_cal), method="isotonic"
        )
        lgbm_cal.fit(X[cal_val], y[cal_val])

        # LightGBM regressor
        reg_df  = df.dropna(subset=[TARGET_REG]).copy()
        X_reg   = reg_df[FEATURE_COLS].fillna(-1).values
        y_reg   = reg_df[TARGET_REG].values
        lgbm_reg = LGBMRegressor(**{k: v for k, v in best_params.items()
                                    if k not in ("class_weight",)})
        lgbm_reg.fit(X_reg, y_reg)

        # ---------------------------------------------------------------
        # Save artefacts
        # ---------------------------------------------------------------
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "lgbm_calibrated.pkl", "wb") as f:
            pickle.dump(lgbm_cal, f)
        with open(MODELS_DIR / "lgbm_regressor.pkl", "wb") as f:
            pickle.dump(lgbm_reg, f)
        with open(MODELS_DIR / "feature_schema.json", "w") as f:
            json.dump(FEATURE_COLS, f, indent=2)

        log.info(f"=== TRAINING COMPLETE (ROC-AUC={lgbm_mean_auc:.4f}) ===")


if __name__ == "__main__":
    train()
