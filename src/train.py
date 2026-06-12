"""
train.py
========
Model training, comparison, evaluation and persistence.

Models trained:
  1. Linear Regression
  2. Random Forest Regressor
  3. Gradient Boosting Regressor
  4. XGBoost Regressor

Metrics: MAE, MSE, RMSE, R²  (+ 5-fold CV R² on training set)
Best model is saved with its preprocessor via joblib.
"""

import json
import logging
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import HouseDataPreprocessor, TARGET

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(BASE_DIR, "data",   "housing.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "house_price_model.pkl")
REPORT_PATH   = os.path.join(MODEL_DIR, "model_performance_report.json")
FEAT_IMP_PATH = os.path.join(MODEL_DIR, "feature_importance.png")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Model catalogue ──────────────────────────────────────────────────────────
MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0,
    ),
}


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    logger.info(f"  {name:<22} MAE={mae:>10,.0f}  RMSE={rmse:>10,.0f}  R²={r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def plot_feature_importance(model, feature_names, path):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[-20:]
    feats  = [feature_names[i] for i in idx]
    values = imp[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feats)))
    bars = ax.barh(feats, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Top-20 Feature Importances", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, values):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot → {path}")


def train(data_path: str = DATA_PATH):
    logger.info("=" * 60)
    logger.info("   HOUSE PRICE PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    df = pd.read_csv(data_path)
    logger.info(f"Raw dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

    pre = HouseDataPreprocessor(k_features=20)
    X_train, y_train = pre.fit_transform(train_df)
    X_test,  y_test  = pre.transform(test_df)
    logger.info(f"Feature matrix — Train: {X_train.shape}, Test: {X_test.shape}")

    logger.info("\n── Training & Evaluating Models ──")
    results, trained = {}, {}
    for name, model in MODELS.items():
        logger.info(f"\nTraining {name} …")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        m = evaluate(model, X_test, y_test, name)
        m["TrainTime_s"] = round(elapsed, 2)
        m["CV_R2_mean"]  = round(float(cv.mean()), 4)
        m["CV_R2_std"]   = round(float(cv.std()),  4)
        results[name] = m
        trained[name] = model

    best_name   = max(results, key=lambda k: results[k]["R2"])
    best_model  = trained[best_name]
    best_m      = results[best_name]
    logger.info(f"\nBest: {best_name}  R²={best_m['R2']:.4f}  RMSE=${best_m['RMSE']:,.0f}")

    plot_feature_importance(best_model, pre.selected_features, FEAT_IMP_PATH)

    artifact = {
        "model":             best_model,
        "preprocessor":      pre,
        "model_name":        best_name,
        "selected_features": pre.selected_features,
        "metrics":           best_m,
        "model_version":     "1.0.0",
        "all_results":       results,
    }
    joblib.dump(artifact, MODEL_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")

    with open(REPORT_PATH, "w") as f:
        json.dump({"best_model": best_name, "model_version": "1.0.0",
                   "all_results": results,
                   "selected_features": pre.selected_features}, f, indent=2)
    logger.info(f"Report saved → {REPORT_PATH}")

    print("\n" + "="*72)
    print(f"{'Model':<22} {'MAE':>12} {'RMSE':>12} {'R²':>8} {'CV R²':>10}")
    print("-"*72)
    for name, m in results.items():
        star = " ★" if name == best_name else ""
        print(f"{name+star:<22} ${m['MAE']:>10,.0f} ${m['RMSE']:>10,.0f}"
              f" {m['R2']:>8.4f} {m['CV_R2_mean']:>10.4f}")
    print("="*72)
    return artifact


if __name__ == "__main__":
    train()
