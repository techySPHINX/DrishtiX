"""
DrishtiX Crowd Anomaly Detection — Model Trainer
=================================================
Trains three unsupervised anomaly detectors on synthetic crowd telemetry
and saves the best pipeline to disk for use in the Streamlit app.

Models trained:
  1. Isolation Forest     (primary — robust, fast, interpretable)
  2. Local Outlier Factor (LOF)    — density-based comparison
  3. One-Class SVM        (OCSVM)  — kernel-based comparison
  4. Elliptic Envelope    (EE)     — Gaussian covariance-based outlier detection

All four models are combined as a weighted ensemble in CrowdAnomalyDetector
(weights proportional to each model's F1 on the held-out test set).

Evaluation uses the synthetic ground-truth labels to compute:
  - Precision / Recall / F1 on anomaly class
  - ROC-AUC and Average Precision
  - Per anomaly-type breakdown

Outputs (written to models/):
  isolation_forest_pipeline.pkl   ← production pipeline
  lof_pipeline.pkl
  ocsvm_pipeline.pkl
  elliptic_env_pipeline.pkl       ← new in v2
  best_model.pkl                  ← best single model by F1
  training_report.json            ← full eval metrics

Usage:
  python train_model.py
  python train_model.py --contamination 0.04 --n-estimators 200
"""

import argparse
import json
import logging
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Feature columns fed to the models
# ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "crowd_density",
    "capacity_utilization",
    "density_delta",
    "density_delta_abs",
    "density_rolling_mean_5",
    "density_rolling_std_5",
    "density_rolling_mean_10",
    "density_rolling_std_10",
    "density_z_score",
    "flow_rate_in",
    "flow_rate_out",
    "net_flow",
    "flow_imbalance_ratio",
    "adjacent_zone_pressure",
    "exit_proximity_score",
    "dwell_time_minutes",
    "crowding_pressure",
    "perceptual_overload",
    "temperature_celsius",
    "noise_level_db",
    "hour_of_day",
    "minutes_since_event_start",
    "is_peak_hour",
]

MODEL_DIR  = Path("models")
DATA_PATH  = Path("data/crowd_telemetry_features.csv")


# ──────────────────────────────────────────────────────────────
# Data loader
# ──────────────────────────────────────────────────────────────
def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns (feature_df, X, y_true).
    Trains on NORMAL rows only; evaluates on full dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run:  python generate_dataset.py  first."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")

    # ── Validation ────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    X     = df[FEATURE_COLS].values.astype(np.float32)
    y     = df["is_anomaly"].values.astype(int)
    return df, X, y


# ──────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────
def build_isolation_forest(contamination: float, n_estimators: int,
                            random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("detector", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples="auto",
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
        )),
    ])


def build_lof(contamination: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("detector", LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            algorithm="ball_tree",
            n_jobs=-1,
            novelty=True,      # enables predict() on new data
        )),
    ])


def build_ocsvm(contamination: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("detector", OneClassSVM(
            kernel="rbf",
            nu=contamination,
            gamma="scale",
            cache_size=500,
        )),
    ])


def build_elliptic_env(contamination: float) -> Pipeline:
    """
    EllipticEnvelope fits a robust covariance estimate to the normal-only data
    and flags samples that violate the Mahalanobis-distance threshold.
    Works best when features are approximately Gaussian — the RobustScaler helps.
    """
    return Pipeline([
        ("scaler", RobustScaler()),
        ("detector", EllipticEnvelope(
            contamination=contamination,
            support_fraction=0.85,   # fraction of support points (robustness vs fit)
            random_state=42,
        )),
    ])


# ──────────────────────────────────────────────────────────────
# Train + evaluate
# ──────────────────────────────────────────────────────────────
def evaluate_model(pipeline, X: np.ndarray, y_true: np.ndarray,
                   model_name: str) -> dict:
    """
    Predict anomalies, convert sklearn convention (-1/+1) → (1/0),
    and compute classification metrics.
    """
    raw_preds = pipeline.predict(X)          # +1 = normal, -1 = anomaly
    y_pred    = np.where(raw_preds == -1, 1, 0)

    # Anomaly score (the lower, the more anomalous)
    if hasattr(pipeline.named_steps["detector"], "decision_function"):
        scores = pipeline.decision_function(X)
        # Invert so higher score → more anomalous
        anomaly_scores = -scores
    elif hasattr(pipeline.named_steps["detector"], "score_samples"):
        scores = pipeline.score_samples(X)
        anomaly_scores = -scores
    else:
        anomaly_scores = y_pred.astype(float)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc   = roc_auc_score(y_true, anomaly_scores)
        ap    = average_precision_score(y_true, anomaly_scores)
    except Exception:
        auc = ap = 0.0

    metrics = {
        "model_name":  model_name,
        "precision":   round(float(prec), 4),
        "recall":      round(float(rec),  4),
        "f1_score":    round(float(f1),   4),
        "roc_auc":     round(float(auc),  4),
        "avg_precision": round(float(ap), 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "n_predicted_anomalies": int(y_pred.sum()),
        "n_true_anomalies":      int(y_true.sum()),
    }

    logger.info(
        f"\n{'='*55}"
        f"\n  {model_name}"
        f"\n  Precision : {prec:.4f}  |  Recall : {rec:.4f}  |  F1 : {f1:.4f}"
        f"\n  ROC-AUC   : {auc:.4f}  |  AvgPrec : {ap:.4f}"
        f"\n  Predicted anomalies : {y_pred.sum():,} / {len(y_true):,}"
        f"\n{'='*55}"
    )
    return metrics


def per_type_breakdown(pipeline, df: pd.DataFrame,
                        X: np.ndarray) -> dict:
    """Recall per anomaly type (how well does the model catch each variant)."""
    raw_preds  = pipeline.predict(X)
    y_pred     = np.where(raw_preds == -1, 1, 0)
    df         = df.copy()
    df["_pred"] = y_pred

    results = {}
    for atype in df["anomaly_label"].unique():
        mask    = df["anomaly_label"] == atype
        n_total = int(mask.sum())
        n_caught = int((df.loc[mask, "_pred"] == 1).sum()) if atype != "NORMAL" else 0
        if atype == "NORMAL":
            fpr = float((df.loc[mask, "_pred"] == 1).mean())
            results[atype] = {"total": n_total, "false_pos_rate": round(fpr, 4)}
        else:
            rec = n_caught / n_total if n_total > 0 else 0.0
            results[atype] = {"total": n_total, "caught": n_caught,
                               "recall": round(rec, 4)}
    return results


# ──────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────
def save_pipeline(pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Saved → {path}")


def save_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report → {path}")


# ──────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────
def train(contamination: float = 0.03, n_estimators: int = 150,
          random_state: int = 42):

    logger.info("─" * 55)
    logger.info("DrishtiX Crowd Anomaly Detection — Model Training")
    logger.info("─" * 55)

    # 1. Load data
    df, X, y_true = load_data()
    anomaly_rate  = y_true.mean()
    logger.info(f"Anomaly rate in dataset: {anomaly_rate:.4f}  "
                f"({y_true.sum():,} / {len(y_true):,} events)")

    # Use provided contamination or auto-detect from data
    cont = contamination if contamination else float(np.clip(anomaly_rate, 0.01, 0.5))
    logger.info(f"Contamination parameter: {cont:.4f}")

    # 2. Train-test split — split by event so no data leakage
    event_ids  = df["event_id"].unique()
    n_train    = int(len(event_ids) * 0.8)
    train_eids = set(event_ids[:n_train])
    test_eids  = set(event_ids[n_train:])

    train_mask = df["event_id"].isin(train_eids)
    test_mask  = df["event_id"].isin(test_eids)

    X_train = X[train_mask]
    X_test  = X[test_mask]
    y_test  = y_true[test_mask]
    df_test = df[test_mask].reset_index(drop=True)

    # Train only on NORMAL samples (pure unsupervised protocol)
    normal_mask = (df["anomaly_label"] == "NORMAL") & train_mask
    X_train_normal = X[normal_mask]
    logger.info(f"Training on {X_train_normal.shape[0]:,} normal observations "
                f"({X_train.shape[1]} features)")

    # ── Model 1: Isolation Forest ──────────────────────────────
    logger.info("\n[1/4] Training Isolation Forest …")
    t0 = time.time()
    if_pipeline = build_isolation_forest(cont, n_estimators, random_state)
    if_pipeline.fit(X_train_normal)
    logger.info(f"      Done in {time.time()-t0:.1f}s")
    if_metrics  = evaluate_model(if_pipeline, X_test, y_test, "Isolation Forest")
    if_breakdown = per_type_breakdown(if_pipeline, df_test, X_test)

    # ── Model 2: LOF ───────────────────────────────────────────
    logger.info("\n[2/4] Training Local Outlier Factor …")
    t0 = time.time()
    lof_pipeline = build_lof(cont)
    lof_pipeline.fit(X_train_normal)
    logger.info(f"      Done in {time.time()-t0:.1f}s")
    lof_metrics  = evaluate_model(lof_pipeline, X_test, y_test, "Local Outlier Factor")
    lof_breakdown = per_type_breakdown(lof_pipeline, df_test, X_test)

    # ── Model 3: One-Class SVM ─────────────────────────────────
    logger.info("\n[3/4] Training One-Class SVM …")
    t0 = time.time()
    # OCSVM is slow on large sets — subsample 15k for training
    n_sub = min(15_000, X_train_normal.shape[0])
    idx_sub = np.random.default_rng(random_state).choice(
        X_train_normal.shape[0], n_sub, replace=False)
    ocsvm_pipeline = build_ocsvm(cont)
    ocsvm_pipeline.fit(X_train_normal[idx_sub])
    logger.info(f"      Done in {time.time()-t0:.1f}s")
    ocsvm_metrics  = evaluate_model(ocsvm_pipeline, X_test, y_test, "One-Class SVM")
    ocsvm_breakdown = per_type_breakdown(ocsvm_pipeline, df_test, X_test)

    # ── Model 4: Elliptic Envelope ─────────────────────────────
    logger.info("\n[4/4] Training Elliptic Envelope …")
    t0 = time.time()
    n_sub_ee = min(20_000, X_train_normal.shape[0])
    idx_ee = np.random.default_rng(random_state + 1).choice(
        X_train_normal.shape[0], n_sub_ee, replace=False)
    ee_pipeline = build_elliptic_env(cont)
    try:
        ee_pipeline.fit(X_train_normal[idx_ee])
        logger.info(f"      Done in {time.time()-t0:.1f}s")
        ee_metrics   = evaluate_model(ee_pipeline, X_test, y_test, "Elliptic Envelope")
        ee_breakdown = per_type_breakdown(ee_pipeline, df_test, X_test)
    except Exception as ee_err:
        logger.warning(f"Elliptic Envelope failed: {ee_err} — skipping.")
        ee_pipeline  = None
        ee_metrics   = {"model_name": "Elliptic Envelope", "f1_score": 0.0,
                        "precision": 0.0, "recall": 0.0, "roc_auc": 0.0,
                        "avg_precision": 0.0, "confusion_matrix": {},
                        "n_predicted_anomalies": 0, "n_true_anomalies": 0}
        ee_breakdown = {}

    # ── Pick best model by F1 ──────────────────────────────────
    candidates = [
        ("isolation_forest", if_pipeline,   if_metrics),
        ("lof",              lof_pipeline,  lof_metrics),
        ("ocsvm",            ocsvm_pipeline, ocsvm_metrics),
    ]
    if ee_pipeline is not None:
        candidates.append(("elliptic_env", ee_pipeline, ee_metrics))

    best_name, best_pipeline, best_metrics = max(
        candidates, key=lambda t: t[2]["f1_score"]
    )
    logger.info(f"\n🏆  Best model → {best_metrics['model_name']}  "
                f"(F1 = {best_metrics['f1_score']:.4f})")

    # ── Save models ────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_pipeline(if_pipeline,    MODEL_DIR / "isolation_forest_pipeline.pkl")
    save_pipeline(lof_pipeline,   MODEL_DIR / "lof_pipeline.pkl")
    save_pipeline(ocsvm_pipeline, MODEL_DIR / "ocsvm_pipeline.pkl")
    if ee_pipeline is not None:
        save_pipeline(ee_pipeline, MODEL_DIR / "elliptic_env_pipeline.pkl")
    save_pipeline(best_pipeline,  MODEL_DIR / "best_model.pkl")

    # Save feature column list alongside (for inference)
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    # ── Save training report ───────────────────────────────────
    report = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "dataset_rows": len(df),
        "train_events": list(train_eids),
        "test_events":  list(test_eids),
        "contamination": cont,
        "n_features": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "anomaly_rate_actual": float(anomaly_rate),
        "models": {
            "isolation_forest": {
                **if_metrics,
                "per_type": if_breakdown,
            },
            "lof": {
                **lof_metrics,
                "per_type": lof_breakdown,
            },
            "ocsvm": {
                **ocsvm_metrics,
                "per_type": ocsvm_breakdown,
            },
            "elliptic_env": {
                **ee_metrics,
                "per_type": ee_breakdown,
            },
        },
        "best_model": best_name,
    }
    save_report(report, MODEL_DIR / "training_report.json")

    logger.info(
        f"\n{'='*55}"
        f"\n  Training Complete!"
        f"\n  Models saved in: {MODEL_DIR.resolve()}"
        f"\n  Run the Streamlit app:  streamlit run streamlit_app.py"
        f"\n{'='*55}"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrishtiX anomaly model trainer")
    parser.add_argument("--contamination", type=float, default=0.03,
                        help="Expected anomaly fraction (0.01–0.5)")
    parser.add_argument("--n-estimators",  type=int,   default=150,
                        help="Number of trees in Isolation Forest")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()
    train(args.contamination, args.n_estimators, args.seed)
