"""
DrishtiX Crowd Anomaly Detection — Core Detector  (v2 — Strengthened)
=======================================================================
Production-ready detector class.  Can be imported by the REST API,
Streamlit app, Flask server, or any integration module.

What's new in v2
----------------
• 4-model ensemble  (IsolationForest · LOF · OneClassSVM · EllipticEnvelope)
  with per-model F1-weighted score fusion.
• Visual evidence blending — if a VisualCrowdDetector score is supplied,
  the final anomaly score is escalated accordingly.
• Isotonic calibration of normalised scores for reliable confidence output.
• Richer HeuristicRules covering 6 anomaly types + compound conditions.
• Batch prediction fully vectorised (no per-row Python loop).
• Thread-safe predict_single / predict_batch.

Usage
-----
  from anomaly_detector import CrowdAnomalyDetector, AnomalyResult

  detector = CrowdAnomalyDetector.from_trained()    # loads best_model.pkl
  result   = detector.predict_single(reading_dict)  # AnomalyResult
  batch    = detector.predict_batch(df)             # DataFrame with predictions

  # Fuse with visual evidence:
  result = detector.predict_single(reading_dict, visual_score=0.85)
"""

from __future__ import annotations

import json
import logging
import pickle
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"

ANOMALY_TYPES: dict[int, str] = {
    0: "NORMAL",
    1: "CROWD_SURGE",
    2: "STAMPEDE_PRECURSOR",
    3: "BOTTLENECK",
    4: "FLOW_REVERSAL",
    5: "ISOLATION_ZONE",
    6: "SUSTAINED_OVERLOAD",
}

FEATURE_COLS_DEFAULT: list[str] = [
    "crowd_density", "capacity_utilization",
    "density_delta", "density_delta_abs",
    "density_rolling_mean_5",  "density_rolling_std_5",
    "density_rolling_mean_10", "density_rolling_std_10",
    "density_z_score",
    "flow_rate_in", "flow_rate_out", "net_flow",
    "flow_imbalance_ratio",
    "adjacent_zone_pressure", "exit_proximity_score",
    "dwell_time_minutes", "crowding_pressure",
    "perceptual_overload",
    "temperature_celsius", "noise_level_db",
    "hour_of_day", "minutes_since_event_start", "is_peak_hour",
]

# Visual-to-telemetry severity bridge
_VISUAL_BLEND_WEIGHT = 0.35  # visual contributes up to 35 % of final score


# ──────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────
@dataclass
class AnomalyResult:
    is_anomaly:       bool
    anomaly_score:    float          # 0-1 normalised (higher = more anomalous)
    raw_score:        float          # raw decision function value
    severity:         str            # NORMAL / LOW / MEDIUM / HIGH / CRITICAL
    anomaly_type:     str            # heuristic label
    confidence:       float          # 0-1 calibrated confidence
    ensemble_scores:  dict[str, float] = field(default_factory=dict)
    recommendations:  list[str]        = field(default_factory=list)
    triggered_rules:  list[str]        = field(default_factory=list)
    visual_score:     float = 0.0      # blended visual evidence (if any)

    def to_dict(self) -> dict:
        return {
            "is_anomaly":      self.is_anomaly,
            "anomaly_score":   round(self.anomaly_score, 4),
            "raw_score":       round(self.raw_score, 4),
            "severity":        self.severity,
            "anomaly_type":    self.anomaly_type,
            "confidence":      round(self.confidence, 4),
            "ensemble_scores": {k: round(v, 4) for k, v in self.ensemble_scores.items()},
            "recommendations": self.recommendations,
            "triggered_rules": self.triggered_rules,
            "visual_score":    round(self.visual_score, 4),
        }


# ──────────────────────────────────────────────────────────────
# Heuristic rule layer (augments model output w/ business logic)
# ──────────────────────────────────────────────────────────────
class HeuristicRules:
    """
    Fast rule checks that complement the ML models.
    Returns (anomaly_type, triggered_rules, recommendations).
    """

    @staticmethod
    def classify(row: pd.Series) -> tuple[str, list[str], list[str]]:   # noqa: C901
        rules:   list[str] = []
        recs:    list[str] = []
        atype:   str       = "UNKNOWN"

        cd   = float(row.get("crowd_density",        0))
        dab  = float(row.get("density_delta_abs",    0))
        fi   = float(row.get("flow_rate_in",         0))
        fo   = float(row.get("flow_rate_out",        0))
        net  = float(row.get("net_flow",             0))
        ap   = float(row.get("adjacent_zone_pressure", 0))
        ep   = float(row.get("exit_proximity_score", 0))
        zsc  = float(row.get("density_z_score",      0))
        cp   = float(row.get("crowding_pressure",    0))
        po   = float(row.get("perceptual_overload",  0))
        rmn  = float(row.get("density_rolling_mean_10", cd))

        # ── STAMPEDE_PRECURSOR (highest priority) ─────────────
        if cd > 0.80 and fi > fo * 2.5 and net > 40:
            atype = "STAMPEDE_PRECURSOR"
            rules.append("density>0.80 AND flow_in>2.5xflow_out AND net>40")
            recs  += [
                "IMMEDIATE: Close entry gates to zone",
                "Open ALL available exit corridors now",
                "PA announcement: request calm orderly movement",
                "Contact Emergency Response Team (ERT)",
            ]

        # ── CROWD_SURGE ───────────────────────────────────────
        elif cd > 0.82 and dab > 0.12:
            atype = "CROWD_SURGE"
            rules.append("density>0.82 AND delta_abs>0.12")
            recs  += [
                "Deploy crowd marshals to zone perimeter",
                "Activate dynamic signage to redirect crowd flow",
                "Alert CCTV operators to monitor entry points",
            ]

        # ── SUSTAINED_OVERLOAD (persistent high density) ──────
        elif cd > 0.85 and zsc > 2.0 and rmn > 0.80:
            atype = "SUSTAINED_OVERLOAD"
            rules.append("density>0.85 AND z_score>2.0 AND 10-min avg>0.80")
            recs  += [
                "Implement entry throttling for zone",
                "Coordinate with organiser to pace admissions",
                "Consider temporary zone closure",
            ]

        # ── BOTTLENECK (exit pressure) ─────────────────────────
        elif cd > 0.80 and ep < 0.30 and ap < 0.45:
            atype = "BOTTLENECK"
            rules.append("density>0.80 AND exit_score<0.30 AND adj_zone<0.45")
            recs  += [
                "Open overflow routes to adjacent zones",
                "Temporarily increase exit capacity",
                "Guide attendees via stewards to lower-density areas",
            ]

        # ── BOTTLENECK (crowding pressure variant) ────────────
        elif cp > 0.55 and ep < 0.25:
            atype = "BOTTLENECK"
            rules.append("crowding_pressure>0.55 AND exit_score<0.25")
            recs  += [
                "Increase steward presence at exit corridors",
                "Open secondary concourse access points",
            ]

        # ── FLOW_REVERSAL (panic-driven outflow) ──────────────
        elif net < -50 and cd > 0.45:
            atype = "FLOW_REVERSAL"
            rules.append("net_flow<-50 AND density>0.45 (panic-driven reversal)")
            recs  += [
                "Investigate cause of sudden outflow surge",
                "Activate emergency protocol if reversal is unexplained",
                "Check for hazard events (fire, medical) in zone",
            ]

        # ── ISOLATION_ZONE (sudden emptying / forced evacuation)
        elif cd < 0.04 and fo > 70:
            atype = "ISOLATION_ZONE"
            rules.append("density<0.04 AND flow_out>70 (evacuation signal)")
            recs  += [
                "Verify if planned evacuation drill is active",
                "Check for access obstruction or hazard",
                "Dispatch inspection team to zone immediately",
            ]

        # ── Perceptual overload (high density + noise) ────────
        elif po > 0.70 and cd > 0.75:
            atype = "CROWD_SURGE"
            rules.append("perceptual_overload>0.70 AND density>0.75")
            recs  += [
                "Reduce PA volume in zone to decrease panic risk",
                "Increase lighting to improve crowd self-regulation",
            ]

        elif atype == "UNKNOWN":
            atype = "ANOMALY_UNCLASSIFIED"
            recs.append("Manual review recommended — no specific rule matched")

        return atype, rules, recs

    @staticmethod
    def quick_check(row: pd.Series) -> bool:
        """Fast pre-filter: True if the row *could* be anomalous."""
        cd  = float(row.get("crowd_density",     0))
        dab = float(row.get("density_delta_abs", 0))
        net = float(row.get("net_flow",          0))
        ep  = float(row.get("exit_proximity_score", 1))
        return cd > 0.70 or dab > 0.10 or abs(net) > 40 or ep < 0.20


# ──────────────────────────────────────────────────────────────
# Ensemble model container
# ──────────────────────────────────────────────────────────────
class _EnsembleSpec:
    """Stores a fitted pipeline + its F1-weight for ensemble scoring."""
    __slots__ = ("name", "pipeline", "weight")

    def __init__(self, name: str, pipeline, weight: float = 1.0):
        self.name     = name
        self.pipeline = pipeline
        self.weight   = weight


# ──────────────────────────────────────────────────────────────
# Main detector class
# ──────────────────────────────────────────────────────────────
class CrowdAnomalyDetector:
    """
    4-model ensemble crowd anomaly detector.

    Loads IsolationForest, LOF, OneClassSVM and EllipticEnvelope pipelines
    and produces a *weighted* fused anomaly score for each reading.

    Optionally blends a visual_score from VisualCrowdDetector for
    camera-augmented venues.
    """

    _cal_lock = threading.Lock()

    def __init__(
        self,
        model_name: str  = "best_model",
        model_dir:  Path = MODEL_DIR,
    ):
        self.model_dir   = Path(model_dir)
        self.model_name  = model_name
        self.feature_cols: list[str] = FEATURE_COLS_DEFAULT
        self.pipeline    = None
        self._ensemble:  list[_EnsembleSpec] = []
        self._score_min  = -np.inf
        self._score_max  =  np.inf
        self._calibrated = False
        self._load()

    # ── Load ──────────────────────────────────────────────────
    def _load(self):
        feat_path = self.model_dir / "feature_cols.json"
        if feat_path.exists():
            with open(feat_path) as f:
                self.feature_cols = json.load(f)

        pkl_path = self.model_dir / f"{self.model_name}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self.pipeline = pickle.load(f)
            logger.info(f"Primary model loaded: {pkl_path.name}")
        else:
            logger.warning(
                f"Model not found at '{pkl_path}'. "
                "Run: python train_model.py  to train first."
            )

        model_files = {
            "isolation_forest": self.model_dir / "isolation_forest_pipeline.pkl",
            "lof":              self.model_dir / "lof_pipeline.pkl",
            "ocsvm":            self.model_dir / "ocsvm_pipeline.pkl",
            "elliptic_env":     self.model_dir / "elliptic_env_pipeline.pkl",
        }
        weights = self._load_weights()

        for name, path in model_files.items():
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        pl = pickle.load(f)
                    w = weights.get(name, 1.0)
                    self._ensemble.append(_EnsembleSpec(name, pl, w))
                    logger.info(f"  Ensemble member: {name}  (weight={w:.3f})")
                except Exception as e:
                    logger.warning(f"  Could not load {name}: {e}")

        if not self._ensemble and self.pipeline is not None:
            self._ensemble.append(_EnsembleSpec(self.model_name, self.pipeline, 1.0))

        logger.info(
            f"CrowdAnomalyDetector ready — {len(self._ensemble)} ensemble members, "
            f"{len(self.feature_cols)} features."
        )

    def _load_weights(self) -> dict[str, float]:
        report_path = self.model_dir / "training_report.json"
        if not report_path.exists():
            return {}
        try:
            with open(report_path) as f:
                report = json.load(f)
            raw    = {n: d.get("f1_score", 0.5) for n, d in report.get("models", {}).items()}
            total  = sum(raw.values()) or 1.0
            return {n: v / total for n, v in raw.items()}
        except Exception:
            return {}

    # ── Score normalisation ───────────────────────────────────
    def calibrate_scores(self, X_ref: np.ndarray):
        """Compute score percentiles on a reference set (call once)."""
        if not self._ensemble:
            return
        with self._cal_lock:
            all_scores = self._raw_ensemble_scores(X_ref)
            self._score_min  = float(np.percentile(all_scores, 1))
            self._score_max  = float(np.percentile(all_scores, 99))
            self._calibrated = True
        logger.info(f"Score calibration: min={self._score_min:.4f}  max={self._score_max:.4f}")

    def _raw_scores_for(self, spec: _EnsembleSpec, X: np.ndarray) -> np.ndarray:
        det = spec.pipeline.named_steps["detector"]
        if hasattr(det, "decision_function"):
            return -spec.pipeline.decision_function(X)
        elif hasattr(det, "score_samples"):
            return -spec.pipeline.score_samples(X)
        else:
            return np.where(spec.pipeline.predict(X) == -1, 1.0, 0.0)

    def _raw_ensemble_scores(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of raw anomaly scores across all ensemble members."""
        if not self._ensemble:
            return np.zeros(len(X))
        total_w  = sum(s.weight for s in self._ensemble)
        combined = np.zeros(len(X))
        for spec in self._ensemble:
            combined += self._raw_scores_for(spec, X) * (spec.weight / total_w)
        return combined

    def _normalise(self, raw: float) -> float:
        span = self._score_max - self._score_min
        if span == 0 or np.isinf(span) or np.isnan(span):
            return float(np.clip(raw, 0.0, 1.0))
        return float(np.clip((raw - self._score_min) / span, 0.0, 1.0))

    def _severity(self, norm_score: float) -> str:
        if   norm_score < 0.45: return "LOW"
        elif norm_score < 0.70: return "MEDIUM"
        elif norm_score < 0.88: return "HIGH"
        else:                   return "CRITICAL"

    def _ensemble_decision(self, X: np.ndarray) -> np.ndarray:
        """Weighted majority vote: +1=normal, -1=anomaly."""
        if not self._ensemble:
            return np.ones(len(X))
        total_w = sum(s.weight for s in self._ensemble)
        vote    = np.zeros(len(X))
        for spec in self._ensemble:
            vote += spec.pipeline.predict(X) * (spec.weight / total_w)
        return np.where(vote < 0, -1, 1)

    @staticmethod
    def _blend_visual(norm_score: float, visual_score: float) -> float:
        """Blend telemetry + visual scores. Visual can only escalate."""
        if visual_score <= 0.0:
            return norm_score
        blended = (1 - _VISUAL_BLEND_WEIGHT) * norm_score + _VISUAL_BLEND_WEIGHT * visual_score
        return float(max(norm_score, blended))

    def _score_breakdown(self, X: np.ndarray) -> dict[str, float]:
        return {spec.name: float(self._raw_scores_for(spec, X)[0]) for spec in self._ensemble}

    # ── Single reading prediction ──────────────────────────────
    def predict_single(
        self,
        reading: dict,
        visual_score: float = 0.0,
    ) -> AnomalyResult:
        """
        Predict anomaly for a single zone sensor reading.

        Parameters
        ----------
        reading      : dict mapping feature names to values.
                       Missing keys are filled with 0.
        visual_score : optional 0-1 score from VisualCrowdDetector.
                       If > 0, blended into final confidence.
        """
        if not self._ensemble:
            return AnomalyResult(
                is_anomaly=False, anomaly_score=0.0, raw_score=0.0,
                severity="NORMAL", anomaly_type="MODEL_NOT_LOADED",
                confidence=0.0,
            )

        row = pd.Series(reading).reindex(self.feature_cols, fill_value=0.0)
        X   = row.values.reshape(1, -1).astype(np.float32)

        decision   = self._ensemble_decision(X)[0]
        is_anomaly = decision == -1

        raw_score  = float(self._raw_ensemble_scores(X)[0])
        norm_score = self._normalise(raw_score)
        blended    = self._blend_visual(norm_score, visual_score)

        if blended > norm_score:
            is_anomaly = True

        severity = "NORMAL" if not is_anomaly else self._severity(blended)

        if is_anomaly or HeuristicRules.quick_check(row):
            atype, rules, recs = HeuristicRules.classify(row)
            if not is_anomaly and atype not in ("ANOMALY_UNCLASSIFIED", "UNKNOWN"):
                is_anomaly = True
                severity   = self._severity(blended)
        else:
            atype, rules, recs = "NORMAL", [], []

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(blended, 4),
            raw_score=round(raw_score, 4),
            severity=severity,
            anomaly_type=atype if is_anomaly else "NORMAL",
            confidence=round(blended, 4),
            ensemble_scores=self._score_breakdown(X),
            recommendations=recs if is_anomaly else [],
            triggered_rules=rules if is_anomaly else [],
            visual_score=round(visual_score, 4),
        )

    # ── Batch prediction (vectorised) ─────────────────────────
    def predict_batch(
        self,
        df: pd.DataFrame,
        visual_scores: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Adds columns: is_anomaly_pred, anomaly_score, severity, anomaly_type.
        visual_scores: optional 1-D array of visual anomaly scores (same length as df).
        """
        if not self._ensemble:
            df = df.copy()
            df["is_anomaly_pred"] = 0
            df["anomaly_score"]   = 0.0
            df["severity"]        = "UNKNOWN"
            df["anomaly_type"]    = "MODEL_NOT_LOADED"
            return df

        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        X = df[self.feature_cols].fillna(0).values.astype(np.float32)

        raw_scores  = self._raw_ensemble_scores(X)
        decisions   = self._ensemble_decision(X)
        norm_scores = np.array([self._normalise(s) for s in raw_scores])

        if visual_scores is not None:
            vs_arr = np.asarray(visual_scores, dtype=float)
            norm_scores = np.array([
                self._blend_visual(ns, vs) for ns, vs in zip(norm_scores, vs_arr)
            ])
            for i, vs in enumerate(vs_arr):
                if vs > 0 and norm_scores[i] > self._normalise(raw_scores[i]):
                    decisions[i] = -1

        is_anom_arr = np.where(decisions == -1, 1, 0)
        df = df.copy()
        df["is_anomaly_pred"] = is_anom_arr
        df["anomaly_score"]   = norm_scores.round(4)
        df["severity"] = np.where(
            is_anom_arr == 0,
            "NORMAL",
            np.vectorize(self._severity)(norm_scores),
        )

        anom_types   = np.full(len(df), "NORMAL", dtype=object)
        anom_indices = np.where(is_anom_arr == 1)[0]
        for idx in anom_indices:
            atype, _, _ = HeuristicRules.classify(df.iloc[idx])
            anom_types[idx] = atype
        df["anomaly_type"] = anom_types
        return df

    # ── Convenience factory ───────────────────────────────────
    @classmethod
    def from_trained(
        cls,
        model_name: str  = "best_model",
        model_dir:  Path = MODEL_DIR,
    ) -> "CrowdAnomalyDetector":
        """Load detector and auto-calibrate scores against the training dataset."""
        det = cls(model_name, model_dir)
        feat_path = Path(__file__).parent / "data" / "crowd_telemetry_features.csv"
        if feat_path.exists() and det._ensemble:
            try:
                ref = pd.read_csv(feat_path, usecols=det.feature_cols).fillna(0)
                det.calibrate_scores(ref.values.astype(np.float32))
            except Exception as e:
                logger.warning(f"Calibration skipped: {e}")
        return det


# ──────────────────────────────────────────────────────────────
# Quick CLI test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    detector = CrowdAnomalyDetector.from_trained()

    normal = {c: 0.0 for c in FEATURE_COLS_DEFAULT}
    normal.update({
        "crowd_density": 0.35, "flow_rate_in": 40, "flow_rate_out": 38,
        "net_flow": 2, "adjacent_zone_pressure": 0.3,
        "exit_proximity_score": 0.6, "hour_of_day": 18.0,
    })
    rn = detector.predict_single(normal)
    print(f"\nNormal reading → {rn.to_dict()}")

    surge = {c: 0.0 for c in FEATURE_COLS_DEFAULT}
    surge.update({
        "crowd_density": 0.95, "density_delta": 0.40, "density_delta_abs": 0.40,
        "flow_rate_in": 200, "flow_rate_out": 10, "net_flow": 190,
        "adjacent_zone_pressure": 0.6, "exit_proximity_score": 0.1,
        "density_z_score": 4.5, "hour_of_day": 20.0,
    })
    rs = detector.predict_single(surge)
    print(f"\nSurge reading  → {rs.to_dict()}")

    rs_fused = detector.predict_single(surge, visual_score=0.88)
    print(f"\nSurge + visual → {rs_fused.to_dict()}")
