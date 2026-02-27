"""
DrishtiX Visual Crowd Anomaly Detector
=======================================
CLIP-based scene classifier adapted for crowd-safety analysis.
Migrated and strengthened from the original anomaly-detection module.

Capabilities
------------
• Classifies single images / individual video frames into crowd safety labels.
• Temporal smoother accumulates per-label probability across a rolling window
  so a single noisy frame cannot trigger a false alert.
• Returns a VisualAnomalyResult that maps CLIP label → severity bucket the same
  way CrowdAnomalyDetector does, enabling safe score-fusion in UnifiedDetector.

Dependencies: torch, clip (pip install git+https://github.com/openai/CLIP.git)
              Pillow, opencv-python, numpy, pyyaml

Usage
-----
  from visual_detector import VisualCrowdDetector, VisualAnomalyResult

  det = VisualCrowdDetector()                        # loads ViT-B/32 by default
  result = det.predict(rgb_frame_as_ndarray)         # VisualAnomalyResult
  print(result.to_dict())
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Default settings path (crowd-specific) alongside this file
# ──────────────────────────────────────────────────────────────
_DEFAULT_SETTINGS = Path(__file__).parent / "settings.yaml"

# ──────────────────────────────────────────────────────────────
# Label → safety category mapping
# Normal categories do NOT trigger an alert.
# ──────────────────────────────────────────────────────────────
LABEL_CATEGORY: dict[str, str] = {
    # ── Normal crowd states ──────────────────────────────
    "people walking normally in a crowd":           "NORMAL",
    "people standing in a queue or line":           "NORMAL",
    "sparse crowd in an open venue":                "NORMAL",
    "crowd watching an event or performance":       "NORMAL",
    "crowd dispersing calmly after an event":       "NORMAL",
    "people sitting in stadium seats":              "NORMAL",
    "group of people talking or socialising":       "NORMAL",
    "empty venue or concert hall":                  "NORMAL",
    # ── Crowd anomaly states ─────────────────────────────
    "dense overcrowded area with no space to move": "CROWD_SURGE",
    "crowd pushing and crushing at entry gate":     "CROWD_SURGE",
    "stampede or mass panic in a crowd":            "STAMPEDE_PRECURSOR",
    "people running in panic or fear":              "STAMPEDE_PRECURSOR",
    "bottleneck crowd blocked at an exit":          "BOTTLENECK",
    "crowd blocked at a narrow corridor":           "BOTTLENECK",
    "sudden reversal of crowd movement direction":  "FLOW_REVERSAL",
    "people rushing back against crowd flow":       "FLOW_REVERSAL",
    "section of venue suddenly empty or evacuated": "ISOLATION_ZONE",
    "fight or physical violence in a crowd":        "FIGHT_VIOLENCE",
    "fire or smoke visible in a crowd":             "FIRE_HAZARD",
    "medical emergency in a crowd":                 "MEDICAL_EMERGENCY",
    "person fallen or trampled in crowd":           "MEDICAL_EMERGENCY",
}

# category → severity
CATEGORY_SEVERITY: dict[str, str] = {
    "NORMAL":             "NORMAL",
    "CROWD_SURGE":        "HIGH",
    "STAMPEDE_PRECURSOR": "CRITICAL",
    "BOTTLENECK":         "HIGH",
    "FLOW_REVERSAL":      "MEDIUM",
    "ISOLATION_ZONE":     "MEDIUM",
    "FIGHT_VIOLENCE":     "HIGH",
    "FIRE_HAZARD":        "CRITICAL",
    "MEDICAL_EMERGENCY":  "CRITICAL",
}

# category → numeric anomaly code (for unified scoring)
CATEGORY_CODE: dict[str, int] = {
    "NORMAL":             0,
    "CROWD_SURGE":        1,
    "STAMPEDE_PRECURSOR": 2,
    "BOTTLENECK":         3,
    "FLOW_REVERSAL":      4,
    "ISOLATION_ZONE":     5,
    "FIGHT_VIOLENCE":     6,
    "FIRE_HAZARD":        7,
    "MEDICAL_EMERGENCY":  8,
}

NORMAL_CATEGORIES = {"NORMAL"}


# ──────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────
@dataclass
class VisualAnomalyResult:
    is_anomaly:     bool
    label:          str            # raw CLIP label text
    category:       str            # CROWD_SURGE / NORMAL / …
    severity:       str            # NORMAL / LOW / MEDIUM / HIGH / CRITICAL
    confidence:     float          # cosine similarity (0-1)
    smoothed_score: float          # temporal-window averaged anomaly score
    recommendations: list[str]    = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_anomaly":      self.is_anomaly,
            "label":           self.label,
            "category":        self.category,
            "severity":        self.severity,
            "confidence":      round(self.confidence, 4),
            "smoothed_score":  round(self.smoothed_score, 4),
            "recommendations": self.recommendations,
            "source":          "visual",
        }


# ──────────────────────────────────────────────────────────────
# Per-category action recommendations
# ──────────────────────────────────────────────────────────────
_RECOMMENDATIONS: dict[str, list[str]] = {
    "CROWD_SURGE": [
        "Deploy crowd marshals to zone perimeter immediately",
        "Activate dynamic signage to redirect flow",
        "Alert CCTV operators to monitor entry points",
    ],
    "STAMPEDE_PRECURSOR": [
        "IMMEDIATE: Close entry gates to affected zone",
        "Open all available exit corridors",
        "Broadcast PA announcement requesting orderly movement",
        "Contact Emergency Response Team (ERT)",
    ],
    "BOTTLENECK": [
        "Open overflow routes to adjacent zones",
        "Temporarily increase exit capacity",
        "Guide attendees via stewards to lower-density areas",
    ],
    "FLOW_REVERSAL": [
        "Investigate cause of sudden crowd reversal",
        "Activate emergency protocol if reversal is unexplained",
        "Check for hazard events (fire, medical) in vicinity",
    ],
    "ISOLATION_ZONE": [
        "Verify if planned evacuation drill is active",
        "Check for access obstruction or hazard near zone",
        "Dispatch inspection team to zone immediately",
    ],
    "FIGHT_VIOLENCE": [
        "Dispatch security personnel to incident location",
        "Isolate combatants from crowd",
        "Notify police if situation escalates",
    ],
    "FIRE_HAZARD": [
        "IMMEDIATE: Activate fire alarm and evacuation protocol",
        "Alert fire brigade — call emergency services",
        "Guide crowd to nearest exits away from smoke/fire",
    ],
    "MEDICAL_EMERGENCY": [
        "Dispatch first-aid team to detected location",
        "Clear a path through crowd for emergency responders",
        "Alert nearest medical station",
    ],
}


# ──────────────────────────────────────────────────────────────
# Temporal smoother — smooths anomaly probability over a window
# ──────────────────────────────────────────────────────────────
class TemporalSmoother:
    """
    Keeps a rolling window of raw anomaly scores (0/1 per frame).
    Returns True when the moving average exceeds `threshold`.
    Prevents single noisy frames from triggering false alerts.
    """

    def __init__(self, window: int = 8, threshold: float = 0.4):
        self.window    = window
        self.threshold = threshold
        self._buffer: deque[float] = deque(maxlen=window)

    def update(self, is_anomaly: bool, confidence: float) -> tuple[float, bool]:
        """Push a new observation and return (smoothed_score, triggered)."""
        score = confidence if is_anomaly else 0.0
        self._buffer.append(score)
        smoothed = float(np.mean(self._buffer)) if self._buffer else 0.0
        triggered = smoothed >= self.threshold
        return smoothed, triggered

    def reset(self):
        self._buffer.clear()


# ──────────────────────────────────────────────────────────────
# Main visual detector class
# ──────────────────────────────────────────────────────────────
class VisualCrowdDetector:
    """
    CLIP-based crowd anomaly detector for images and video frames.

    Parameters
    ----------
    settings_path : path to YAML config (model name, threshold, labels).
                    Falls back to built-in LABEL_CATEGORY if not found.
    smoother_window : temporal smoothing window (frames). 0 = disabled.
    smoother_threshold : fraction of window that must be anomalous to trigger.
    """

    def __init__(
        self,
        settings_path: str | Path = _DEFAULT_SETTINGS,
        smoother_window: int     = 8,
        smoother_threshold: float = 0.40,
    ):
        self._model       = None
        self._preprocess  = None
        self._text_features = None
        self._labels: list[str] = []
        self._conf_threshold: float = 0.22
        self._device      = "cpu"
        self._loaded      = False
        self.smoother     = TemporalSmoother(smoother_window, smoother_threshold)
        self._load(settings_path)

    # ── Internal loader ───────────────────────────────────────
    def _load(self, settings_path: str | Path):
        """Lazy-load CLIP model so import doesn't crash if torch is absent."""
        try:
            import clip
            import torch
        except ImportError:
            logger.warning(
                "CLIP / torch not installed. VisualCrowdDetector will return "
                "NORMAL for all inputs. Install with:\n"
                "  pip install torch torchvision\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )
            return

        settings_path = Path(settings_path)
        model_name = "ViT-B/32"

        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    cfg = yaml.safe_load(f)
                model_name           = cfg["model-settings"].get("model-name", model_name)
                self._conf_threshold = cfg["model-settings"].get("prediction-threshold", 0.22)
                raw_labels           = cfg["label-settings"].get("labels", [])
                if raw_labels:
                    # Override built-in label list with YAML-defined ones
                    self._labels = raw_labels
            except Exception as e:
                logger.warning(f"settings.yaml parse error: {e} — using built-in labels.")

        if not self._labels:
            self._labels = list(LABEL_CATEGORY.keys())

        self._device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP {model_name} on {self._device} …")

        try:
            self._model, self._preprocess = clip.load(model_name, device=self._device)
            self._text_features = self._encode_labels(
                ["a photo of " + lbl for lbl in self._labels]
            )
            self._loaded = True
            logger.info(f"VisualCrowdDetector ready — {len(self._labels)} crowd labels.")
        except Exception as e:
            logger.error(f"CLIP load failed: {e}")

    @property
    def is_ready(self) -> bool:
        return self._loaded

    # ── Text encoding ─────────────────────────────────────────
    def _encode_labels(self, texts: list[str]):
        import clip
        import torch
        with torch.no_grad():
            tokens  = clip.tokenize(texts).to(self._device)
            feats   = self._model.encode_text(tokens)
        return feats

    # ── Image encoding ────────────────────────────────────────
    def _encode_image(self, image: np.ndarray):
        import torch
        from PIL import Image as PILImage
        with torch.no_grad():
            pil = PILImage.fromarray(image).convert("RGB")
            tf  = self._preprocess(pil).unsqueeze(0).to(self._device)
            feats = self._model.encode_image(tf)
        return feats

    # ── Core prediction ───────────────────────────────────────
    def predict(
        self,
        image: np.ndarray,
        use_smoother: bool = True,
    ) -> VisualAnomalyResult:
        """
        Predict crowd anomaly category for a single RGB image/frame.

        Parameters
        ----------
        image : np.ndarray  RGB image (H×W×3, uint8)
        use_smoother : apply temporal smoothing (recommended for video streams)

        Returns
        -------
        VisualAnomalyResult
        """
        if not self._loaded:
            return VisualAnomalyResult(
                is_anomaly=False, label="MODEL_NOT_LOADED",
                category="NORMAL", severity="NORMAL",
                confidence=0.0, smoothed_score=0.0,
            )

        import torch
        with torch.no_grad():
            img_feats = self._encode_image(image)
            txt_feats = self._text_features

            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            similarity = (img_feats @ txt_feats.T)[0]   # (n_labels,)
            best_idx   = int(similarity.argmax().cpu().item())
            confidence = float(similarity[best_idx].cpu().item())

        label = self._labels[best_idx]

        # Below threshold → unknown / normal (mirrors original model behaviour)
        if confidence < self._conf_threshold:
            label    = "scene unclear"
            category = "NORMAL"
        else:
            category = LABEL_CATEGORY.get(label, "NORMAL")

        severity     = CATEGORY_SEVERITY.get(category, "NORMAL")
        is_anomaly   = category not in NORMAL_CATEGORIES
        recs         = _RECOMMENDATIONS.get(category, [])

        smoothed, triggered = self.smoother.update(is_anomaly, confidence) \
            if use_smoother else (confidence if is_anomaly else 0.0, is_anomaly)

        # Smoother can escalate or suppress
        if use_smoother:
            is_anomaly = triggered

        return VisualAnomalyResult(
            is_anomaly=is_anomaly,
            label=label,
            category=category,
            severity=severity if is_anomaly else "NORMAL",
            confidence=confidence,
            smoothed_score=smoothed if use_smoother else confidence,
            recommendations=recs if is_anomaly else [],
        )

    # ── Batch prediction over video frames ────────────────────
    def predict_video(
        self,
        frames: list[np.ndarray],
        frame_stride: int = 1,
        use_smoother: bool = True,
    ) -> list[VisualAnomalyResult]:
        """
        Run prediction on a list of RGB frames (e.g., from cv2.VideoCapture).
        frame_stride=2 processes every other frame for speed.
        """
        self.smoother.reset()
        results: list[VisualAnomalyResult] = []
        for i, frame in enumerate(frames):
            if i % frame_stride != 0:
                continue
            results.append(self.predict(frame, use_smoother=use_smoother))
        return results

    # ── Aggregate video summary ───────────────────────────────
    def summarise_video(
        self,
        frames: list[np.ndarray],
        frame_stride: int = 2,
    ) -> dict:
        """
        Returns a summary dict with overall severity, dominant anomaly type,
        peak confidence, and anomaly frame count.
        """
        results = self.predict_video(frames, frame_stride=frame_stride)
        n_anomaly = sum(1 for r in results if r.is_anomaly)
        peak_conf = max((r.confidence for r in results), default=0.0)
        categories = [r.category for r in results if r.is_anomaly]
        dominant = max(set(categories), key=categories.count) if categories else "NORMAL"
        severity = CATEGORY_SEVERITY.get(dominant, "NORMAL") if categories else "NORMAL"

        return {
            "total_frames":    len(results),
            "anomaly_frames":  n_anomaly,
            "anomaly_rate":    round(n_anomaly / max(len(results), 1), 4),
            "dominant_category": dominant,
            "severity":        severity,
            "peak_confidence": round(peak_conf, 4),
            "is_anomaly":      n_anomaly > 0,
            "recommendations": _RECOMMENDATIONS.get(dominant, []),
        }
