"""
DrishtiX Crowd Anomaly Detection — Unified REST API  (v2)
=========================================================
Merged Flask server that exposes both:
  • Telemetry-based crowd anomaly detection  (via CrowdAnomalyDetector)
  • Visual-based crowd scene classification   (via VisualCrowdDetector + CLIP)
  • Fused prediction combining both signals

Endpoints
---------
  GET  /health                        — health check
  GET  /model/info                    — loaded model metadata

  POST /predict/telemetry             — single zone sensor reading → AnomalyResult
  POST /predict/telemetry/batch       — batch of readings (JSON array or CSV upload)
  POST /predict/visual                — single image upload → VisualAnomalyResult
  POST /predict/fused                 — image + telemetry → merged result

  POST /video/analyse                 — upload video file → per-frame + summary
  GET  /video/stream                  — live MJPEG camera stream with overlay

  GET  /simulation/step               — step synthetic zone simulation (demo)

Start
-----
  python flask_app.py
  # or via gunicorn:
  gunicorn -w 2 -b 0.0.0.0:5001 flask_app:app
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, send_file

# ──────────────────────────────────────────────────────────────
# Ensure this module's directory is on the path
# ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from anomaly_detector import CrowdAnomalyDetector, FEATURE_COLS_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# App init
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

os.makedirs(_HERE / "uploads", exist_ok=True)
os.makedirs(_HERE / "outputs", exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Model singletons (loaded once at startup)
# ──────────────────────────────────────────────────────────────
_telemetry_detector: CrowdAnomalyDetector | None = None
_visual_detector    = None   # VisualCrowdDetector — optional
_models_lock        = threading.Lock()


def _load_models():
    global _telemetry_detector, _visual_detector
    with _models_lock:
        if _telemetry_detector is None:
            logger.info("Loading CrowdAnomalyDetector …")
            _telemetry_detector = CrowdAnomalyDetector.from_trained()

        if _visual_detector is None:
            try:
                from visual_detector import VisualCrowdDetector
                _visual_detector = VisualCrowdDetector()
                if not _visual_detector.is_ready:
                    _visual_detector = None
                    logger.warning("VisualCrowdDetector not ready (CLIP/torch missing?).")
                else:
                    logger.info("VisualCrowdDetector loaded.")
            except Exception as e:
                logger.warning(f"VisualCrowdDetector unavailable: {e}")
                _visual_detector = None


# Load at startup
_load_models()


def _telem() -> CrowdAnomalyDetector:
    if _telemetry_detector is None:
        _load_models()
    return _telemetry_detector


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _parse_image(file_storage) -> np.ndarray | None:
    """Decode an uploaded image file to an RGB ndarray."""
    try:
        from PIL import Image as PILImage
        img_bytes = file_storage.read()
        pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(pil)
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        return None


def _reading_from_request() -> dict:
    """Extract a telemetry reading dict from JSON body or form fields."""
    if request.is_json:
        data = request.get_json(force=True) or {}
    else:
        data = request.form.to_dict()
    # Cast numeric strings
    return {k: float(v) if isinstance(v, str) else v for k, v in data.items()}


# ──────────────────────────────────────────────────────────────
# Routes — Health & Info
# ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    det = _telem()
    return jsonify({
        "status":          "ok",
        "telemetry_model": det.model_name if det else "not_loaded",
        "ensemble_size":   len(det._ensemble) if det else 0,
        "visual_model":    "loaded" if _visual_detector else "unavailable",
        "feature_count":   len(det.feature_cols) if det else 0,
    })


@app.route("/model/info")
def model_info():
    det = _telem()
    report_path = _HERE / "models" / "training_report.json"
    report = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    return jsonify({
        "ensemble_members": [
            {"name": s.name, "weight": round(s.weight, 4)}
            for s in (det._ensemble if det else [])
        ],
        "feature_cols":     det.feature_cols if det else [],
        "calibrated":       det._calibrated   if det else False,
        "training_report":  report,
    })


# ──────────────────────────────────────────────────────────────
# Routes — Telemetry Prediction
# ──────────────────────────────────────────────────────────────
@app.route("/predict/telemetry", methods=["POST"])
def predict_telemetry():
    """
    Body (JSON):
      { "crowd_density": 0.9, "flow_rate_in": 200, ... }

    Returns AnomalyResult dict.
    """
    reading = _reading_from_request()
    if not reading:
        return jsonify({"error": "No reading data provided"}), 400

    det    = _telem()
    result = det.predict_single(reading)
    return jsonify(result.to_dict())


@app.route("/predict/telemetry/batch", methods=["POST"])
def predict_telemetry_batch():
    """
    Accept either:
      • JSON array of reading dicts
      • CSV file upload (field name: 'file')
    Returns array of prediction dicts.
    """
    import pandas as pd

    if "file" in request.files:
        f = request.files["file"]
        try:
            df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
        except Exception as e:
            return jsonify({"error": f"CSV parse error: {e}"}), 400
    elif request.is_json:
        rows = request.get_json(force=True)
        if not isinstance(rows, list):
            return jsonify({"error": "Expected JSON array of readings"}), 400
        df = pd.DataFrame(rows)
    else:
        return jsonify({"error": "Provide JSON array or CSV file"}), 400

    det    = _telem()
    result = det.predict_batch(df)
    return jsonify(result.to_dict(orient="records"))


# ──────────────────────────────────────────────────────────────
# Routes — Visual Prediction
# ──────────────────────────────────────────────────────────────
@app.route("/predict/visual", methods=["POST"])
def predict_visual():
    """
    Upload an image (field: 'image') and get a VisualAnomalyResult.
    """
    if _visual_detector is None:
        return jsonify({"error": "Visual detector not available. Install torch + CLIP."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field: 'image')"}), 400

    rgb = _parse_image(request.files["image"])
    if rgb is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = _visual_detector.predict(rgb, use_smoother=False)
    return jsonify(result.to_dict())


# ──────────────────────────────────────────────────────────────
# Routes — Fused Prediction (visual + telemetry)
# ──────────────────────────────────────────────────────────────
@app.route("/predict/fused", methods=["POST"])
def predict_fused():
    """
    Multipart form with:
      • 'image'       — optional image file
      • All telemetry fields as form keys OR JSON body with 'telemetry' key
      • 'telemetry'   — nested JSON object if Content-Type is application/json

    Returns merged prediction: telemetry result escalated by visual evidence.
    """
    # Parse telemetry
    if request.is_json:
        body     = request.get_json(force=True) or {}
        reading  = body.get("telemetry", body)
        img_arr  = None
    else:
        reading = {k: float(v) for k, v in request.form.items() if k != "image"}
        img_arr = None
        if "image" in request.files:
            img_arr = _parse_image(request.files["image"])

    visual_score = 0.0
    visual_data  = {}

    if img_arr is not None and _visual_detector is not None:
        vres         = _visual_detector.predict(img_arr, use_smoother=False)
        visual_score = vres.smoothed_score
        visual_data  = vres.to_dict()

    det    = _telem()
    result = det.predict_single(reading, visual_score=visual_score)

    return jsonify({
        "telemetry": result.to_dict(),
        "visual":    visual_data,
        "fused": {
            "is_anomaly":    result.is_anomaly,
            "final_score":   result.anomaly_score,
            "severity":      result.severity,
            "anomaly_type":  result.anomaly_type,
            "visual_score":  visual_score,
        },
    })


# ──────────────────────────────────────────────────────────────
# Routes — Video Analysis
# ──────────────────────────────────────────────────────────────
_video_status: dict = {"status": "idle", "progress": 0, "total": 0, "message": ""}


@app.route("/video/analyse", methods=["POST"])
def analyse_video():
    """
    Upload a video file (field: 'video'). Processes asynchronously.
    Poll GET /video/status for progress; GET /video/result for results.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file (field: 'video')"}), 400

    f = request.files["video"]
    tmp = tempfile.NamedTemporaryFile(
        suffix=Path(f.filename).suffix, delete=False,
        dir=str(_HERE / "uploads")
    )
    f.save(tmp.name)

    def _process(path: str):
        global _video_status
        _video_status = {"status": "processing", "progress": 0, "total": 0,
                         "message": "Starting…", "results": []}
        try:
            if _visual_detector is None:
                _video_status["status"]  = "error"
                _video_status["message"] = "Visual detector not available."
                return

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _video_status["total"] = total

            frames  = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            summary = _visual_detector.summarise_video(frames, frame_stride=2)
            _video_status.update({
                "status":  "completed",
                "progress": total,
                "message": f"Done — {summary['anomaly_frames']} anomaly frames",
                "summary": summary,
            })
        except Exception as e:
            _video_status.update({"status": "error", "message": str(e)})
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    t = threading.Thread(target=_process, args=(tmp.name,), daemon=True)
    t.start()
    return jsonify({"message": "Video upload accepted. Poll /video/status."})


@app.route("/video/status")
def video_status():
    return jsonify(_video_status)


# ──────────────────────────────────────────────────────────────
# Routes — Live Camera Stream
# ──────────────────────────────────────────────────────────────
_camera      = None
_camera_lock = threading.Lock()


def _gen_camera_frames():
    global _camera
    det = _telem()
    while True:
        with _camera_lock:
            if _camera is None:
                break
            ok, frame = _camera.read()
        if not ok:
            break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        text = "NORMAL"
        color = (0, 200, 0)

        # Quick visual classification if available
        if _visual_detector:
            vres  = _visual_detector.predict(rgb, use_smoother=True)
            text  = f"{vres.category} ({vres.smoothed_score:.2f})"
            color = (0, 0, 200) if vres.is_anomaly else (0, 200, 0)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (8, 8), (20 + tw, 24 + th), (0, 0, 0), -1)
        cv2.putText(frame, text, (14, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/video/stream")
def camera_stream():
    return Response(_gen_camera_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera/start", methods=["POST"])
def camera_start():
    global _camera
    with _camera_lock:
        if _camera is not None:
            return jsonify({"error": "Camera already active"}), 400
        idx = int(request.json.get("device", 0)) if request.is_json else 0
        _camera = cv2.VideoCapture(idx)
        if not _camera.isOpened():
            _camera = None
            return jsonify({"error": "Could not open camera"}), 500
    return jsonify({"message": "Camera started. Stream at /video/stream"})


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    global _camera
    with _camera_lock:
        if _camera:
            _camera.release()
            _camera = None
    return jsonify({"message": "Camera stopped"})


# ──────────────────────────────────────────────────────────────
# Routes — Synthetic Zone Simulation (demo)
# ──────────────────────────────────────────────────────────────
_sim_step = 0


@app.route("/simulation/step", methods=["GET", "POST"])
def simulation_step():
    """One-step simulation: returns predictions for all 9 zones."""
    global _sim_step
    det     = _telem()
    zones   = ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "ZONE_E", "ZONE_F",
               "ENTRY_GATE_NORTH", "ENTRY_GATE_SOUTH", "CONCOURSE_MAIN"]
    inject  = (request.json or {}).get("inject_zone")
    results = {}

    for zone in zones:
        rng     = np.random.default_rng(_sim_step * 17 + abs(hash(zone)) % 1000)
        t_norm  = min(1.0, _sim_step / 80)
        density = float(np.clip(0.2 + 0.5 * np.sin(np.pi * t_norm) + rng.normal(0, 0.04), 0, 1))
        if zone == inject:
            density = float(np.clip(density + rng.uniform(0.35, 0.55), 0, 1))

        fi  = density * 5000 * rng.uniform(0.03, 0.07)
        fo  = density * 5000 * rng.uniform(0.02, 0.06)
        reading = {
            "crowd_density": density, "capacity_utilization": density,
            "density_delta": rng.uniform(-0.05, 0.05) if zone != inject else 0.35,
            "density_delta_abs": 0.35 if zone == inject else abs(rng.uniform(-0.05, 0.05)),
            "density_rolling_mean_5": density * 0.95, "density_rolling_std_5": 0.02,
            "density_rolling_mean_10": density * 0.92, "density_rolling_std_10": 0.04,
            "density_z_score": 4.5 if zone == inject else (density - 0.5) / 0.15,
            "flow_rate_in": float(fi), "flow_rate_out": float(fo),
            "net_flow": float(fi - fo),
            "flow_imbalance_ratio": (fi - fo) / max(fi + fo, 1),
            "adjacent_zone_pressure": density * 0.8, "exit_proximity_score": 0.4,
            "dwell_time_minutes": float(np.clip(density * 60, 5, 120)),
            "crowding_pressure": density * 0.7 * 0.6,
            "perceptual_overload": density * 70 / 100,
            "temperature_celsius": 22 + density * 8,
            "noise_level_db": 55 + density * 45,
            "hour_of_day": 18.0 + _sim_step / 60,
            "minutes_since_event_start": float(_sim_step),
            "is_peak_hour": 1,
        }
        res = det.predict_single(reading)
        results[zone] = {
            "density":    round(density, 3),
            "is_anomaly": res.is_anomaly,
            "score":      res.anomaly_score,
            "severity":   res.severity,
            "type":       res.anomaly_type,
        }

    _sim_step += 1
    return jsonify({"step": _sim_step, "zones": results})


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  DrishtiX Crowd Anomaly Detection — REST API")
    print("  http://127.0.0.1:5001")
    print("=" * 60)
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5001)
