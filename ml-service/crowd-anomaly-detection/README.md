# DrishtiX — Crowd Anomaly Detection Module

Standalone, fully trained Isolation Forest anomaly detection system for
crowd safety monitoring at live events. Designed to integrate with the
DrishtiX platform (REST API, incident_logs, zone telemetry).

---

## Quick Start (3 commands)

```bash
cd ml-service/crowd-anomaly-detection

pip install -r requirements.txt

python generate_dataset.py        # step 1 — create synthetic crowd data
python train_model.py             # step 2 — train all three models
streamlit run streamlit_app.py    # step 3 — open the dashboard
```

The Streamlit app will be available at `http://localhost:8501`

You can also click **⚡ Generate Dataset** and **🤖 Train Models** directly
from the sidebar inside the app.

---

## Directory Structure

```
crowd-anomaly-detection/
├── generate_dataset.py       Synthetic crowd data generator
├── train_model.py            Trains IF / LOF / One-Class SVM
├── anomaly_detector.py       Reusable detector class (for API integration)
├── streamlit_app.py          Full Streamlit dashboard
├── requirements.txt
├── data/
│   ├── crowd_telemetry.csv          Raw simulated readings
│   └── crowd_telemetry_features.csv Engineered features for training
└── models/
    ├── isolation_forest_pipeline.pkl
    ├── lof_pipeline.pkl
    ├── ocsvm_pipeline.pkl
    ├── best_model.pkl                Best model by F1 score
    ├── feature_cols.json
    └── training_report.json
```

---

## What Gets Generated

### Synthetic Dataset

10 events × 9 zones × ~600 readings each = **~54 000 rows** by default.

Each row represents a 30-second zone telemetry snapshot containing:

| Feature | Description |
|---------|-------------|
| `crowd_density` | Normalised density (0.0 – 1.0) |
| `density_delta` | Change per reading |
| `density_rolling_mean_5/10` | Rolling averages |
| `density_z_score` | Standard deviations from local mean |
| `flow_rate_in/out` | People entering / leaving (per minute) |
| `net_flow` | Net crowd movement |
| `flow_imbalance_ratio` | Asymmetry of flows |
| `adjacent_zone_pressure` | Avg density in neighbouring zones |
| `exit_proximity_score` | Effective exit capacity ratio |
| `crowding_pressure` | Composite congestion index |
| `temperature_celsius` | Ambient + body heat estimate |
| `noise_level_db` | Sound level (correlated with crowd) |
| `dwell_time_minutes` | Estimated stay time |
| `perceptual_overload` | Density × noise index |

### Anomaly Types Injected

| Type | Description | Severity |
|------|-------------|----------|
| `CROWD_SURGE` | Sudden density spike (+0.35–0.55) | HIGH |
| `STAMPEDE_PRECURSOR` | High density + blocked exits + inward flow | CRITICAL |
| `BOTTLENECK` | Zone critically full, neighbours empty | HIGH |
| `FLOW_REVERSAL` | Mass egress reversal (panic signal) | MEDIUM |
| `ISOLATION_ZONE` | Density drops to 0 — forced evacuation | MEDIUM |
| `SUSTAINED_OVERLOAD` | Density > 0.85 for > 10 consecutive readings | CRITICAL |

---

## Models

### 1. Isolation Forest (primary)
- Trains on **normal-only** data — pure unsupervised
- 150 trees, RobustScaler preprocessing
- Fast prediction: < 1 ms per reading

### 2. Local Outlier Factor
- Density-based; detects local neighbourhood anomalies
- Good at detecting gradual drift

### 3. One-Class SVM
- Kernel-based boundary around normal behaviour
- Complementary to IF for non-linear anomalies

The best model by F1 on the held-out test set is saved as `best_model.pkl`.

---

## Integration with DrishtiX

### Python import

```python
from anomaly_detector import CrowdAnomalyDetector, AnomalyResult

detector = CrowdAnomalyDetector.from_trained()        # loads best_model.pkl

reading = {
    "crowd_density": 0.92,
    "density_delta": 0.38,
    "density_delta_abs": 0.38,
    "flow_rate_in": 210,
    "flow_rate_out": 8,
    "net_flow": 202,
    "adjacent_zone_pressure": 0.65,
    "exit_proximity_score": 0.08,
    "density_z_score": 4.8,
    # ... remaining features default to 0 if omitted ...
}

result: AnomalyResult = detector.predict_single(reading)
print(result.to_dict())
# {
#   "is_anomaly": true,
#   "anomaly_score": 0.9312,
#   "severity": "CRITICAL",
#   "anomaly_type": "CROWD_SURGE",
#   "recommendations": ["Deploy crowd marshals ...", ...]
# }
```

### REST API wrapper (FastAPI)

```python
from fastapi import FastAPI
from anomaly_detector import CrowdAnomalyDetector

app      = FastAPI()
detector = CrowdAnomalyDetector.from_trained()

@app.post("/predict")
def predict(reading: dict):
    return detector.predict_single(reading).to_dict()
```

### Mapping to DrishtiX incident_logs schema

| AnomalyResult field | incident_logs column |
|---------------------|---------------------|
| `anomaly_type` | `type` |
| `severity` | `severity` |
| `is_anomaly` | triggers INSERT |
| `recommendations` | `description` |
| zone / event from reading | `event_id`, `affected_zones` |

---

## Customising the Dataset

```bash
python generate_dataset.py \
  --events 20 \
  --duration 480 \
  --interval 15 \
  --anomaly-rate 0.05 \
  --seed 123
```

| Option | Default | Description |
|--------|---------|-------------|
| `--events` | 10 | Number of events to simulate |
| `--duration` | 300 | Event duration (minutes) |
| `--interval` | 30 | Reading interval (seconds) |
| `--anomaly-rate` | 0.03 | Fraction of anomalous readings |

---

## Streamlit App Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPIs, density timeline, anomaly distribution, architecture diagram |
| 📊 Dataset Explorer | Filter, visualise, download the training data |
| 🔧 Train Models | Tune hyperparameters and retrain in-browser |
| 🔍 Single Reading | Manual slider inputs → instant prediction + gauge |
| 📂 Batch Prediction | Upload CSV or score the generated dataset |
| 📈 Model Performance | F1/AUC comparison, radar chart, per-type recall |
| 🗺️ Zone Simulation | Live animated grid of 9 zones with anomaly injection |
| 📋 Incident Log | Timeline view of all anomalies, filterable, exportable |

---

## License

Copyright © 2025 DrishtiX. Proprietary and confidential.
See `LICENSE` in the project root.
