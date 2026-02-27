"""
DrishtiX Crowd Anomaly Detection — Synthetic Dataset Generator
===============================================================
Generates realistic crowd monitoring telemetry for event venues,
matching the DrishtiX platform's data schema (zones, incidents, analytics).

Anomaly types injected (mirrors incident_logs.type field):
  - CROWD_SURGE        : sudden density spike in a zone
  - STAMPEDE_PRECURSOR : high density + inward flow + blocked exits
  - BOTTLENECK         : one zone critically full while neighbours are empty
  - FLOW_REVERSAL      : crowd direction suddenly reverses (panic signal)
  - ISOLATION_ZONE     : density drops abruptly to 0 (forced evacuation)
  - SUSTAINED_OVERLOAD : density above 0.85 for > 10 consecutive readings

Output: data/crowd_telemetry.csv  (raw)
        data/crowd_telemetry_features.csv  (engineered features, model-ready)

Usage:
  python generate_dataset.py
  python generate_dataset.py --samples 30000 --anomaly-rate 0.04
"""

import argparse
import logging
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Venue / zone topology (realistic for a 50 000-seat stadium)
# ──────────────────────────────────────────────────────────────
ZONES = {
    "ZONE_A": {"capacity": 5000, "exits": 4, "adj": ["ZONE_B", "ZONE_F"]},
    "ZONE_B": {"capacity": 4500, "exits": 3, "adj": ["ZONE_A", "ZONE_C"]},
    "ZONE_C": {"capacity": 6000, "exits": 5, "adj": ["ZONE_B", "ZONE_D"]},
    "ZONE_D": {"capacity": 5500, "exits": 4, "adj": ["ZONE_C", "ZONE_E"]},
    "ZONE_E": {"capacity": 4000, "exits": 3, "adj": ["ZONE_D", "ZONE_F"]},
    "ZONE_F": {"capacity": 5000, "exits": 4, "adj": ["ZONE_E", "ZONE_A"]},
    "ENTRY_GATE_NORTH": {"capacity": 2000, "exits": 8, "adj": ["ZONE_A", "ZONE_B"]},
    "ENTRY_GATE_SOUTH": {"capacity": 2000, "exits": 8, "adj": ["ZONE_D", "ZONE_E"]},
    "CONCOURSE_MAIN":   {"capacity": 8000, "exits": 10, "adj": ["ZONE_A", "ZONE_C", "ZONE_D", "ZONE_F"]},
}

EVENT_TYPES = ["CONCERT", "SPORTS_FINAL", "FESTIVAL", "CONFERENCE", "EXHIBITION"]

# Anomaly label → numeric code for training
ANOMALY_CODES = {
    "NORMAL":               0,
    "CROWD_SURGE":          1,
    "STAMPEDE_PRECURSOR":   2,
    "BOTTLENECK":           3,
    "FLOW_REVERSAL":        4,
    "ISOLATION_ZONE":       5,
    "SUSTAINED_OVERLOAD":   6,
}


# ──────────────────────────────────────────────────────────────
# Helper: realistic daily density profile for an event
# ──────────────────────────────────────────────────────────────
def _event_density_profile(minutes_since_start: float, event_type: str) -> float:
    """
    Returns expected base density (0-1) given time into event.
    Mirrors multi-phase crowd patterns: entry ramp, peak, intermission, exit rush.
    """
    t = minutes_since_start

    if event_type in ("CONCERT", "FESTIVAL"):
        # Slow fill → peak → drop at end
        if t < 60:
            return 0.1 + 0.6 * (t / 60)
        elif t < 180:
            return 0.7 + 0.1 * math.sin(math.pi * (t - 60) / 120)
        elif t < 210:
            return 0.75 - 0.3 * ((t - 180) / 30)
        else:
            return max(0.05, 0.45 - 0.3 * ((t - 210) / 60))

    elif event_type in ("SPORTS_FINAL",):
        # Sharp gates-open ramp → two halftime dips
        if t < 30:
            return 0.05 + 0.7 * (t / 30)
        elif t < 45:
            return 0.75
        elif t < 90:
            return 0.75 + 0.1 * math.sin(math.pi * t / 90)
        elif t < 105:
            return 0.65  # halftime, some leave for food
        else:
            return 0.7 + 0.05 * math.sin(math.pi * t / 45)

    else:  # CONFERENCE / EXHIBITION
        if t < 30:
            return 0.1 + 0.4 * (t / 30)
        elif t < 240:
            return 0.45 + 0.1 * math.sin(math.pi * t / 120)
        else:
            return max(0.05, 0.5 - 0.3 * ((t - 240) / 60))


# ──────────────────────────────────────────────────────────────
# Single event simulation
# ──────────────────────────────────────────────────────────────
def simulate_event(
    event_id: str,
    event_type: str,
    event_start: datetime,
    duration_minutes: int,
    reading_interval_seconds: int,
    anomaly_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate crowd telemetry for one event across all zones."""
    records = []
    zone_names = list(ZONES.keys())
    n_zones = len(zone_names)

    # Per-zone states
    zone_density = {z: 0.05 for z in zone_names}
    zone_overload_streak = {z: 0 for z in zone_names}

    total_steps = (duration_minutes * 60) // reading_interval_seconds

    for step in range(total_steps):
        ts = event_start + timedelta(seconds=step * reading_interval_seconds)
        minutes_elapsed = step * reading_interval_seconds / 60.0
        hour_of_day = ts.hour + ts.minute / 60.0
        is_peak_hour = 1 if (17 <= ts.hour <= 21) else 0

        # Determine if this step gets an anomaly injected
        inject_anomaly = rng.random() < anomaly_rate
        anomaly_zone = rng.choice(zone_names) if inject_anomaly else None
        anomaly_type_choice = rng.choice([
            "CROWD_SURGE", "STAMPEDE_PRECURSOR", "BOTTLENECK",
            "FLOW_REVERSAL", "ISOLATION_ZONE"
        ]) if inject_anomaly else None

        for zone_id in zone_names:
            zone_meta = ZONES[zone_id]
            base = _event_density_profile(minutes_elapsed, event_type)

            # Zone-level jitter (each zone fills at slightly different rates)
            zone_bias = rng.normal(0, 0.04)
            base_density = float(np.clip(base + zone_bias, 0.0, 1.0))

            # Smooth update: carry 70% previous + 30% new target
            smooth_density = 0.7 * zone_density[zone_id] + 0.3 * base_density
            smooth_density += rng.normal(0, 0.015)  # sensor noise
            smooth_density = float(np.clip(smooth_density, 0.0, 1.0))

            # ── Anomaly injection ──────────────────────────
            anomaly_label = "NORMAL"
            if inject_anomaly and zone_id == anomaly_zone:

                if anomaly_type_choice == "CROWD_SURGE":
                    smooth_density = float(np.clip(smooth_density + rng.uniform(0.35, 0.55), 0.0, 1.0))
                    anomaly_label = "CROWD_SURGE"

                elif anomaly_type_choice == "STAMPEDE_PRECURSOR":
                    smooth_density = float(np.clip(smooth_density + rng.uniform(0.3, 0.5), 0.0, 1.0))
                    anomaly_label = "STAMPEDE_PRECURSOR"

                elif anomaly_type_choice == "BOTTLENECK":
                    smooth_density = float(np.clip(smooth_density + rng.uniform(0.4, 0.6), 0.0, 1.0))
                    anomaly_label = "BOTTLENECK"

                elif anomaly_type_choice == "FLOW_REVERSAL":
                    # Density itself doesn't spike but flow metrics will
                    anomaly_label = "FLOW_REVERSAL"

                elif anomaly_type_choice == "ISOLATION_ZONE":
                    smooth_density = rng.uniform(0.0, 0.02)  # sudden evacuation
                    anomaly_label = "ISOLATION_ZONE"

            # ── Sustained overload streak ──────────────────
            if smooth_density > 0.85:
                zone_overload_streak[zone_id] += 1
            else:
                zone_overload_streak[zone_id] = 0

            if zone_overload_streak[zone_id] > 10 and anomaly_label == "NORMAL":
                anomaly_label = "SUSTAINED_OVERLOAD"

            # ── Flow rates (people / minute) ──────────────
            capacity = zone_meta["capacity"]
            if anomaly_label == "STAMPEDE_PRECURSOR":
                flow_in = rng.uniform(80, 150)
                flow_out = rng.uniform(5, 20)
            elif anomaly_label == "FLOW_REVERSAL":
                # net flow flips sign
                flow_in = rng.uniform(5, 20)
                flow_out = rng.uniform(80, 150)
            elif anomaly_label == "ISOLATION_ZONE":
                flow_in = 0.0
                flow_out = rng.uniform(100, 200)
            else:
                flow_in = float(capacity * smooth_density * rng.uniform(0.02, 0.08))
                flow_out = float(capacity * smooth_density * rng.uniform(0.015, 0.07))

            net_flow = flow_in - flow_out

            # ── Environmental sensors ──────────────────────
            base_temp = 22.0 + 5.0 * math.sin(math.pi * hour_of_day / 12)
            crowd_heat = smooth_density * 8.0
            temperature = base_temp + crowd_heat + rng.normal(0, 0.5)
            noise_db = 55 + smooth_density * 45 + rng.normal(0, 2)

            # ── Adjacent zone pressure ─────────────────────
            adj_zones = zone_meta["adj"]
            adj_densities = [zone_density[z] for z in adj_zones if z in zone_density]
            adj_pressure = float(np.mean(adj_densities)) if adj_densities else smooth_density

            # ── Exit proximity score (exits per 1000 people at current density)
            exit_proximity = zone_meta["exits"] / max(1, smooth_density * capacity / 1000)
            exit_proximity = float(np.clip(exit_proximity / 10.0, 0.0, 1.0))

            # ── Dwell time estimate (inverse of flow)  ─────
            dwell_time = float(np.clip(capacity * smooth_density / max(1, flow_out), 5, 120))

            records.append({
                "event_id":                  event_id,
                "event_type":                event_type,
                "zone_id":                   zone_id,
                "timestamp":                 ts.isoformat(),
                "hour_of_day":               round(hour_of_day, 3),
                "minutes_since_event_start": round(minutes_elapsed, 2),
                "is_peak_hour":              is_peak_hour,
                # Core density signals
                "crowd_density":             round(smooth_density, 4),
                "capacity_utilization":      round(smooth_density, 4),  # alias clarity
                # Flow signals
                "flow_rate_in":              round(flow_in, 2),
                "flow_rate_out":             round(flow_out, 2),
                "net_flow":                  round(net_flow, 2),
                # Environmental
                "temperature_celsius":       round(temperature, 2),
                "noise_level_db":            round(noise_db, 2),
                # Spatial
                "adjacent_zone_pressure":    round(adj_pressure, 4),
                "exit_proximity_score":      round(exit_proximity, 4),
                "dwell_time_minutes":        round(dwell_time, 2),
                # Labels
                "anomaly_label":             anomaly_label,
                "anomaly_code":              ANOMALY_CODES[anomaly_label],
                "is_anomaly":                0 if anomaly_label == "NORMAL" else 1,
            })

            zone_density[zone_id] = smooth_density

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# Feature engineering (rolling / lagged features)
# ──────────────────────────────────────────────────────────────
def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling statistics and derived signals per zone.
    These are the actual features fed to the Isolation Forest.
    """
    logger.info("Engineering features (rolling windows, z-scores, deltas) …")
    dfs = []
    for (event_id, zone_id), grp in raw_df.groupby(["event_id", "zone_id"]):
        grp = grp.sort_values("timestamp").copy()
        cd = grp["crowd_density"]

        # Rolling windows (5 or 10 readings = 2.5–5 min at 30-s intervals)
        grp["density_rolling_mean_5"]  = cd.rolling(5,  min_periods=1).mean()
        grp["density_rolling_std_5"]   = cd.rolling(5,  min_periods=1).std().fillna(0)
        grp["density_rolling_mean_10"] = cd.rolling(10, min_periods=1).mean()
        grp["density_rolling_std_10"]  = cd.rolling(10, min_periods=1).std().fillna(0)

        # Delta (rate of change per reading)
        grp["density_delta"]           = cd.diff().fillna(0)
        grp["density_delta_abs"]       = grp["density_delta"].abs()

        # Z-score relative to 10-reading window
        rm10 = grp["density_rolling_mean_10"]
        rs10 = grp["density_rolling_std_10"].replace(0, 1e-6)
        grp["density_z_score"]         = (cd - rm10) / rs10

        # Flow ratio (in vs out asymmetry)
        denom = (grp["flow_rate_in"] + grp["flow_rate_out"]).replace(0, 1e-6)
        grp["flow_imbalance_ratio"]    = (grp["flow_rate_in"] - grp["flow_rate_out"]) / denom

        # Crowding pressure index
        grp["crowding_pressure"]       = (
            grp["crowd_density"] * grp["adjacent_zone_pressure"] *
            (1 - grp["exit_proximity_score"])
        )

        # Density × noise (perceptual overload)
        grp["perceptual_overload"]     = grp["crowd_density"] * grp["noise_level_db"] / 100

        dfs.append(grp)

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Feature engineering complete → {len(result):,} rows, {result.shape[1]} columns")
    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main(n_events: int = 10, duration: int = 300, interval: int = 30,
         anomaly_rate: float = 0.03, seed: int = 42):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    start_base = datetime(2025, 7, 1, 17, 0, 0)
    all_raw: list[pd.DataFrame] = []

    for i in range(n_events):
        etype = random.choice(EVENT_TYPES)
        eid   = f"EVT-{i+1:03d}"
        estart = start_base + timedelta(days=i * 3)
        logger.info(f"  Simulating {eid} ({etype}) …")
        df = simulate_event(eid, etype, estart, duration, interval, anomaly_rate, rng)
        all_raw.append(df)

    raw_df = pd.concat(all_raw, ignore_index=True)
    feat_df = engineer_features(raw_df)

    # Save
    Path("data").mkdir(exist_ok=True)
    raw_path  = Path("data/crowd_telemetry.csv")
    feat_path = Path("data/crowd_telemetry_features.csv")
    raw_df.to_csv(raw_path,  index=False)
    feat_df.to_csv(feat_path, index=False)

    anomaly_counts = feat_df[feat_df["is_anomaly"] == 1]["anomaly_label"].value_counts()
    logger.info(
        f"\n✅  Dataset generated:"
        f"\n   Raw rows  : {len(raw_df):,}"
        f"\n   Feature rows: {len(feat_df):,}"
        f"\n   Total anomalies : {feat_df['is_anomaly'].sum():,} "
        f"({feat_df['is_anomaly'].mean()*100:.2f}%)"
        f"\n   By type:\n{anomaly_counts.to_string()}"
        f"\n   Files → {raw_path}  |  {feat_path}"
    )
    return feat_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrishtiX synthetic crowd data generator")
    parser.add_argument("--events",       type=int,   default=10,   help="Number of events")
    parser.add_argument("--duration",     type=int,   default=300,  help="Event duration in minutes")
    parser.add_argument("--interval",     type=int,   default=30,   help="Reading interval in seconds")
    parser.add_argument("--anomaly-rate", type=float, default=0.03, help="Fraction of readings that are anomalous")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()
    main(args.events, args.duration, args.interval, args.anomaly_rate, args.seed)
