"""
=============================================================
Compressor Stability Inference Engine
KES22 8.5 bar | Real-Time & Replay Modes
=============================================================
Usage (replay):
    python compressor_inference.py --model model.pkl \
        --data KES22_8p5_synthetic_test_data.xlsx --mode replay

Usage (real-time streaming):
    python compressor_inference.py --model model.pkl \
        --data live_feed.xlsx --mode realtime

Live integration (import as module):
    from compressor_inference import CompressorInference
    engine = CompressorInference("model.pkl")
    result = engine.predict(window_df)

Requirements:
    pip install pandas numpy scikit-learn openpyxl
=============================================================
"""

import argparse
import pickle
import warnings
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = _ROOT / "data" / "quantile_models.pkl"
DEFAULT_DATA_PATH = _ROOT / "data" / "KES22_8p5_synthetic_test_data.xlsx"

warnings.filterwarnings("ignore")


# ============================================================
# CONSTANTS
# ============================================================
SAFETY_BUFFER_MIN = 10    # added to p90-equivalent bound for recommended end
BASELINE_TEST_MIN = 180   # standard test duration (minutes)
MIN_READINGS      = 20    # minimum rows before inference fires
MIN_ELAPSED_MIN   = 15.0  # minimum elapsed time before inference fires


# ============================================================
# HELPER
# ============================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# FEATURE ENGINEERING  (must mirror compressor_train.py exactly)
# ============================================================
def _engineer_features(window_df: pd.DataFrame,
                        stable_ref: pd.Series,
                        stable_std: pd.Series,
                        sensors: list,
                        spec_columns: list = []) -> dict:
    """
    Compute the same feature snapshot used during training.
    window_df must contain 'elapsed_time_min' + all sensor columns.
    """
    feats       = {}
    w           = window_df[sensors].copy()
    roll20_std  = w.rolling(20, min_periods=1).std()
    roll10_std  = w.rolling(10, min_periods=1).std()
    roll20_mean = w.rolling(20, min_periods=1).mean()

    i = len(window_df) - 1  # last row index

    for s in sensors:
        cur = w[s].iloc[-1]
        feats[f"{s}_cur"]        = cur
        feats[f"{s}_dev"]        = (cur - stable_ref[s]) / stable_std[s]
        feats[f"{s}_roll20_std"] = roll20_std[s].iloc[-1]
        feats[f"{s}_roll10_std"] = roll10_std[s].iloc[-1]

        vals20 = w[s].iloc[max(0, i - 19): i + 1].values
        vals10 = w[s].iloc[max(0, i - 9):  i + 1].values

        feats[f"{s}_slope20"] = (
            np.polyfit(np.arange(len(vals20)), vals20, 1)[0]
            if len(vals20) >= 2 else 0.0
        )
        feats[f"{s}_slope10"] = (
            np.polyfit(np.arange(len(vals10)), vals10, 1)[0]
            if len(vals10) >= 2 else 0.0
        )

    devs_cur  = np.array([(w[s].iloc[-1] - stable_ref[s]) / stable_std[s] for s in sensors])
    devs_roll = np.array([(roll20_mean[s].iloc[-1] - stable_ref[s]) / stable_std[s] for s in sensors])

    feats["total_dev_roll20"] = float(np.abs(devs_cur).mean())
    feats["total_dev_roll10"] = float(np.abs(devs_roll).mean())
    feats["obs_time_min"]     = float(window_df["elapsed_time_min"].iloc[-1])

    # Spec columns
    for col in spec_columns:
        if col in window_df.columns:
            feats[col] = window_df[col].iloc[-1]
        else:
            feats[col] = 0.0

    # Relative features (matching compressor_train.py)
    # Note: These names are hardcoded in training script's engineer_features_batch
    if "rated_fad_cfm" in window_df.columns and window_df["rated_fad_cfm"].iloc[-1] != 0:
        feats["fad_ratio"] = float(w["fad_cfm"].iloc[-1] / window_df["rated_fad_cfm"].iloc[-1])
        feats["fad_deviation_pct"] = float(
            ((w["fad_cfm"].iloc[-1] - window_df["rated_fad_cfm"].iloc[-1]) /
             window_df["rated_fad_cfm"].iloc[-1]) * 100
        )
    else:
        feats["fad_ratio"] = 0.0
        feats["fad_deviation_pct"] = 0.0

    if "rated_motor_output_kw" in window_df.columns and window_df["rated_motor_output_kw"].iloc[-1] != 0:
        feats["motor_power_ratio"] = float(w["motor_output_power_kw"].iloc[-1] / window_df["rated_motor_output_kw"].iloc[-1])
        feats["motor_power_deviation_pct"] = float(
            ((w["motor_output_power_kw"].iloc[-1] - window_df["rated_motor_output_kw"].iloc[-1]) /
             window_df["rated_motor_output_kw"].iloc[-1]) * 100
        )
    else:
        feats["motor_power_ratio"] = 0.0
        feats["motor_power_deviation_pct"] = 0.0

    return feats


# ============================================================
# CONFIDENCE SCORE
# ============================================================
def _confidence(elapsed: float, total_dev: float, time_to_stability: float) -> float:
    """
    Heuristic confidence score (0–100 %).

    Factors:
      - time elapsed  (more elapsed → more confident)
      - total sensor deviation  (lower deviation → more confident)
      - predicted remaining time  (shorter → more confident)
    """
    time_conf = min(elapsed / 60.0, 1.0) * 50              # 0–50 % from elapsed time
    dev_conf  = max(0.0, 1.0 - total_dev / 3.0) * 30       # 0–30 % from sensor convergence
    rem_conf  = max(0.0, 1.0 - time_to_stability / 60.0) * 20  # 0–20 % from imminence
    return round(min(time_conf + dev_conf + rem_conf, 100.0), 1)


# ============================================================
# ACTION DECISION
# ============================================================
def _action(elapsed: float, confidence: float, time_to_stability: float) -> str:
    if elapsed < MIN_ELAPSED_MIN:
        return "WAIT"
    if time_to_stability <= 0:
        return "ALREADY_STABLE"
    if confidence >= 85:
        return "SAFE_TO_STOP"
    if confidence >= 70:
        return "PREPARE_STOP"
    return "CONTINUE_MONITORING"


# ============================================================
# LIVE SENSOR BUFFER  (PLC / MQTT / OPC-UA → DataFrame)
# ============================================================
class SensorStreamBuffer:
    """
    Accumulates time-ordered sensor samples without CSV/XLSX.

    Each sample must provide ``elapsed_time_min`` (from the test clock)
    and one float per trained sensor column. Optional ``phase`` is kept
    if present for traceability.

    Use :meth:`to_dataframe` with :meth:`CompressorInference.predict`.
    """

    def __init__(self, sensors: list[str], max_rows: int = 500):
        self.sensors = list(sensors)
        self.max_rows = int(max_rows)
        self._rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows = []

    def append(
        self,
        elapsed_time_min: float,
        values: dict[str, float],
        phase: Optional[str] = None,
    ) -> pd.DataFrame:
        missing = [s for s in self.sensors if s not in values]
        if missing:
            raise ValueError(f"Missing sensor keys: {missing}")

        row: dict[str, Any] = {"elapsed_time_min": float(elapsed_time_min)}
        for s in self.sensors:
            row[s] = float(values[s])
        if phase is not None:
            row["phase"] = phase
        self._rows.append(row)
        if len(self._rows) > self.max_rows:
            self._rows = self._rows[-self.max_rows :]
        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            cols = ["elapsed_time_min"] + self.sensors
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self._rows)
        df = df.sort_values("elapsed_time_min").reset_index(drop=True)
        return df


# ============================================================
# MAIN INFERENCE CLASS
# ============================================================
class CompressorInference:
    """
    Real-time inference engine for compressor stability prediction.

    Loads the bundle produced by compressor_train.py and exposes
    a single predict() method suitable for both live and replay use.

    Parameters
    ----------
    model_path : str
        Path to model.pkl produced by compressor_train.py
    """

    def __init__(self, model_path: str):
        log(f"Loading model from {model_path} ...")

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        # Validate required keys
        required = ["model", "scaler", "features", "sensors",
                    "stable_ref", "stable_std", "stable_time", "future_horizon"]
        missing = [k for k in required if k not in bundle]
        if missing:
            raise KeyError(f"Model bundle is missing keys: {missing}. "
                           f"Re-train with compressor_train.py.")

        self.model          = bundle["model"]           # MultiOutputRegressor
        self.scaler         = bundle["scaler"]
        self.features       = bundle["features"]
        self.sensors        = bundle["sensors"]
        self.spec_columns   = bundle.get("spec_columns", [])
        self.stable_ref     = bundle["stable_ref"]
        self.stable_std     = bundle["stable_std"]
        self.stable_time    = bundle["stable_time"]
        self.future_horizon = bundle["future_horizon"]
        self.cv_mae         = bundle.get("cv_mae", None)

        log(f"Model loaded | Stable onset: {self.stable_time:.1f} min"
            + (f" | CV MAE: {self.cv_mae:.2f} min" if self.cv_mae else ""))

    # ----------------------------------------------------------
    def predict(self,
                window_df: pd.DataFrame,
                timestamp: Optional[str] = None) -> dict:
        """
        Run one inference tick.

        Parameters
        ----------
        window_df : pd.DataFrame
            All sensor readings from test start up to the current moment.
            Required columns: 'elapsed_time_min' + all sensor columns.
        timestamp : str, optional
            ISO-format timestamp string. Defaults to UTC now.

        Returns
        -------
        dict  —  JSON-serialisable result payload.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        elapsed    = float(window_df["elapsed_time_min"].iloc[-1])
        n_readings = len(window_df)

        # Not enough data yet
        if n_readings < MIN_READINGS or elapsed < MIN_ELAPSED_MIN:
            return {
                "timestamp":       timestamp,
                "elapsed_min":     round(elapsed, 2),
                "n_readings":      n_readings,
                "action":          "WAIT",
                "confidence_pct":  0.0,
                "sensor_status":   "transient",
                "message": (
                    f"Insufficient data — need {MIN_READINGS} readings & "
                    f"{MIN_ELAPSED_MIN:.0f} min elapsed "
                    f"(got {n_readings} readings, {elapsed:.1f} min)."
                ),
            }

        # Feature engineering
        feats      = _engineer_features(
            window_df,
            self.stable_ref,
            self.stable_std,
            self.sensors,
            self.spec_columns
        )
        feat_vec   = np.array([[feats.get(f, 0.0) for f in self.features]])
        feat_scaled = self.scaler.transform(feat_vec)

        # Model prediction  →  [time_to_stability, sensor_1, ..., sensor_N]
        raw_pred         = self.model.predict(feat_scaled)[0]
        time_to_stability = max(float(raw_pred[0]), 0.0)
        sensor_forecast   = {
            s: round(float(raw_pred[1 + idx]), 3)
            for idx, s in enumerate(self.sensors)
        }

        # Derived metrics
        predicted_stable_min  = elapsed + time_to_stability
        recommended_end_min   = predicted_stable_min + SAFETY_BUFFER_MIN
        time_saved_min        = max(BASELINE_TEST_MIN - recommended_end_min, 0.0)

        total_dev = feats.get("total_dev_roll20", 999.0)
        confidence = _confidence(elapsed, total_dev, time_to_stability)
        action     = _action(elapsed, confidence, time_to_stability)

        sensor_status = (
            "stable"     if total_dev < 0.3 else
            "converging" if total_dev < 1.0 else
            "transient"
        )

        out = {
            "timestamp":              timestamp,
            "elapsed_min":            round(elapsed, 2),
            "n_readings":             n_readings,
            "time_to_stability_min":  round(time_to_stability, 2),
            "predicted_stable_min":   round(predicted_stable_min, 1),
            "recommended_end_min":    round(recommended_end_min, 1),
            "time_saved_min":         round(time_saved_min, 1),
            "confidence_pct":         confidence,
            "sensor_status":          sensor_status,
            "total_deviation":        round(total_dev, 4),
            "action":                 action,
            "current_sensors":        {s: round(float(window_df[s].iloc[-1]), 3) for s in self.sensors},
            "predicted_sensors":      sensor_forecast,
        }
        if self.cv_mae is not None:
            out["model_mae_min"] = round(float(self.cv_mae), 2)
            out["time_to_stability_band_min"] = (
                round(max(time_to_stability - float(self.cv_mae), 0.0), 2),
                round(time_to_stability + float(self.cv_mae), 2),
            )
        return out


# ============================================================
# REPLAY MODE
# ============================================================
def replay(engine: CompressorInference, data_path: str, out_path: str = "inference_log.csv"):
    """
    Run inference on every row of a historical file and save results.
    """
    log("Starting replay...")

    df = pd.read_excel(data_path) if data_path.endswith(".xlsx") else pd.read_csv(data_path)
    df = df.sort_values("elapsed_time_min").reset_index(drop=True)

    log(f"Loaded {len(df)} rows from {data_path}")

    records = []
    for i in range(len(df)):
        window    = df.iloc[:i + 1]
        ts        = str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else None
        result    = engine.predict(window, ts)
        records.append(result)

        if i % 20 == 0:
            print("\n" + "=" * 60)
            log(f"Elapsed:          {result.get('elapsed_min', 0):.1f} min")
            log(f"Time to stable:   {result.get('time_to_stability_min', '—')}")
            log(f"Confidence:       {result.get('confidence_pct', 0):.1f}%")
            log(f"Action:           {result.get('action', '—')}")
            if "predicted_sensors" in result:
                log("Predicted sensors (next horizon):")
                for k, v in result["predicted_sensors"].items():
                    print(f"   {k}: {v}")

    log_df = pd.DataFrame(records)
    log_df.to_csv(out_path, index=False)
    log(f"Inference log saved → {out_path}")

    # Summary table at key windows
    print("\n── Prediction Summary (key observation windows) ──")
    checkpoints = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    for cp in checkpoints:
        rows = log_df[log_df["elapsed_min"].between(cp - 0.5, cp + 0.5)]
        if not rows.empty:
            r = rows.iloc[0]
            print(
                f"  t={r['elapsed_min']:5.1f} min | "
                f"TTS={r.get('time_to_stability_min','—'):6} | "
                f"stable@{r.get('predicted_stable_min','—'):6} | "
                f"conf={r.get('confidence_pct',0):5.1f}% | "
                f"action={r.get('action','—')}"
            )

    return log_df


# ============================================================
# REAL-TIME STREAMING MODE
# ============================================================
def run_realtime(engine: CompressorInference, data_path: str, poll_interval: float = 2.0):
    """
    Poll a growing data file every poll_interval seconds.
    New rows are appended to a rolling buffer and inference runs
    on each new row as it arrives.
    """
    log(f"Starting REAL-TIME inference | polling every {poll_interval}s")
    log(f"Watching: {data_path}")

    seen_rows = 0
    buffer_df = pd.DataFrame()

    while True:
        try:
            df = pd.read_excel(data_path) if data_path.endswith(".xlsx") else pd.read_csv(data_path)
            df = df.sort_values("elapsed_time_min").reset_index(drop=True)

            if len(df) > seen_rows:
                new_rows = df.iloc[seen_rows:]
                seen_rows = len(df)

                for _, row in new_rows.iterrows():
                    buffer_df = pd.concat([buffer_df, pd.DataFrame([row])], ignore_index=True)
                    buffer_df = buffer_df.tail(200)  # keep rolling window

                    result = engine.predict(buffer_df)

                    print("\n" + "=" * 60)
                    log(f"Elapsed:         {result.get('elapsed_min', 0):.1f} min")
                    log(f"Time to stable:  {result.get('time_to_stability_min', '—')}")
                    log(f"Confidence:      {result.get('confidence_pct', 0):.1f}%")
                    log(f"Action:          {result.get('action', '—')}")
                    log(f"Sensor status:   {result.get('sensor_status', '—')}")

                    if "predicted_sensors" in result:
                        log("Predicted sensors (next horizon):")
                        for k, v in result["predicted_sensors"].items():
                            print(f"   {k}: {v}")

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            log("Real-time mode stopped by user.")
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(poll_interval)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compressor stability inference engine")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--mode",  default="replay", choices=["replay", "realtime"],
                        help="'replay' to run on a full historical file, "
                             "'realtime' to poll a growing file")
    parser.add_argument("--out",   default="inference_log.csv",
                        help="Output CSV path for replay mode")
    parser.add_argument("--poll",  type=float, default=2.0,
                        help="Poll interval in seconds for realtime mode")
    args = parser.parse_args()

    engine = CompressorInference(args.model)

    if args.mode == "realtime":
        run_realtime(engine, args.data, poll_interval=args.poll)
    else:
        replay(engine, args.data, out_path=args.out)