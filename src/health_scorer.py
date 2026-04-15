"""
health_scorer.py — Production-grade Compressor Health Scoring Engine
====================================================================

Upgrades the original deviation-based health score with:
 1. Temporal awareness        – rolling drift detection (15-60 min)
 2. Derived sensor metrics    – efficiency, thermal load ratios
 3. Context-aware baselines   – smooth blend between stable_ref and forecast
 4. Confidence scoring        – based on sensor availability & data quality
 5. Robustness                – noise deadband + deviation clamping
 6. Explainability            – top drivers + human-readable interpretation
 7. Degradation vs anomaly    – persistent trend vs transient spike separation
 8. Rule-based interpretation – pattern → likely root cause mapping
 9. Persistence logic         – warnings only after N consecutive violations
10. Rich outputs              – full structured dict per tick

All computation is O(1) per tick (uses fixed-size ring buffers).
No ML models — fully interpretable, real-time safe.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

_HEALTH_SENSORS = (
    "airend_discharge_temp_c",
    "fad_cfm",
    "motor_output_power_kw",
)

_DEFAULT_WEIGHTS: dict[str, float] = {
    "airend_discharge_temp_c": 0.30,
    "fad_cfm":                 0.25,
    "motor_output_power_kw":   0.25,
    "efficiency_ratio":        0.10,
    "thermal_load_ratio":      0.10,
}

SCALING_FACTOR = 12.5

# Noise deadband: deviations below this (in sigma) are treated as zero.
NOISE_DEADBAND_SIGMA = 0.15

# Deviation clamp: cap extreme deviations to prevent single-sensor blowout.
MAX_DEVIATION_SIGMA = 8.0

# Drift detection window sizes (in number of ticks).
DRIFT_WINDOW_SHORT = 15
DRIFT_WINDOW_LONG = 60

# Blend weight for combining instantaneous score and drift score.
INSTANT_WEIGHT = 0.70
DRIFT_WEIGHT = 0.30

# Persistence: how many consecutive ticks a condition must hold
# before triggering a band change.
PERSISTENCE_TICKS_WARNING = 3
PERSISTENCE_TICKS_CRITICAL = 2

# EMA default alpha.
DEFAULT_EMA_ALPHA = 0.2

# Baseline blend: how much to trust forecast vs stable_ref
# (1.0 = pure forecast, 0.0 = pure stable_ref).
# Ramps from 0 → this value as matured forecasts become available.
MAX_FORECAST_BLEND = 0.6

# Band thresholds (on smoothed score).
BAND_HEALTHY_THRESHOLD = 80.0
BAND_WARNING_THRESHOLD = 60.0


# ═══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

_INTERPRETATION_RULES: list[dict[str, Any]] = [
    {
        "condition": lambda devs: devs.get("airend_discharge_temp_c", 0) > 2.0,
        "label": "high_discharge_temp",
        "cause": "Possible cooling system degradation or ambient heat rise",
        "severity": "warning",
    },
    {
        "condition": lambda devs: devs.get("airend_discharge_temp_c", 0) > 4.0,
        "label": "critical_discharge_temp",
        "cause": "Cooling failure likely — check oil cooler, coolant level, fan",
        "severity": "critical",
    },
    {
        "condition": lambda devs: devs.get("fad_cfm", 0) > 2.0,
        "label": "low_flow",
        "cause": "Reduced air delivery — check inlet filter, valve seating, or demand mismatch",
        "severity": "warning",
    },
    {
        "condition": lambda devs: devs.get("motor_output_power_kw", 0) > 2.5,
        "label": "high_motor_power",
        "cause": "Excess power draw — check for mechanical drag, bearing wear, or overloaded stage",
        "severity": "warning",
    },
    {
        "condition": lambda devs: (
            devs.get("efficiency_ratio", 0) > 1.5
        ),
        "label": "low_efficiency",
        "cause": "Degraded volumetric or mechanical efficiency — service due",
        "severity": "warning",
    },
    {
        "condition": lambda devs: (
            devs.get("thermal_load_ratio", 0) > 2.0
        ),
        "label": "high_thermal_load",
        "cause": "Disproportionate heat generation per unit power — check compression ratio or intercooler",
        "severity": "warning",
    },
    {
        "condition": lambda devs: (
            devs.get("airend_discharge_temp_c", 0) > 1.5
            and devs.get("motor_output_power_kw", 0) > 1.5
        ),
        "label": "temp_power_correlation",
        "cause": "Simultaneous temp & power rise — suspect bearing or lubrication issue",
        "severity": "warning",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: sensor scale (2-tier, same logic as fastapi_app)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sensor_scale(
    stable_std: dict[str, float],
    stable_ref: dict[str, float],
    sensor: str,
) -> float:
    MIN_STD_FRACTION = 0.03
    RANGE_PROXY_PCT = 0.05

    std = stable_std.get(sensor)
    ref = stable_ref.get(sensor)

    try:
        std = float(std) if std is not None else None
    except (TypeError, ValueError):
        std = None
    try:
        ref = float(ref) if ref is not None else None
    except (TypeError, ValueError):
        ref = None

    if std is not None and std > 0 and ref is not None and abs(ref) > 1e-6:
        if std / abs(ref) >= MIN_STD_FRACTION:
            return std
        return abs(ref) * RANGE_PROXY_PCT

    if std is not None and std > 0:
        return std

    if ref is not None and abs(ref) > 1e-6:
        return abs(ref) * RANGE_PROXY_PCT

    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  RING BUFFER for rolling stats
# ═══════════════════════════════════════════════════════════════════════════════

class _RingBuffer:
    """Fixed-size ring buffer with O(1) append and incremental mean/slope."""

    __slots__ = ("_buf", "_maxlen")

    def __init__(self, maxlen: int):
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def append(self, value: float) -> None:
        self._buf.append(value)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def full(self) -> bool:
        return len(self._buf) == self._maxlen

    def mean(self) -> float | None:
        if not self._buf:
            return None
        return sum(self._buf) / len(self._buf)

    def slope(self) -> float | None:
        """Least-squares slope over the buffer (units: per-tick)."""
        n = len(self._buf)
        if n < 3:
            return None
        sx = sy = sxy = sx2 = 0.0
        for i, v in enumerate(self._buf):
            sx += i
            sy += v
            sxy += i * v
            sx2 += i * i
        denom = n * sx2 - sx * sx
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sxy - sx * sy) / denom

    def last(self) -> float | None:
        return self._buf[-1] if self._buf else None

    def values(self) -> list[float]:
        return list(self._buf)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SCORER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CompressorHealthScorer:
    """
    Production-grade health scorer for compressor digital twin.

    Instantiate once per session. Call `score()` on every sensor tick.
    Call `reset()` when the session/test restarts.

    Parameters
    ----------
    stable_ref : dict   — {sensor: reference_value} from training.
    stable_std : dict   — {sensor: std_deviation} from training.
    sensors    : list   — ordered sensor names from the model bundle.
    """

    def __init__(
        self,
        stable_ref: dict[str, float],
        stable_std: dict[str, float],
        sensors: list[str],
    ):
        self.stable_ref = dict(stable_ref)
        self.stable_std = dict(stable_std)
        self.all_sensors = list(sensors)

        # Resolve which health sensors are available in this model.
        engine_set = set(sensors)
        self.health_sensors = [s for s in _HEALTH_SENSORS if s in engine_set]

        # Derived metric sensors — always present if base sensors exist.
        self.has_efficiency = (
            "fad_cfm" in engine_set and "motor_output_power_kw" in engine_set
        )
        self.has_thermal_load = (
            "airend_discharge_temp_c" in engine_set
            and "motor_output_power_kw" in engine_set
        )

        # Build the full list of scorable channels.
        self.scored_channels: list[str] = list(self.health_sensors)
        if self.has_efficiency:
            self.scored_channels.append("efficiency_ratio")
        if self.has_thermal_load:
            self.scored_channels.append("thermal_load_ratio")

        # Load weights (env-overridable).
        self.weights = self._load_weights()

        # Load EMA alpha (env-overridable).
        alpha_raw = os.environ.get("COMPRESSOR_HEALTH_EMA_ALPHA", "")
        try:
            self.ema_alpha = float(alpha_raw)
            if not (0.0 < self.ema_alpha <= 1.0):
                self.ema_alpha = DEFAULT_EMA_ALPHA
        except (TypeError, ValueError):
            self.ema_alpha = DEFAULT_EMA_ALPHA

        # Expected-value mode.
        self.expected_mode = os.environ.get(
            "COMPRESSOR_HEALTH_EXPECTED_MODE", "forecast_matured"
        ).strip().lower()
        if self.expected_mode not in ("forecast_matured", "stable_ref", "forecast"):
            self.expected_mode = "forecast_matured"

        # Compute reference values for derived metrics.
        self._ref_efficiency: float | None = None
        self._ref_thermal_load: float | None = None
        self._scale_efficiency: float = 1.0
        self._scale_thermal_load: float = 1.0
        self._compute_derived_refs()

        # Pre-compute per-sensor scales.
        self.scales: dict[str, float] = {}
        for s in self.health_sensors:
            self.scales[s] = _get_sensor_scale(self.stable_std, self.stable_ref, s)
        if self.has_efficiency and self._ref_efficiency:
            self.scales["efficiency_ratio"] = self._scale_efficiency
        if self.has_thermal_load and self._ref_thermal_load:
            self.scales["thermal_load_ratio"] = self._scale_thermal_load

        # ── Stateful accumulators ──
        self._ema: float | None = None
        self._tick_count: int = 0

        # Forecast maturation queue.
        self._forecast_queue: list[dict[str, Any]] = []

        # Per-channel rolling buffers for drift detection.
        self._drift_short: dict[str, _RingBuffer] = {
            ch: _RingBuffer(DRIFT_WINDOW_SHORT) for ch in self.scored_channels
        }
        self._drift_long: dict[str, _RingBuffer] = {
            ch: _RingBuffer(DRIFT_WINDOW_LONG) for ch in self.scored_channels
        }

        # Score history for persistence logic.
        self._band_history: deque[str] = deque(
            maxlen=max(PERSISTENCE_TICKS_WARNING, PERSISTENCE_TICKS_CRITICAL) + 1
        )
        self._confirmed_band: str = "Healthy"

        # Raw score history for drift-of-score.
        self._score_history: _RingBuffer = _RingBuffer(DRIFT_WINDOW_LONG)

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: derived refs
    # ───────────────────────────────────────────────────────────────────────

    def _compute_derived_refs(self) -> None:
        fad_ref = self.stable_ref.get("fad_cfm")
        power_ref = self.stable_ref.get("motor_output_power_kw")
        temp_ref = self.stable_ref.get("airend_discharge_temp_c")

        if fad_ref and power_ref and power_ref > 1e-6:
            self._ref_efficiency = fad_ref / power_ref
            self._scale_efficiency = abs(self._ref_efficiency) * 0.05
            if self._scale_efficiency < 1e-6:
                self._scale_efficiency = 1.0

        if temp_ref and power_ref and power_ref > 1e-6:
            self._ref_thermal_load = temp_ref / power_ref
            self._scale_thermal_load = abs(self._ref_thermal_load) * 0.05
            if self._scale_thermal_load < 1e-6:
                self._scale_thermal_load = 1.0

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: weight loading
    # ───────────────────────────────────────────────────────────────────────

    def _load_weights(self) -> dict[str, float]:
        raw = os.environ.get("COMPRESSOR_HEALTH_WEIGHTS", "").strip()
        w: dict[str, float] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        try:
                            w[str(k)] = float(v)
                        except (TypeError, ValueError):
                            continue
            except json.JSONDecodeError:
                w = {}

        if not w:
            w = {
                ch: _DEFAULT_WEIGHTS.get(ch, 0.0)
                for ch in self.scored_channels
            }

        w = {k: v for k, v in w.items() if k in self.scored_channels and v > 0}
        if not w:
            w = {ch: 1.0 for ch in self.scored_channels}

        w_sum = sum(w.values()) or 1.0
        return {k: v / w_sum for k, v in w.items()}

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: expected value resolution with smooth blending
    # ───────────────────────────────────────────────────────────────────────

    def _resolve_expected(
        self,
        sensor: str,
        matured_pred: dict | None,
        pred: dict,
    ) -> float | None:
        ref_val = self.stable_ref.get(sensor)

        if self.expected_mode == "stable_ref":
            return float(ref_val) if ref_val is not None else None

        if self.expected_mode == "forecast":
            v = pred.get(sensor)
            return float(v) if v is not None else None

        # forecast_matured with smooth blending
        forecast_val = (matured_pred or {}).get(sensor)

        if ref_val is None and forecast_val is None:
            return None
        if ref_val is None:
            return float(forecast_val)
        if forecast_val is None:
            return float(ref_val)

        # Blend: ramp from pure stable_ref toward forecast as ticks accumulate.
        # After DRIFT_WINDOW_LONG ticks the blend reaches MAX_FORECAST_BLEND.
        ramp = min(self._tick_count / max(DRIFT_WINDOW_LONG, 1), 1.0)
        blend = ramp * MAX_FORECAST_BLEND
        return (1.0 - blend) * float(ref_val) + blend * float(forecast_val)

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: derived metric computation
    # ───────────────────────────────────────────────────────────────────────

    def _compute_derived(
        self, cur: dict[str, float]
    ) -> dict[str, float | None]:
        derived: dict[str, float | None] = {}

        fad = cur.get("fad_cfm")
        power = cur.get("motor_output_power_kw")
        temp = cur.get("airend_discharge_temp_c")

        if self.has_efficiency and fad is not None and power is not None and power > 1e-6:
            derived["efficiency_ratio"] = fad / power
        else:
            derived["efficiency_ratio"] = None

        if self.has_thermal_load and temp is not None and power is not None and power > 1e-6:
            derived["thermal_load_ratio"] = temp / power
        else:
            derived["thermal_load_ratio"] = None

        return derived

    def _derived_expected(
        self, matured_pred: dict | None, pred: dict
    ) -> dict[str, float | None]:
        """Compute expected values for derived metrics using the same blend logic."""
        result: dict[str, float | None] = {}

        if self.has_efficiency:
            result["efficiency_ratio"] = (
                float(self._ref_efficiency)
                if self._ref_efficiency is not None
                else None
            )
        if self.has_thermal_load:
            result["thermal_load_ratio"] = (
                float(self._ref_thermal_load)
                if self._ref_thermal_load is not None
                else None
            )

        return result

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: forecast queue management
    # ───────────────────────────────────────────────────────────────────────

    def _update_forecast_queue(
        self, n_readings: int | None, pred: dict, future_horizon: int
    ) -> None:
        if isinstance(n_readings, int) and pred:
            self._forecast_queue.append({
                "due_n": n_readings + future_horizon,
                "pred": dict(pred),
            })
            if len(self._forecast_queue) > 1200:
                self._forecast_queue = self._forecast_queue[-800:]

    def _pop_matured_forecast(self, n_readings: int | None) -> dict | None:
        if not isinstance(n_readings, int):
            return None
        due = sorted(
            [f for f in self._forecast_queue if f["due_n"] <= n_readings],
            key=lambda x: x["due_n"],
        )
        if not due:
            return None
        result = due[0].get("pred")
        cut = due[0]["due_n"]
        self._forecast_queue = [
            f for f in self._forecast_queue if f["due_n"] > cut
        ]
        return result if isinstance(result, dict) else None

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: drift scoring
    # ───────────────────────────────────────────────────────────────────────

    def _compute_drift_score(self) -> tuple[float, dict[str, float | None]]:
        """
        Drift score: penalise channels whose deviation is trending upward.

        Returns (drift_penalty_0_to_100, per_channel_slope_sigma_per_tick).
        """
        channel_slopes: dict[str, float | None] = {}
        drift_penalty = 0.0

        for ch in self.scored_channels:
            w = self.weights.get(ch, 0.0)
            slope_short = self._drift_short[ch].slope()
            slope_long = self._drift_long[ch].slope()

            if slope_short is None and slope_long is None:
                channel_slopes[ch] = None
                continue

            slope = slope_short if slope_short is not None else slope_long
            channel_slopes[ch] = round(slope, 6) if slope is not None else None

            # Only penalise worsening trends (positive slope = growing deviation).
            if slope is not None and slope > 0:
                # Scale slope into a meaningful penalty.
                # A slope of 0.1 sigma/tick over 15 ticks = 1.5 sigma drift.
                projected_drift = abs(slope) * DRIFT_WINDOW_SHORT
                drift_penalty += w * projected_drift * SCALING_FACTOR

        drift_score_loss = min(drift_penalty, 100.0)
        return drift_score_loss, channel_slopes

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: confidence
    # ───────────────────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        valid_sensors: int,
        total_sensors: int,
    ) -> tuple[float, bool]:
        """
        Confidence based on sensor availability.
        Returns (confidence_0_to_1, low_confidence_flag).
        """
        if total_sensors == 0:
            return 0.0, True

        coverage = valid_sensors / total_sensors

        # Ramp confidence with tick count (data maturity).
        maturity = min(self._tick_count / 30.0, 1.0)

        confidence = coverage * 0.7 + maturity * 0.3
        confidence = max(0.0, min(1.0, confidence))

        low = confidence < 0.5
        return round(confidence, 4), low

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: persistence
    # ───────────────────────────────────────────────────────────────────────

    def _apply_persistence(self, raw_band: str) -> str:
        self._band_history.append(raw_band)
        history = list(self._band_history)

        if raw_band == "Critical":
            recent = history[-PERSISTENCE_TICKS_CRITICAL:]
            if len(recent) >= PERSISTENCE_TICKS_CRITICAL and all(
                b == "Critical" for b in recent
            ):
                self._confirmed_band = "Critical"
                return "Critical"

        if raw_band in ("Warning", "Critical"):
            n = PERSISTENCE_TICKS_WARNING
            recent = history[-n:]
            if len(recent) >= n and all(b != "Healthy" for b in recent):
                self._confirmed_band = raw_band
                return raw_band

        if raw_band == "Healthy" and self._confirmed_band != "Healthy":
            n = PERSISTENCE_TICKS_WARNING
            recent = history[-n:]
            if len(recent) >= n and all(b == "Healthy" for b in recent):
                self._confirmed_band = "Healthy"
                return "Healthy"
            return self._confirmed_band

        return self._confirmed_band

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: explainability
    # ───────────────────────────────────────────────────────────────────────

    def _explain(
        self,
        deviations: dict[str, float | None],
        contrib: dict[str, float | None],
        band: str,
    ) -> tuple[str | None, str]:
        """
        Identify top contributing sensor and produce a human-readable explanation.
        """
        valid = {k: v for k, v in contrib.items() if v is not None and v > 0}
        if not valid:
            return None, "All sensors within normal operating range."

        top_driver = max(valid, key=lambda k: valid[k])
        top_dev = deviations.get(top_driver, 0) or 0

        _FRIENDLY = {
            "airend_discharge_temp_c": "discharge temperature",
            "fad_cfm": "free air delivery (FAD)",
            "motor_output_power_kw": "motor power",
            "efficiency_ratio": "efficiency ratio (FAD/power)",
            "thermal_load_ratio": "thermal load ratio (temp/power)",
        }
        friendly = _FRIENDLY.get(top_driver, top_driver)

        if band == "Critical":
            explanation = (
                f"CRITICAL: {friendly} is {top_dev:.1f}sigma from expected — "
                f"immediate attention required."
            )
        elif band == "Warning":
            explanation = (
                f"WARNING: {friendly} deviating at {top_dev:.1f}sigma — "
                f"monitor closely for further degradation."
            )
        else:
            explanation = f"Healthy operation. Largest contributor: {friendly} at {top_dev:.1f}sigma."

        return top_driver, explanation

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: rule-based interpretation
    # ───────────────────────────────────────────────────────────────────────

    def _interpret_rules(
        self, signed_deviations: dict[str, float]
    ) -> list[dict[str, str]]:
        abs_devs = {k: abs(v) for k, v in signed_deviations.items()}
        triggered: list[dict[str, str]] = []
        for rule in _INTERPRETATION_RULES:
            try:
                if rule["condition"](abs_devs):
                    triggered.append({
                        "label": rule["label"],
                        "cause": rule["cause"],
                        "severity": rule["severity"],
                    })
            except Exception:
                continue
        return triggered

    # ───────────────────────────────────────────────────────────────────────
    #  PRIVATE: degradation vs anomaly classification
    # ───────────────────────────────────────────────────────────────────────

    def _classify_issue_type(
        self,
        channel: str,
        current_dev: float,
        slope: float | None,
    ) -> str:
        """
        Classify a channel's issue as:
          'normal'      — within deadband
          'transient'   — spike without trend
          'degradation' — persistent drift
        """
        if current_dev < NOISE_DEADBAND_SIGMA * 3:
            return "normal"

        if slope is not None and abs(slope) > 0.005:
            return "degradation"

        return "transient"

    # ═══════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Clear all stateful accumulators. Call when session restarts."""
        self._ema = None
        self._tick_count = 0
        self._forecast_queue.clear()
        for ch in self.scored_channels:
            self._drift_short[ch] = _RingBuffer(DRIFT_WINDOW_SHORT)
            self._drift_long[ch] = _RingBuffer(DRIFT_WINDOW_LONG)
        self._band_history.clear()
        self._confirmed_band = "Healthy"
        self._score_history = _RingBuffer(DRIFT_WINDOW_LONG)

    def score(
        self,
        result: dict[str, Any],
        engine_stable_ref: dict[str, float] | None = None,
        engine_stable_std: dict[str, float] | None = None,
        future_horizon: int = 10,
    ) -> dict[str, Any] | None:
        """
        Compute a full health assessment for one sensor tick.

        Parameters
        ----------
        result : dict
            The prediction result dict from CompressorInference.predict().
            Must contain 'current_sensors' and optionally 'predicted_sensors',
            'n_readings'.
        engine_stable_ref / engine_stable_std : dict, optional
            Override refs/stds (normally passed at construction).
        future_horizon : int
            Model's forecast horizon in ticks.

        Returns
        -------
        dict with all health fields, or None if scoring not possible.
        """
        cur = result.get("current_sensors") or {}
        if not cur:
            return None

        pred = result.get("predicted_sensors") or {}
        n_readings = result.get("n_readings")

        ref = engine_stable_ref or self.stable_ref
        std = engine_stable_std or self.stable_std

        self._tick_count += 1

        # ── 1. Update forecast queue ──────────────────────────────────────
        self._update_forecast_queue(n_readings, pred, future_horizon)
        matured_pred = (
            self._pop_matured_forecast(n_readings)
            if self.expected_mode == "forecast_matured"
            else None
        )

        # ── 2. Compute derived metrics ────────────────────────────────────
        derived = self._compute_derived(cur)
        derived_expected = self._derived_expected(matured_pred, pred)

        # ── 3. Per-channel deviation + contribution ───────────────────────
        deviations: dict[str, float | None] = {}
        signed_deviations: dict[str, float] = {}
        contrib: dict[str, float | None] = {}
        total_points = 0.0
        valid_count = 0
        total_count = len(self.scored_channels)

        eps = 1e-9

        for ch in self.scored_channels:
            weight = self.weights.get(ch, 0.0)
            scale = self.scales.get(ch, 1.0)

            # Get actual value
            if ch in ("efficiency_ratio", "thermal_load_ratio"):
                av = derived.get(ch)
                expected = derived_expected.get(ch)
            else:
                av = cur.get(ch)
                expected = self._resolve_expected(ch, matured_pred, pred)

            if av is None or expected is None:
                deviations[ch] = None
                contrib[ch] = None
                continue

            # Raw signed deviation (in sigma).
            raw_dev = (float(av) - float(expected)) / (eps + scale)
            signed_deviations[ch] = raw_dev

            # Absolute deviation with noise deadband and clamping.
            abs_dev = abs(raw_dev)
            if abs_dev < NOISE_DEADBAND_SIGMA:
                abs_dev = 0.0
            abs_dev = min(abs_dev, MAX_DEVIATION_SIGMA)

            deviations[ch] = round(abs_dev, 6)

            c = weight * abs_dev * SCALING_FACTOR
            contrib[ch] = round(c, 4)
            total_points += c
            valid_count += 1

            # Feed drift buffers.
            self._drift_short[ch].append(abs_dev)
            self._drift_long[ch].append(abs_dev)

        if valid_count == 0:
            return None

        # ── 4. Instantaneous score ────────────────────────────────────────
        instant_score = max(0.0, min(100.0, 100.0 - total_points))

        # ── 5. Drift score ────────────────────────────────────────────────
        drift_penalty, channel_slopes = self._compute_drift_score()
        drift_adjusted_score = max(0.0, min(100.0, 100.0 - drift_penalty))

        # ── 6. Combined score ─────────────────────────────────────────────
        combined_raw = (
            INSTANT_WEIGHT * instant_score + DRIFT_WEIGHT * drift_adjusted_score
        )
        score = max(0.0, min(100.0, combined_raw))

        self._score_history.append(score)

        # ── 7. EMA smoothing ─────────────────────────────────────────────
        if self._ema is None:
            self._ema = score
        else:
            self._ema = self.ema_alpha * score + (1.0 - self.ema_alpha) * self._ema
        smoothed = max(0.0, min(100.0, self._ema))

        # ── 8. Raw band ──────────────────────────────────────────────────
        if smoothed > BAND_HEALTHY_THRESHOLD:
            raw_band = "Healthy"
        elif smoothed >= BAND_WARNING_THRESHOLD:
            raw_band = "Warning"
        else:
            raw_band = "Critical"

        # ── 9. Persistence-filtered band ─────────────────────────────────
        confirmed_band = self._apply_persistence(raw_band)

        # ── 10. Confidence ───────────────────────────────────────────────
        confidence, low_confidence = self._compute_confidence(
            valid_count, total_count
        )

        # ── 11. Explainability ───────────────────────────────────────────
        top_driver, explanation = self._explain(deviations, contrib, confirmed_band)

        # ── 12. Rule-based interpretation ────────────────────────────────
        interpretations = self._interpret_rules(signed_deviations)

        # ── 13. Per-channel issue classification ─────────────────────────
        issue_types: dict[str, str] = {}
        for ch in self.scored_channels:
            dev = deviations.get(ch)
            slope = channel_slopes.get(ch)
            if dev is not None:
                issue_types[ch] = self._classify_issue_type(ch, dev, slope)
            else:
                issue_types[ch] = "unknown"

        # ── 14. Score trend ──────────────────────────────────────────────
        score_slope = self._score_history.slope()
        if score_slope is not None:
            if score_slope < -0.1:
                score_trend = "declining"
            elif score_slope > 0.1:
                score_trend = "improving"
            else:
                score_trend = "stable"
        else:
            score_trend = "insufficient_data"

        # ── Assemble output ──────────────────────────────────────────────
        return {
            "health_score": round(score, 2),
            "health_score_instant": round(instant_score, 2),
            "health_score_drift": round(drift_adjusted_score, 2),
            "health_score_smoothed": round(smoothed, 2),
            "health_band": confirmed_band,
            "health_band_raw": raw_band,
            "health_confidence": round(confidence, 4),
            "health_low_confidence": low_confidence,
            "health_deviation": {
                k: round(v, 6) if v is not None else None
                for k, v in deviations.items()
            },
            "health_signed_deviation": {
                k: round(v, 4) for k, v in signed_deviations.items()
            },
            "health_risk_contrib": {
                k: round(v, 4) if v is not None else None
                for k, v in contrib.items()
            },
            "health_valid_sensors": valid_count,
            "health_top_driver": top_driver,
            "health_explanation": explanation,
            "health_interpretations": interpretations,
            "health_issue_types": issue_types,
            "health_channel_slopes": {
                k: round(v, 6) if v is not None else None
                for k, v in channel_slopes.items()
            },
            "health_score_trend": score_trend,
            "health_derived_metrics": {
                k: round(v, 4) if v is not None else None
                for k, v in derived.items()
            },
        }
