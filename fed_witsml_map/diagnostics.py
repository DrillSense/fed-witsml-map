"""Post-prediction diagnostics: unit inference and mis-mapping detection.

Given a model's predicted property class and actual curve statistics (mean,
std, min, max), this module:

  1. **Infers the most likely unit of measure** by checking which unit's
     expected value range best contains the observed data.
  2. **Flags likely mis-mapped channels** when the observed statistics are
     outside the expected range for *all* known units of the predicted
     property — suggesting the mnemonic was probably assigned to the wrong
     physical measurement.

Usage:
    from fed_witsml_map.diagnostics import diagnose_channel

    result = diagnose_channel(
        predicted_property="gamma_ray",
        stats={"mean": 65.2, "std": 22.1, "min": 8.0, "max": 198.0},
        declared_unit="GAPI",
    )
    # result.likely_unit   -> "GAPI"
    # result.unit_match    -> True
    # result.quality_flags -> []
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mnemonic_catalog import PROPERTY_CLASSES


# ---------------------------------------------------------------------------
# Expected value ranges per property class per unit.
# (min_typical, max_typical) — the range where 95 %+ of real-world values
# should fall under normal operating conditions.
#
# Sources: PWLS, SLB CMD, GDR standard, Volve dataset statistics.
# ---------------------------------------------------------------------------

EXPECTED_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "measured_depth": {
        "M": (0, 12000),
        "FT": (0, 40000),
    },
    "true_vertical_depth": {
        "M": (0, 10000),
        "FT": (0, 35000),
    },
    "time": {
        "S": (0, 1e9),
        "HR": (0, 1e6),
    },
    "bit_depth": {
        "M": (0, 12000),
        "FT": (0, 40000),
    },
    "hole_depth": {
        "M": (0, 12000),
        "FT": (0, 40000),
    },
    "block_position": {
        "M": (0, 50),
        "FT": (0, 160),
    },
    "weight_on_bit": {
        "KLB": (0, 80),
        "KN": (0, 350),
    },
    "hookload": {
        "KLB": (0, 600),
        "KN": (0, 2700),
    },
    "rotary_speed": {
        "RPM": (0, 250),
    },
    "rotary_torque": {
        "KFTLB": (0, 50),
        "KNM": (0, 70),
    },
    "rate_of_penetration": {
        "FT/HR": (0, 500),
        "M/HR": (0, 150),
    },
    "standpipe_pressure": {
        "PSI": (0, 7000),
        "KPA": (0, 50000),
        "BAR": (0, 500),
    },
    "pump_rate": {
        "GPM": (0, 1200),
        "LPM": (0, 4500),
    },
    "pump_strokes": {
        "SPM": (0, 200),
        "STROKES": (0, 1e8),
    },
    "mud_flow_in": {
        "GPM": (0, 1200),
        "LPM": (0, 4500),
    },
    "mud_flow_out": {
        "GPM": (0, 1200),
        "PERCENT": (0, 200),
    },
    "mud_weight_in": {
        "PPG": (6, 22),
        "KG/M3": (700, 2700),
        "SG": (0.7, 2.7),
    },
    "mud_weight_out": {
        "PPG": (6, 22),
        "KG/M3": (700, 2700),
        "SG": (0.7, 2.7),
    },
    "mud_temperature_in": {
        "DEGF": (40, 400),
        "DEGC": (5, 200),
    },
    "mud_temperature_out": {
        "DEGF": (40, 400),
        "DEGC": (5, 200),
    },
    "total_gas": {
        "PERCENT": (0, 100),
        "UNITS": (0, 100000),
        "PPM": (0, 1e6),
    },
    "rig_activity": {
        "UNITLESS": (0, 50),
    },
    "gamma_ray": {
        "GAPI": (0, 300),
    },
    "bulk_density": {
        "G/CC": (1.0, 3.5),
        "KG/M3": (1000, 3500),
    },
    "neutron_porosity": {
        "V/V": (-0.15, 1.0),
        "PU": (-15, 100),
    },
    "deep_resistivity": {
        "OHMM": (0.01, 100000),
    },
    "shallow_resistivity": {
        "OHMM": (0.01, 100000),
    },
    "sonic_compressional": {
        "US/FT": (30, 200),
        "US/M": (100, 650),
    },
    "sonic_shear": {
        "US/FT": (50, 500),
        "US/M": (170, 1650),
    },
    "caliper": {
        "IN": (2, 30),
        "MM": (50, 760),
    },
    "spontaneous_potential": {
        "MV": (-200, 200),
    },
    "photoelectric_factor": {
        "B/E": (0, 10),
    },
    "inclination": {
        "DEG": (0, 180),
    },
    "azimuth": {
        "DEG": (0, 360),
    },
    "dogleg_severity": {
        "DEG/100FT": (0, 30),
        "DEG/30M": (0, 30),
    },
}


@dataclass
class ChannelDiagnosis:
    """Result of diagnosing a single channel."""

    predicted_property: str
    confidence: float
    declared_unit: str | None
    likely_unit: str | None
    unit_match: bool
    quality_flags: list[str] = field(default_factory=list)


def _value_in_range(
    stats: dict[str, float],
    lo: float,
    hi: float,
    tolerance: float = 0.3,
) -> float:
    """Score how well the observed statistics fit within [lo, hi].

    Returns a score between 0.0 (completely outside) and 1.0 (perfect fit).
    The tolerance factor widens the range to account for outliers.
    """
    expanded_lo = lo - tolerance * abs(hi - lo)
    expanded_hi = hi + tolerance * abs(hi - lo)
    obs_min = stats.get("min", stats.get("mean", 0) - 2 * stats.get("std", 0))
    obs_max = stats.get("max", stats.get("mean", 0) + 2 * stats.get("std", 0))
    obs_mean = stats.get("mean", (obs_min + obs_max) / 2)

    if obs_mean < expanded_lo or obs_mean > expanded_hi:
        return 0.0

    # Constant channel (stuck sensor): just check if the single value is in range
    if abs(obs_max - obs_min) < 1e-10:
        return 1.0 if expanded_lo <= obs_mean <= expanded_hi else 0.0

    in_range = max(0, min(obs_max, expanded_hi) - max(obs_min, expanded_lo))
    total_span = max(obs_max - obs_min, 1e-12)
    return min(in_range / total_span, 1.0)


def infer_unit(
    predicted_property: str,
    stats: dict[str, float],
) -> str | None:
    """Infer the most likely unit of measure from curve statistics.

    Checks the observed value range against the expected ranges for each
    known unit of the predicted property class and returns the best match.
    Returns None if no unit is a reasonable fit.
    """
    ranges = EXPECTED_RANGES.get(predicted_property, {})
    if not ranges:
        return None

    best_unit = None
    best_score = 0.0
    for unit, (lo, hi) in ranges.items():
        score = _value_in_range(stats, lo, hi)
        if score > best_score:
            best_score = score
            best_unit = unit

    return best_unit if best_score > 0.1 else None


def detect_mismatch(
    predicted_property: str,
    stats: dict[str, float],
    declared_unit: str | None = None,
) -> list[str]:
    """Detect likely data quality issues for a channel.

    Returns a list of human-readable quality flags. An empty list means
    no issues detected.
    """
    flags: list[str] = []
    ranges = EXPECTED_RANGES.get(predicted_property, {})
    if not ranges:
        return flags

    obs_mean = stats.get("mean")
    obs_min = stats.get("min")
    obs_max = stats.get("max")
    obs_std = stats.get("std")

    # Check if values fit ANY known unit for this property
    any_fit = False
    for unit, (lo, hi) in ranges.items():
        if _value_in_range(stats, lo, hi) > 0.1:
            any_fit = True
            break

    if not any_fit and obs_mean is not None:
        flags.append(
            f"VALUES_OUT_OF_RANGE: mean={obs_mean:.2f} does not match any known "
            f"unit for {predicted_property}. Possible mis-mapped channel."
        )

    # Check declared unit specifically
    if declared_unit and declared_unit.upper() in ranges:
        lo, hi = ranges[declared_unit.upper()]
        if _value_in_range(stats, lo, hi) < 0.1:
            likely = infer_unit(predicted_property, stats)
            hint = f" (values suggest {likely})" if likely else ""
            flags.append(
                f"UNIT_MISMATCH: declared unit {declared_unit} but values "
                f"outside expected range [{lo}, {hi}]{hint}."
            )

    # Check for constant/stuck sensor
    if obs_std is not None and obs_std < 1e-10 and obs_mean is not None:
        flags.append(
            f"STUCK_SENSOR: std={obs_std:.6f}, channel appears to be constant "
            f"at {obs_mean:.2f}."
        )

    # Check for negative values where they shouldn't exist
    non_negative_properties = {
        "measured_depth", "true_vertical_depth", "bit_depth", "hole_depth",
        "block_position", "weight_on_bit", "hookload", "rotary_speed",
        "pump_rate", "pump_strokes", "gamma_ray", "caliper",
        "photoelectric_factor", "inclination",
    }
    if predicted_property in non_negative_properties and obs_min is not None and obs_min < -1:
        flags.append(
            f"NEGATIVE_VALUES: min={obs_min:.2f} for {predicted_property} "
            f"which should be non-negative."
        )

    return flags


def diagnose_channel(
    predicted_property: str,
    confidence: float,
    stats: dict[str, float],
    declared_unit: str | None = None,
) -> ChannelDiagnosis:
    """Run full diagnostics on a single channel.

    Args:
        predicted_property: Property class predicted by the model.
        confidence: Model prediction confidence (0-1).
        stats: Curve statistics dict with keys: mean, std, min, max.
        declared_unit: The unit string from the WITSML/LAS header (if available).

    Returns:
        ChannelDiagnosis with inferred unit, match status, and quality flags.
    """
    likely_unit = infer_unit(predicted_property, stats)
    quality_flags = detect_mismatch(predicted_property, stats, declared_unit)

    unit_match = True
    if declared_unit and likely_unit:
        unit_match = declared_unit.upper() == likely_unit.upper()

    if confidence < 0.5:
        quality_flags.append(
            f"LOW_CONFIDENCE: model confidence {confidence:.2f} — "
            f"mnemonic may be ambiguous or unknown."
        )

    return ChannelDiagnosis(
        predicted_property=predicted_property,
        confidence=confidence,
        declared_unit=declared_unit,
        likely_unit=likely_unit,
        unit_match=unit_match,
        quality_flags=quality_flags,
    )
