"""WITSML mnemonic catalog and synthetic data generation.

Built from publicly available sources:
  - Energistics PWLS v3.0 (Practical Well Log Standard)
  - SLB Curve Mnemonic Dictionary (OSDD / apps.slb.com/cmd/)
  - WITS Specification Rev 1.1 (Petrospec)
  - GDR Drilling Data Standard (US DOE — Pason/RigCloud mappings)
  - lasmnemonicsid (Nobleza Energy, MIT license)
  - TotalDepth (Schlumberger OSDD lookup)

Each property class maps to a list of known (mnemonic, unit) pairs from
different vendors and legacy systems.  The vendor profiles simulate the
non-IID naming conventions that make federated learning valuable here:
a model trained on only one vendor's data cannot recognise another's
mnemonics for the same physical measurement.
"""

from __future__ import annotations

import numpy as np

PROPERTY_CLASSES: list[str] = [
    # --- Index / depth / time ---
    "measured_depth",
    "true_vertical_depth",
    "time",
    # --- Drilling surface (EDR / WITS Record 1) ---
    "bit_depth",
    "hole_depth",
    "block_position",
    "weight_on_bit",
    "hookload",
    "rotary_speed",
    "rotary_torque",
    "rate_of_penetration",
    "standpipe_pressure",
    "pump_rate",
    "pump_strokes",
    "mud_flow_in",
    "mud_flow_out",
    "mud_weight_in",
    "mud_weight_out",
    "mud_temperature_in",
    "mud_temperature_out",
    "total_gas",
    "rig_activity",
    # --- Formation evaluation / wireline / MWD-LWD ---
    "gamma_ray",
    "bulk_density",
    "neutron_porosity",
    "deep_resistivity",
    "shallow_resistivity",
    "sonic_compressional",
    "sonic_shear",
    "caliper",
    "spontaneous_potential",
    "photoelectric_factor",
    # --- Directional / survey ---
    "inclination",
    "azimuth",
    "dogleg_severity",
]

NUM_CLASSES = len(PROPERTY_CLASSES)
PROPERTY_TO_IDX = {p: i for i, p in enumerate(PROPERTY_CLASSES)}


# ---------------------------------------------------------------------------
# Known mnemonics per property class.
# Each entry is (mnemonic, typical_unit).
# Sources: PWLS, SLB CMD, WITS, GDR, lasmnemonicsid, public Halliburton /
#          Baker Hughes / Weatherford documentation.
# ---------------------------------------------------------------------------

MNEMONIC_DB: dict[str, list[tuple[str, str]]] = {
    "measured_depth": [
        ("DEPT", "M"), ("DEPTH", "M"), ("MD", "M"), ("MDEP", "M"),
        ("DEPTMD", "M"), ("DEPT_MD", "M"), ("DMEA", "M"),
        ("DEPT", "FT"), ("DEPTH", "FT"), ("MD", "FT"),
    ],
    "true_vertical_depth": [
        ("TVD", "M"), ("TVDSS", "M"), ("DEPTVERT", "M"), ("TVDM", "M"),
        ("TVD", "FT"), ("TVDSS", "FT"), ("DVERT", "FT"),
    ],
    "time": [
        ("TIME", "S"), ("ETIM", "S"), ("DATETIME", "S"), ("DTIM", "S"),
        ("TIME", "HR"), ("ETIME", "HR"), ("TIMESTAMP", "S"),
    ],
    "bit_depth": [
        ("DBIT", "M"), ("DMEA", "M"), ("BITDEP", "M"), ("DEPTBIT", "M"),
        ("DBIT", "FT"), ("BITDEPTH", "FT"), ("BIT_DEPTH", "FT"),
        ("BDEP", "M"), ("DPTS", "FT"),
    ],
    "hole_depth": [
        ("DHOLE", "M"), ("HDEP", "M"), ("HOLEDEP", "M"), ("HOLE_DEPTH", "M"),
        ("DHOLE", "FT"), ("HDEP", "FT"), ("DPTH", "FT"), ("DPHL", "FT"),
    ],
    "block_position": [
        ("BPOS", "M"), ("BLKPOS", "M"), ("BLOCK_POS", "M"),
        ("BPOS", "FT"), ("BLKP", "FT"), ("BLK_HT", "FT"),
        ("BLOCK_HEIGHT", "FT"), ("BKPH", "FT"),
    ],
    "weight_on_bit": [
        ("WOB", "KLB"), ("SWOB", "KLB"), ("WOBA", "KLB"), ("WOB_AVG", "KLB"),
        ("WOB", "KN"), ("WOBX", "KLB"), ("WOB_MAX", "KLB"),
        ("WOBSFC", "KLB"), ("WOB_SURFACE", "KLB"), ("SFC_WOB", "KLB"),
    ],
    "hookload": [
        ("HKLA", "KLB"), ("HKLD", "KLB"), ("HKL", "KLB"), ("HOOKLOAD", "KLB"),
        ("HKLX", "KLB"), ("HKL_AVG", "KLB"), ("HKL_MAX", "KLB"),
        ("HKLA", "KN"), ("HOOKLD", "KN"),
    ],
    "rotary_speed": [
        ("RPM", "RPM"), ("RPMA", "RPM"), ("SRPM", "RPM"), ("ROT_SPD", "RPM"),
        ("TDS_RPM", "RPM"), ("TDRPM", "RPM"), ("ROTARY_RPM", "RPM"),
        ("SURFACE_RPM", "RPM"), ("SFRPM", "RPM"), ("RPMX", "RPM"),
    ],
    "rotary_torque": [
        ("TRQ", "KFTLB"), ("TORA", "KFTLB"), ("TORQUE", "KFTLB"),
        ("STOR", "KFTLB"), ("TRQ_AVG", "KFTLB"), ("TORX", "KFTLB"),
        ("SFC_TRQ", "KFTLB"), ("TD_TRQ", "KFTLB"), ("TDTQ", "KNM"),
    ],
    "rate_of_penetration": [
        ("ROP", "FT/HR"), ("ROPA", "FT/HR"), ("ROP5", "FT/HR"),
        ("ROP1", "FT/HR"), ("ROPX", "FT/HR"), ("ROP_AVG", "FT/HR"),
        ("ROP", "M/HR"), ("ROPA", "M/HR"), ("ROPINST", "M/HR"),
    ],
    "standpipe_pressure": [
        ("SPP", "PSI"), ("SPPA", "PSI"), ("SPP_AVG", "PSI"), ("STKP", "PSI"),
        ("PPRS", "PSI"), ("SPRESS", "PSI"), ("STANDPIPE", "PSI"),
        ("SPP", "KPA"), ("SPPA", "KPA"), ("SPP", "BAR"),
    ],
    "pump_rate": [
        ("GPM", "GPM"), ("MFIA", "GPM"), ("FLOWIN", "GPM"),
        ("PUMP_RATE", "GPM"), ("TFLOW", "GPM"), ("PUMPS_GPM", "GPM"),
        ("GPM", "LPM"), ("FLOW_RATE", "LPM"), ("MFIR", "LPM"),
    ],
    "pump_strokes": [
        ("SPM", "SPM"), ("SPM1", "SPM"), ("SPM2", "SPM"), ("SPM3", "SPM"),
        ("SPMT", "SPM"), ("STK_RATE", "SPM"), ("PUMP_SPM", "SPM"),
        ("STKCUM", "STROKES"), ("TSTK", "STROKES"),
    ],
    "mud_flow_in": [
        ("MFIA", "GPM"), ("FLOW_IN", "GPM"), ("FLOWIN", "GPM"),
        ("MFIN", "GPM"), ("MFI", "GPM"), ("MUDFLOWIN", "GPM"),
        ("MFIA", "LPM"), ("FLOW_IN", "LPM"),
    ],
    "mud_flow_out": [
        ("MFOA", "GPM"), ("FLOW_OUT", "GPM"), ("FLOWOUT", "GPM"),
        ("MFOUT", "GPM"), ("FLWO", "GPM"), ("MFO", "GPM"),
        ("MFOP", "PERCENT"), ("FLOWOUT_PCT", "PERCENT"),
    ],
    "mud_weight_in": [
        ("MWI", "PPG"), ("MWIN", "PPG"), ("MW_IN", "PPG"),
        ("MUDIN", "PPG"), ("MUD_WT_IN", "PPG"),
        ("MWI", "KG/M3"), ("MWIN", "SG"),
    ],
    "mud_weight_out": [
        ("MWO", "PPG"), ("MWOUT", "PPG"), ("MW_OUT", "PPG"),
        ("MUDOUT", "PPG"), ("MUD_WT_OUT", "PPG"),
        ("MWO", "KG/M3"), ("MWOUT", "SG"),
    ],
    "mud_temperature_in": [
        ("MTI", "DEGF"), ("MTIN", "DEGF"), ("TEMP_IN", "DEGF"),
        ("MUD_TEMP_IN", "DEGF"), ("TEMPIN", "DEGF"),
        ("MTI", "DEGC"), ("MTIN", "DEGC"),
    ],
    "mud_temperature_out": [
        ("MTO", "DEGF"), ("MTOUT", "DEGF"), ("TEMP_OUT", "DEGF"),
        ("MUD_TEMP_OUT", "DEGF"), ("TEMPOUT", "DEGF"),
        ("MTO", "DEGC"), ("MTOUT", "DEGC"),
    ],
    "total_gas": [
        ("GAS", "PERCENT"), ("TGAS", "PERCENT"), ("TOTAL_GAS", "PERCENT"),
        ("CGAS", "UNITS"), ("GAS_TOTAL", "UNITS"), ("CHROMO_GAS", "PPM"),
        ("GASA", "PERCENT"), ("TG", "PERCENT"),
    ],
    "rig_activity": [
        ("ACTC", "UNITLESS"), ("RIG_STATE", "UNITLESS"),
        ("DRILL_STATE", "UNITLESS"), ("RSTATUS", "UNITLESS"),
        ("ACT_CODE", "UNITLESS"), ("RIGSTATE", "UNITLESS"),
        ("SUPER_STATE", "UNITLESS"), ("SUB_STATE", "UNITLESS"),
    ],
    "gamma_ray": [
        ("GR", "GAPI"), ("CGR", "GAPI"), ("ECGR", "GAPI"), ("HSGR", "GAPI"),
        ("SGR", "GAPI"), ("GAM", "GAPI"), ("GAMMA", "GAPI"), ("GRN", "GAPI"),
        ("GRD", "GAPI"), ("GR_EDTC", "GAPI"), ("GR_ARC", "GAPI"),
        ("MRGR", "GAPI"), ("GR_RT", "GAPI"), ("GRGC", "GAPI"),
        ("GAMMARAY", "GAPI"), ("GR_CDR", "GAPI"),
    ],
    "bulk_density": [
        ("RHOB", "G/CC"), ("RHOZ", "G/CC"), ("DEN", "G/CC"), ("DENS", "G/CC"),
        ("ZDEN", "G/CC"), ("FDC", "G/CC"), ("CDL", "G/CC"), ("LDL", "G/CC"),
        ("RHOM", "G/CC"), ("BULK_DEN", "G/CC"), ("DENB", "G/CC"),
        ("RHOB", "KG/M3"), ("RHOZ", "KG/M3"),
    ],
    "neutron_porosity": [
        ("NPHI", "V/V"), ("TNPH", "V/V"), ("NPOR", "V/V"), ("NEU", "V/V"),
        ("NEUT", "V/V"), ("CNL", "V/V"), ("SNP", "V/V"), ("PHIN", "V/V"),
        ("APNP", "V/V"), ("BPHI", "V/V"), ("NPHI", "PU"), ("TNPH", "PU"),
        ("NPOR", "PU"), ("NEUTPHI", "PU"),
    ],
    "deep_resistivity": [
        ("RT", "OHMM"), ("ILD", "OHMM"), ("IDPH", "OHMM"), ("AT90", "OHMM"),
        ("LLD", "OHMM"), ("RDEP", "OHMM"), ("RD", "OHMM"), ("RILM", "OHMM"),
        ("RILD", "OHMM"), ("HDRS", "OHMM"), ("RLA5", "OHMM"),
        ("DEEP_RES", "OHMM"), ("RES_DEEP", "OHMM"), ("AHT90", "OHMM"),
    ],
    "shallow_resistivity": [
        ("RXO", "OHMM"), ("MSFL", "OHMM"), ("SFL", "OHMM"), ("RFOC", "OHMM"),
        ("RSHL", "OHMM"), ("RS", "OHMM"), ("RMLL", "OHMM"), ("SN", "OHMM"),
        ("RLLS", "OHMM"), ("AT10", "OHMM"), ("AHT10", "OHMM"),
        ("SHAL_RES", "OHMM"), ("RES_SHAL", "OHMM"),
    ],
    "sonic_compressional": [
        ("DT", "US/FT"), ("DTCO", "US/FT"), ("AC", "US/FT"), ("DT4P", "US/FT"),
        ("DTLN", "US/FT"), ("DTC", "US/FT"), ("SONIC", "US/FT"),
        ("BHC", "US/FT"), ("DT_COMP", "US/FT"), ("DTCR", "US/FT"),
        ("DT", "US/M"), ("DTCO", "US/M"),
    ],
    "sonic_shear": [
        ("DTS", "US/FT"), ("DTSM", "US/FT"), ("DT_SHEAR", "US/FT"),
        ("DTSH", "US/FT"), ("DTS1", "US/FT"), ("DTS2", "US/FT"),
        ("DTSF", "US/FT"), ("DTS", "US/M"), ("DTSM", "US/M"),
    ],
    "caliper": [
        ("CALI", "IN"), ("CAL", "IN"), ("HCAL", "IN"), ("CALS", "IN"),
        ("C1", "IN"), ("C2", "IN"), ("BS", "IN"), ("BDIA", "IN"),
        ("CALIPER", "IN"), ("DCAL", "IN"), ("LCAL", "IN"),
        ("CALI", "MM"), ("CAL", "MM"), ("HCAL", "MM"),
    ],
    "spontaneous_potential": [
        ("SP", "MV"), ("SSP", "MV"), ("PSP", "MV"), ("SPONT_POT", "MV"),
        ("SPOT", "MV"), ("SP_CORR", "MV"),
    ],
    "photoelectric_factor": [
        ("PEF", "B/E"), ("PE", "B/E"), ("PEFZ", "B/E"), ("PEFLA", "B/E"),
        ("PHOTO", "B/E"), ("PEF8", "B/E"), ("PEFL", "B/E"),
    ],
    "inclination": [
        ("INCL", "DEG"), ("INC", "DEG"), ("DEVI", "DEG"), ("INCLIN", "DEG"),
        ("HAZI", "DEG"), ("INCL_MWD", "DEG"), ("DINC", "DEG"),
    ],
    "azimuth": [
        ("AZIM", "DEG"), ("AZI", "DEG"), ("HAZ", "DEG"), ("AZIMUTH", "DEG"),
        ("DAZI", "DEG"), ("AZIM_MWD", "DEG"), ("MAG_AZ", "DEG"),
        ("AZ_CORR", "DEG"),
    ],
    "dogleg_severity": [
        ("DLS", "DEG/100FT"), ("DLSEV", "DEG/30M"), ("DOGLEG", "DEG/100FT"),
        ("DLS", "DEG/30M"), ("DOG_LEG", "DEG/100FT"),
    ],
}


# ---------------------------------------------------------------------------
# Vendor profiles — simulate non-IID naming conventions.
# Each vendor "prefers" a subset of mnemonics for each property.
# ---------------------------------------------------------------------------

def _pick(entries: list[tuple[str, str]], indices: list[int]) -> list[tuple[str, str]]:
    """Pick entries at the given indices, wrapping if out of range."""
    return [entries[i % len(entries)] for i in indices]


VENDOR_PROFILES: list[dict[str, list[tuple[str, str]]]] = []

# Vendor 0: SLB-style (standard PWLS preferred mnemonics)
_slb: dict[str, list[tuple[str, str]]] = {}
for prop, entries in MNEMONIC_DB.items():
    _slb[prop] = entries[:3]
VENDOR_PROFILES.append(_slb)

# Vendor 1: Halliburton-style (offset by 2-4 entries)
_hal: dict[str, list[tuple[str, str]]] = {}
for prop, entries in MNEMONIC_DB.items():
    _hal[prop] = _pick(entries, [2, 3, 4])
VENDOR_PROFILES.append(_hal)

# Vendor 2: Baker Hughes-style (offset by 4-6)
_bhi: dict[str, list[tuple[str, str]]] = {}
for prop, entries in MNEMONIC_DB.items():
    _bhi[prop] = _pick(entries, [4, 5, 6])
VENDOR_PROFILES.append(_bhi)

# Vendor 3: Legacy operator (uses descriptive/long-form names, offset by 6-8)
_leg: dict[str, list[tuple[str, str]]] = {}
for prop, entries in MNEMONIC_DB.items():
    _leg[prop] = _pick(entries, [6, 7, 0])
VENDOR_PROFILES.append(_leg)

# Vendor 4: EDR / Pason-style (uses mixed conventions, offset by 1, 5, 8)
_edr: dict[str, list[tuple[str, str]]] = {}
for prop, entries in MNEMONIC_DB.items():
    _edr[prop] = _pick(entries, [1, 5, 8])
VENDOR_PROFILES.append(_edr)


# ---------------------------------------------------------------------------
# Augmentation — simulate the real-world mnemonic chaos.
# ---------------------------------------------------------------------------

def _augment_mnemonic(mnem: str, rng: np.random.Generator) -> str:
    """Apply random perturbation to a mnemonic string."""
    r = rng.random()
    if r < 0.15:
        return mnem.lower()
    if r < 0.25:
        return mnem.lower().capitalize()
    if r < 0.35:
        return mnem + str(rng.integers(1, 4))
    if r < 0.45:
        return mnem + "_" + str(rng.integers(1, 4))
    if r < 0.50 and len(mnem) > 2:
        prefix = rng.choice(["E", "H", "C", "M", "S", "D"])
        return prefix + mnem
    return mnem


def generate_vendor_data(
    vendor_id: int,
    num_vendors: int = 5,
    samples_per_class: int = 40,
    seed: int = 42,
) -> list[tuple[str, str, int]]:
    """Generate (mnemonic, unit, label) training data for one vendor.

    Each vendor sees its preferred subset of mnemonics with augmentations,
    creating the non-IID distribution that makes federated learning valuable.
    """
    rng = np.random.default_rng(seed + vendor_id * num_vendors * 100)
    profile = VENDOR_PROFILES[vendor_id % len(VENDOR_PROFILES)]
    data: list[tuple[str, str, int]] = []

    for prop_name, label in PROPERTY_TO_IDX.items():
        entries = profile.get(prop_name, MNEMONIC_DB.get(prop_name, []))
        if not entries:
            continue
        for _ in range(samples_per_class):
            mnem, unit = entries[rng.integers(0, len(entries))]
            mnem = _augment_mnemonic(mnem, rng)
            data.append((mnem, unit, label))

    rng.shuffle(data)
    return data
