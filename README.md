# Fed-WITSML-Map: Federated WITSML Mnemonic Mapping

**By [DrillSense](https://drillsense.com) -- The Anti-Platform Platform for Drilling Intelligence**

Federated learning for the industry's most persistent data management problem: WITSML channel name mapping. The oil and gas industry has ~50,000 known curve mnemonics -- the same physical measurement (e.g., gamma ray) appears as "GR", "CGR", "ECGR", "HSGR", "GAM", "GAMMA", or dozens of other vendor-specific labels. Every operator maintains internal mapping tables they will never share.

This Flower app trains a **character-level CNN classifier** that maps arbitrary WITSML mnemonics to standard [PWLS](https://energistics.org/practical-well-log-standard) property classes. Each federated client represents a different service company or operator with distinct naming conventions. The server aggregates using **FedProx** to handle the heterogeneous vendor data. The result is a universal mnemonic mapper that has effectively learned from dozens of vendors' naming conventions without requiring anyone to manually curate and contribute a central mapping database.

## Why Federated Learning?

Every operator and service company has built internal mnemonic lookup tables -- often spreadsheets maintained by one person. These tables are proprietary because they reveal what tools an operator runs, what measurements they value, and how their workflows are structured.

But every operator suffers from the same problem: when they receive WITSML data from a partner, a new service company, or a legacy system, they cannot automatically determine what each channel measures. Manual mapping costs weeks per well program.

Federated learning solves this without requiring a centralized data collection effort:

- Each client trains on their **own mnemonic mapping knowledge** locally
- Only **model weight updates** are shared -- no need to export, clean, or contribute mapping tables to a shared repo
- **FedProx** prevents any single vendor's conventions from dominating the global model
- The global model generalises to **unseen mnemonics** via learned character-level patterns
- New vendors or legacy systems can be onboarded by adding a new client -- the global model improves automatically

## Capabilities

### 1. Mnemonic Classification
Maps any raw mnemonic string to one of 35 standard PWLS property classes with a confidence score. Works on mnemonics the model has never seen before via learned character-level patterns.

### 2. Unit-of-Measure Inference
Given actual curve statistics (mean, std, min, max), infers the most likely unit by checking which unit's expected value range best fits the observed data. For example, a "depth" channel with values 1000-15000 is likely in feet; 300-5000 is likely in meters.

### 3. Mis-Mapping Detection
Flags channels where the mnemonic label and the actual data disagree. A channel labeled "GR" (gamma ray, expected 0-300 GAPI) that reads 0.1-10000 is probably resistivity data that got mislabeled. Also detects stuck sensors, negative values where they shouldn't exist, and unit declaration mismatches.

```python
from fed_witsml_map.diagnostics import diagnose_channel

result = diagnose_channel(
    predicted_property="gamma_ray",
    confidence=0.94,
    stats={"mean": 65.2, "std": 22.1, "min": 8.0, "max": 198.0},
    declared_unit="GAPI",
)
# result.likely_unit   -> "GAPI"
# result.unit_match    -> True
# result.quality_flags -> []  (no issues)

# Mis-mapped channel example:
result = diagnose_channel(
    predicted_property="gamma_ray",
    confidence=0.31,
    stats={"mean": 450.0, "std": 2100.0, "min": 0.2, "max": 9500.0},
    declared_unit="GAPI",
)
# result.quality_flags -> [
#   "VALUES_OUT_OF_RANGE: ...",
#   "UNIT_MISMATCH: ...",
#   "LOW_CONFIDENCE: ...",
# ]
```

## Property Classes

The model classifies mnemonics into 35 standard PWLS property classes covering drilling, logging, and directional measurements:

| Category | Properties |
|---|---|
| **Index** | measured_depth, true_vertical_depth, time |
| **Drilling** | bit_depth, hole_depth, block_position, weight_on_bit, hookload, rotary_speed, rotary_torque, rate_of_penetration, standpipe_pressure, pump_rate, pump_strokes, mud_flow_in, mud_flow_out, mud_weight_in, mud_weight_out, mud_temperature_in, mud_temperature_out, total_gas, rig_activity |
| **Logging** | gamma_ray, bulk_density, neutron_porosity, deep_resistivity, shallow_resistivity, sonic_compressional, sonic_shear, caliper, spontaneous_potential, photoelectric_factor |
| **Directional** | inclination, azimuth, dogleg_severity |

## Model Architecture

```
Input: mnemonic string (up to 24 chars) + unit string (up to 12 chars)
  -> Character tokenisation (A-Z, 0-9, _, -, /, ., space)
  -> Mnemonic branch: Embedding(42, 32) -> Conv1d(32,128,k=3) -> Conv1d(128,128,k=3) -> AvgPool
  -> Unit branch:     Embedding(42, 32) -> Conv1d(32, 64, k=3) -> AvgPool
  -> Concat (192-d) -> Linear(192, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 35)
Output: 35-class property probabilities + confidence score
```

## Data Sources

The built-in mnemonic catalog is compiled from publicly available standards:

- [Energistics PWLS v3.0](https://energistics.org/practical-well-log-standard) -- ~50,000 curve types
- [SLB Curve Mnemonic Dictionary](https://www.apps.slb.com/cmd/) (OSDD) -- 50,000+ entries
- [WITS Specification Rev 1.1](https://www.petrospec-technologies.com/resource/wits_doc.htm) -- all 25 record types
- [GDR Drilling Data Standard](https://gdr.openei.org/drilling_data_standard) -- Pason/RigCloud mappings
- [lasmnemonicsid](https://github.com/Nobleza-Energy/LASMnemonicsID) -- alias dictionaries (MIT)
- [osdu-pwls](https://github.com/geosoft-as/osdu-pwls) -- PWLS in JSON/Excel for OSDU
- [PDS WITSML](https://github.com/pds-technology/witsml) -- WITSML data object definitions

In simulation mode, 5 vendor profiles generate synthetic training data with different mnemonic preferences and augmentations (case variations, numeric suffixes, vendor prefixes).

## Run the App

### Simulation

This app runs with **5 virtual SuperNodes** by default, each simulating a different vendor/service company (SLB-style, Halliburton-style, Baker Hughes-style, legacy operator, EDR/Pason-style). No external data or downloads required.

```bash
cd fed-witsml-map && pip install -e .
flwr run .
```

Override settings:

```bash
flwr run . --run-config "num-server-rounds=20 proximal-mu=0.05 samples-per-class=120"
```

### Deployment

In Deployment mode each SuperNode loads its own mnemonic mapping table locally. Replace `load_demo_data()` in `task.py` with a loader that reads your internal mapping CSV (columns: `mnemonic`, `unit`, `property_class`).

```bash
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/operator_mapping_table.csv"
```

Then launch the run:

```bash
flwr run . <SUPERLINK-CONNECTION> --stream
```

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `num-server-rounds` | 10 | Federated rounds |
| `batch-size` | 32 | Local batch size |
| `local-epochs` | 5 | Epochs per client per round |
| `learning-rate` | 0.003 | Adam learning rate |
| `test-fraction` | 0.2 | Fraction of data held out for validation |
| `num-partitions` | 5 | Virtual vendors in simulation |
| `samples-per-class` | 80 | Training samples per property class per vendor |
| `model-save-path` | `output/witsml_mapper.pt` | Where the server saves the global model |
| `seed` | 42 | Random seed |
| `proximal-mu` | 0.1 | FedProx proximal term. `0.0` = plain FedAvg |

## Resources

- PWLS v3.0: [energistics.org/practical-well-log-standard](https://energistics.org/practical-well-log-standard)
- PWLS Curve Catalog: [community.opengroup.org/energistics/pwls-curve-catalog](https://community.opengroup.org/energistics/pwls-curve-catalog)
- SLB CMD: [apps.slb.com/cmd/](https://www.apps.slb.com/cmd/)
- PDS WITSML: [github.com/pds-technology/witsml](https://github.com/pds-technology/witsml)
- Flower: [flower.ai](https://flower.ai/)

## License

Apache-2.0.
