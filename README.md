# Actigraphy-Based Sleep Staging Without Cardiac Data

Automated classification of sleep macro-architecture (Wake / NREM / REM) from wrist-worn accelerometer signals alone, evaluated under strict subject-isolated cross-validation.

## Research Question

Can computational motion analysis resolve the **Stillness Paradox** — the near-identical accelerometer profile produced by REM paralysis and deep NREM rest — without recourse to cardiac or neurophysiological signals?

This repository implements a complete pipeline for investigating that question: raw tri-axial acceleration is transformed into engineered epoch-level features, classified by ensemble tree models, and evaluated via out-of-fold predictions that never leak information across subjects.

## Dataset

[Motion and Heart Rate from a Wrist-Worn Wearable and Labeled Sleep from Polysomnography](https://physionet.org/content/sleep-accel/1.0.0/) (PhysioNet, 2019).

- **31 subjects**, each with one night of concurrent PSG and wrist actigraphy
- Tri-axial acceleration at ~32 Hz; PSG-derived 30-second epoch labels
- Sleep stages: Wake (0), N1 (1), N2 (2), N3 (3), REM (5)
- This project uses **acceleration data only** — all heart rate files are excluded

## Results

### 3-Stage Classification (Wake / NREM / REM)

| Configuration | Features | Macro F1 | Cohen's Kappa | Per-Subject F1 |
|:---|---:|---:|---:|:---|
| Full RF (all features) | 23 | **0.524** | **0.231** | 0.52 ± 0.11 |
| Full XGB (all features) | 23 | 0.524 | 0.218 | 0.50 ± 0.10 |
| Circadian + Transition RF | 16 | 0.513 | 0.222 | 0.51 ± 0.10 |
| Baseline RF (v2 reproduction) | 7 | 0.470 | 0.180 | 0.45 ± 0.11 |

All metrics are computed on **out-of-fold predictions** via `GroupKFold` (5 splits, subject-isolated). No reported number reflects training-set performance.

### Key Findings

- **Circadian features dominate.** Time-since-onset and night-fraction are the two most important predictors, confirming that REM concentration in the latter half of the night provides strong discriminative signal even from motion data alone.
- **Frequency-domain features (FFT) do not help in isolation.** Per-epoch spectral power at 0.5–4 Hz adds marginal value only when combined with temporal context features.
- **Per-subject z-score normalization reduces variance but hurts macro F1**, suggesting that absolute movement magnitude carries inter-subject information the model exploits.

## Repository Structure

```
├── v1/                         # Original notebooks (archived, flawed evaluation)
├── v2/                         # Audit-fixed notebooks (proper OOF evaluation)
├── v3/                         # Local Python pipeline (active development)
│   ├── config.py               # Constants, paths, seeds, feature lists
│   ├── data_loader.py          # Zip extraction, subject discovery, DataFrame loading
│   ├── preprocessing.py        # Label mapping, merge_asof, epoch aggregation
│   ├── features/
│   │   ├── baseline.py         # VM statistics, rolling windows, movement thresholds
│   │   ├── frequency.py        # FFT power bands, spectral entropy per epoch
│   │   ├── circadian.py        # Time-of-night, sin/cos sleep cycle encoding
│   │   ├── transition.py       # Delta VM, lag features (t-1, t-2, t-3)
│   │   ├── angular.py          # Wrist angle from arctan2(y, z)
│   │   └── normalization.py    # Per-subject z-score normalization
│   ├── models.py               # Pipeline builders (RF, XGBoost + SMOTE)
│   ├── evaluation.py           # OOF evaluation, threshold optimization, reporting
│   ├── smoothing.py            # Per-subject rolling mode post-processing
│   ├── experiment.py           # CLI entry point
│   ├── compare.py              # Cross-experiment comparison table
│   └── results/                # Experiment output JSONs
├── audit_report.qmd            # Comprehensive code audit (Quarto source)
├── audit_report.pdf            # Rendered audit report
└── CLAUDE.md                   # Project conventions and current state
```

## Setup

### Prerequisites

- Python 3.10+
- The PhysioNet dataset (`data.zip`) placed in the repository root

### Installation

```bash
# Clone the repository
git clone https://github.com/Baglecake/local_PGHD.git
cd local_PGHD

# Install dependencies
pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost
```

### Opening in VS Code

1. Open VS Code
2. **File → Open Folder** → select the cloned `local_PGHD` directory
3. Open a terminal in VS Code: **Terminal → New Terminal** (or `` Ctrl+` ``)
4. All commands below run from this terminal

### Data Setup

Place the PhysioNet dataset as `data.zip` in the repository root. The pipeline extracts it automatically on first run.

```
local_PGHD/
├── data.zip          ← place here
├── v3/
│   └── ...
```

## Running Experiments

All experiments are executed from the `v3/` directory.

```bash
cd v3

# Reproduce the baseline (7 features, ~3 min)
python experiment.py --name baseline_rf --model rf --stages 3 --features baseline

# Add circadian and transition features (16 features)
python experiment.py --name circadian_rf --model rf --stages 3 --features baseline,circadian,transition

# Full feature set (23 features)
python experiment.py --name full_rf --model rf --stages 3 --features baseline,frequency,circadian,transition,angular

# XGBoost variant
python experiment.py --name full_xgb --model xgb --stages 3 --features baseline,frequency,circadian,transition,angular

# With per-subject z-score normalization
python experiment.py --name full_rf_zscore --model rf --stages 3 --features baseline,circadian,transition --normalize

# Compare all experiments
python compare.py
```

### CLI Reference

| Flag | Options | Default | Description |
|:---|:---|:---|:---|
| `--name` | any string | *required* | Experiment identifier (used for results filename) |
| `--model` | `rf`, `xgb` | `rf` | Classifier type |
| `--stages` | `3`, `5` | `3` | Number of sleep stages |
| `--features` | comma-separated | `baseline` | Feature modules to include |
| `--normalize` | flag | off | Apply per-subject z-score normalization |
| `--notes` | any string | `""` | Free-text annotation saved with results |

### Available Feature Modules

| Module | Features | Description |
|:---|---:|:---|
| `baseline` | 7 | VM mean/std, rolling windows (2 min, 5 min), time since movement |
| `frequency` | 4 | FFT power bands (0.5–2 Hz, 2–4 Hz), dominant frequency, spectral entropy |
| `circadian` | 4 | Time since onset, night fraction, sin/cos of 90-min cycle |
| `transition` | 5 | Delta VM, delta std, lag features at t-1, t-2, t-3 |
| `angular` | 3 | Mean wrist angle, angle variability, angle change |

## Methodological Conventions

These invariants hold across all code in the repository:

- **Out-of-fold evaluation only.** All reported metrics derive from `cross_val_predict` with `GroupKFold`. The final `pipeline.fit()` is used exclusively for feature importance extraction.
- **Subject isolation.** `GroupKFold` ensures no subject appears in both training and validation within any fold.
- **Deterministic ordering.** `sorted(subject_ids)` and `random_state=42` throughout.
- **Per-subject smoothing.** Post-prediction rolling mode operates within `groupby('subject_id')`, preventing cross-subject boundary contamination.
- **Categorical smoothing.** Rolling mode (not median) for sleep stage predictions — arithmetic operations on categorical encodings are order-dependent and invalid.
- **Temporal alignment guard.** `merge_asof(tolerance=30)` prevents stale label assignment.
- **Cohen's Kappa** reported alongside F1 as the standard inter-rater agreement metric in sleep staging literature.

## License

Dataset: [PhysioNet Open Data License](https://physionet.org/content/sleep-accel/1.0.0/)
