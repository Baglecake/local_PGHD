# Ben Colab - Sleep Staging from Wrist Actigraphy

## Project
Patient-calibrated sleep staging using wrist-worn accelerometer data only (no heart rate).
PhysioNet "Motion and Heart Rate" dataset, 31 subjects.

## Structure
- `v1/` — Original notebooks (archived, flawed evaluation on training data)
- `v2/` — Fixed notebooks with proper OOF cross-validation evaluation
- `v3/` — Local Python pipeline for optimization experiments
- `audit_report.qmd` / `.pdf` — Comprehensive code audit

## Notebooks (4 variants)
- {3-Stage, 5-Stage} x {Random Forest, XGBoost}
- 3-Stage: Wake / NREM / REM (7 features)
- 5-Stage: Wake / N1 / N2 / N3 / REM (9 features)

## Conventions
- All evaluation must use out-of-fold predictions (`cross_val_predict`), never training-set predictions
- Post-prediction smoothing must use per-subject rolling mode, not global rolling median
- `random_state=42` everywhere for reproducibility
- `sorted(subject_ids)` for deterministic ordering
- `tolerance=30` on all `merge_asof` calls
- Cohen's Kappa required alongside F1 for sleep staging evaluation
- Notebooks run on Google Colab; data path: `/content/heartratedata.zip`

## Current Performance (3-Stage, validated OOF)
- Best: Full RF (23 features) — Macro F1=0.524, Kappa=0.231, REM thresh=0.52
- Baseline RF (7 features): Macro F1=0.470, Kappa=0.180
- Circadian+transition features gave the biggest lift (+0.04 F1)
- Per-subject F1: 0.52 +/- 0.11 (range 0.31-0.82)
- Primary bottleneck: REM/NREM confusion ("Stillness Paradox")

## v3 Optimization Results (3-Stage)
See `v3/results/` for full JSON outputs. Use `python3 v3/compare.py` for comparison table.
- Circadian features (time_since_onset, night_fraction) are the most impactful new features
- FFT alone does not help; angular features add marginal value in combination
- Z-score normalization reduces per-subject variance but slightly hurts macro F1

## Next Steps
1. Hyperparameter tuning — Optuna/RandomizedSearchCV with GroupKFold
2. HMM post-processing — replace rolling mode with learned transition probabilities
3. Extend to 5-stage with best feature set
4. LightGBM comparison
See memory/optimization.md for full details.
