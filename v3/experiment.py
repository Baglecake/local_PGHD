"""CLI entry point for running sleep staging experiments."""

import argparse
import json
import os
import sys
import time
import numpy as np

# Add v3 to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LABEL_MAP_3STAGE, LABEL_MAP_5STAGE,
    BASELINE_FEATURES_3, BASELINE_FEATURES_5,
    STAGE_TO_INT_3, INT_TO_STAGE_3,
    STAGE_TO_INT_5, INT_TO_STAGE_5,
    SMOOTHING_WINDOW, RESULTS_DIR,
)
from data_loader import load_data
from preprocessing import build_epoch_df
from features.baseline import add_baseline_features
from features.normalization import add_zscore_features
from features.frequency import add_frequency_features, FFT_FEATURE_COLS
from features.circadian import add_circadian_features, CIRCADIAN_FEATURE_COLS
from features.transition import add_transition_features, TRANSITION_FEATURE_COLS
from features.angular import add_angular_features, ANGULAR_FEATURE_COLS
from models import build_pipeline
from evaluation import run_cv, optimize_thresholds, generate_final_predictions, full_report


def parse_args():
    parser = argparse.ArgumentParser(description='Sleep staging experiment runner')
    parser.add_argument('--name', required=True, help='Experiment name (used for results filename)')
    parser.add_argument('--model', default='rf', choices=['rf', 'xgb'], help='Model type')
    parser.add_argument('--stages', type=int, default=3, choices=[3, 5], help='Number of sleep stages')
    parser.add_argument('--features', default='baseline', help='Comma-separated feature sets: baseline,frequency,circadian,transition,angular')
    parser.add_argument('--normalize', action='store_true', help='Apply per-subject z-score normalization')
    parser.add_argument('--notes', default='', help='Optional notes about this experiment')
    return parser.parse_args()


def run_experiment(args):
    start_time = time.time()

    # Setup
    label_map = LABEL_MAP_3STAGE if args.stages == 3 else LABEL_MAP_5STAGE
    stage_to_int = STAGE_TO_INT_3 if args.stages == 3 else STAGE_TO_INT_5
    int_to_stage = INT_TO_STAGE_3 if args.stages == 3 else INT_TO_STAGE_5
    include_extended = (args.stages == 5)

    print(f"{'='*60}")
    print(f"Experiment: {args.name}")
    print(f"Model: {args.model.upper()}, Stages: {args.stages}, Normalize: {args.normalize}")
    print(f"Features: {args.features}")
    print(f"{'='*60}\n")

    # 1. Load data
    print("--- Loading data ---")
    motion_df, labels_df, subject_ids = load_data()

    # 2. Build epoch DataFrame
    print("\n--- Preprocessing ---")
    epoch_df, merged_df = build_epoch_df(motion_df, labels_df, label_map)

    # 3. Feature engineering
    print("\n--- Feature engineering ---")
    feature_sets = [f.strip() for f in args.features.split(',')]

    if 'baseline' in feature_sets:
        epoch_df = add_baseline_features(epoch_df, include_extended=include_extended)

    # Determine feature columns
    if args.stages == 3:
        feature_cols = list(BASELINE_FEATURES_3)
    else:
        feature_cols = list(BASELINE_FEATURES_5)

    if 'frequency' in feature_sets:
        print("Adding frequency-domain (FFT) features...")
        epoch_df = add_frequency_features(epoch_df, merged_df)
        feature_cols.extend(FFT_FEATURE_COLS)

    if 'circadian' in feature_sets:
        print("Adding circadian features...")
        epoch_df = add_circadian_features(epoch_df)
        feature_cols.extend(CIRCADIAN_FEATURE_COLS)

    if 'transition' in feature_sets:
        print("Adding transition/lag features...")
        epoch_df = add_transition_features(epoch_df)
        feature_cols.extend(TRANSITION_FEATURE_COLS)

    if 'angular' in feature_sets:
        print("Adding angular features...")
        epoch_df = add_angular_features(epoch_df, merged_df)
        feature_cols.extend(ANGULAR_FEATURE_COLS)

    # 4. Normalization (optional)
    if args.normalize:
        print("Applying per-subject z-score normalization...")
        epoch_df = add_zscore_features(epoch_df, feature_cols)

    # 5. Prepare X, y, groups
    X = epoch_df[feature_cols]
    y = epoch_df['sleep_stage']
    groups = epoch_df['subject_id']
    subject_ids_array = epoch_df['subject_id'].values

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")

    # 6. Build pipeline and run CV
    print("\n--- Model training & OOF evaluation ---")
    pipeline, requires_encoding = build_pipeline(args.model, args.stages)
    oof_probs, cv_scores, classes, clf, encoder = run_cv(pipeline, X, y, groups, requires_encoding)

    # 7. Threshold optimization
    print("\n--- Threshold optimization ---")
    best_thresholds, best_f1 = optimize_thresholds(
        oof_probs, y.values, classes, subject_ids_array,
        stage_to_int, int_to_stage, n_stages=args.stages
    )

    # 8. Final evaluation
    print("\n--- Final evaluation (OOF) ---")
    y_pred_final = generate_final_predictions(
        oof_probs, classes, best_thresholds, subject_ids_array,
        stage_to_int, int_to_stage, n_stages=args.stages
    )
    report = full_report(y.values, y_pred_final, subject_ids_array, classes)

    # 9. Feature importances
    importances = clf.feature_importances_
    fi = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    print("\nFeature Importances:")
    for feat, imp in fi:
        print(f"  {feat:30s}: {imp:.4f}")

    # 10. Save results
    elapsed = time.time() - start_time
    result = {
        'name': args.name,
        'model': args.model,
        'stages': args.stages,
        'features': feature_cols,
        'normalize': args.normalize,
        'feature_sets': feature_sets,
        'notes': args.notes,
        'cv_scores': {k: {'mean': v[0], 'std': v[1]} for k, v in cv_scores.items()},
        'best_thresholds': best_thresholds,
        'best_threshold_f1': best_f1,
        'report': report,
        'feature_importances': {f: float(i) for f, i in fi},
        'elapsed_seconds': elapsed,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{args.name}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.name}")
    print(f"  Macro F1: {report['macro_f1']:.4f}")
    print(f"  Kappa:    {report['kappa']:.4f}")
    print(f"  Accuracy: {report['accuracy']:.2%}")
    print(f"  Per-subject F1: {report['per_subject_f1_mean']:.4f} +/- {report['per_subject_f1_std']:.4f}")
    print(f"  Thresholds: {best_thresholds}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Saved to: {result_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
