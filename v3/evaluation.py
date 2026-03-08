"""OOF evaluation, threshold optimization, and reporting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from config import CV_SPLITS, RANDOM_STATE, SMOOTHING_WINDOW
from smoothing import smooth_rolling_mode


def apply_thresholds_3stage(probs, classes, thresh_rem):
    """Vectorized threshold application for 3-stage classification."""
    rem_idx = classes.index('REM')
    rem_mask = probs[:, rem_idx] >= thresh_rem
    default_probs = probs.copy()
    default_probs[:, rem_idx] = -1
    default_preds_idx = np.argmax(default_probs, axis=1)
    return np.where(rem_mask, 'REM', np.array(classes)[default_preds_idx])


def apply_thresholds_5stage(probs, classes, thresh_rem, thresh_n3):
    """Vectorized threshold application for 5-stage classification."""
    rem_idx = classes.index('REM')
    n3_idx = classes.index('N3')
    rem_mask = probs[:, rem_idx] >= thresh_rem
    n3_mask = (~rem_mask) & (probs[:, n3_idx] >= thresh_n3)
    default_probs = probs.copy()
    default_probs[:, rem_idx] = -1
    default_probs[:, n3_idx] = -1
    default_preds_idx = np.argmax(default_probs, axis=1)
    y_pred = np.array(classes)[default_preds_idx]
    y_pred[rem_mask] = 'REM'
    y_pred[n3_mask] = 'N3'
    return y_pred


def run_cv(pipeline, X, y, groups, requires_encoding=False):
    """Run GroupKFold CV and return OOF probabilities + CV scores.

    Returns:
        oof_probs: ndarray of OOF probability predictions
        cv_scores: dict with mean/std for accuracy, f1_macro, f1_weighted
        classes: list of class labels in model order
        y_for_eval: the y values to use for evaluation (original strings)
        encoder: LabelEncoder if used, else None
    """
    cv = GroupKFold(n_splits=CV_SPLITS)
    encoder = None

    if requires_encoding:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        y_cv = y_encoded
    else:
        y_cv = y

    # CV metrics
    cv_results = cross_validate(
        pipeline, X, y_cv, groups=groups, cv=cv,
        scoring=['accuracy', 'f1_macro', 'f1_weighted'],
    )
    scores = {
        'accuracy': (cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std()),
        'f1_macro': (cv_results['test_f1_macro'].mean(), cv_results['test_f1_macro'].std()),
        'f1_weighted': (cv_results['test_f1_weighted'].mean(), cv_results['test_f1_weighted'].std()),
    }

    print("Cross-Validation Results (GroupKFold, 5 splits):")
    for metric, (mean, std) in scores.items():
        print(f"  {metric:15s}: {mean:.4f} +/- {std:.4f}")

    # OOF probabilities
    oof_probs = cross_val_predict(pipeline, X, y_cv, groups=groups, cv=cv, method='predict_proba')

    # Final fit for feature importances only
    pipeline.fit(X, y_cv)
    clf = pipeline.named_steps['clf']

    if requires_encoding:
        classes = list(encoder.inverse_transform(clf.classes_))
    else:
        classes = list(clf.classes_)

    print(f"\nClasses (model order): {classes}")
    return oof_probs, scores, classes, clf, encoder


def optimize_thresholds(oof_probs, y_true, classes, subject_ids, stage_to_int, int_to_stage, n_stages=3):
    """Search for optimal thresholds on OOF predictions.

    Returns:
        best_thresholds: dict with threshold values
        best_f1: best macro F1 achieved
    """
    print("\nThreshold optimization on OOF predictions...")

    if n_stages == 3:
        best_f1 = 0.0
        best_thresh = 0.50
        for thresh in np.arange(0.30, 0.71, 0.01):
            y_pred = apply_thresholds_3stage(oof_probs, classes, thresh)
            y_pred_smooth = smooth_rolling_mode(y_pred, subject_ids, stage_to_int, int_to_stage)
            f1 = f1_score(y_true, y_pred_smooth, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        print(f"Best REM threshold: {best_thresh:.2f} (OOF F1 Macro: {best_f1:.4f})")
        return {'rem': best_thresh}, best_f1

    else:  # 5-stage
        best_f1 = 0.0
        best_rem, best_n3 = 0.50, 0.70
        for t_rem in np.arange(0.30, 0.71, 0.02):
            for t_n3 in np.arange(0.50, 0.81, 0.05):
                y_pred = apply_thresholds_5stage(oof_probs, classes, t_rem, t_n3)
                y_pred_smooth = smooth_rolling_mode(y_pred, subject_ids, stage_to_int, int_to_stage)
                f1 = f1_score(y_true, y_pred_smooth, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_rem, best_n3 = t_rem, t_n3
        print(f"Best REM threshold: {best_rem:.2f}, N3 threshold: {best_n3:.2f} (OOF F1 Macro: {best_f1:.4f})")
        return {'rem': best_rem, 'n3': best_n3}, best_f1


def generate_final_predictions(oof_probs, classes, thresholds, subject_ids, stage_to_int, int_to_stage, n_stages=3):
    """Apply best thresholds and smoothing to OOF predictions."""
    if n_stages == 3:
        y_pred = apply_thresholds_3stage(oof_probs, classes, thresholds['rem'])
    else:
        y_pred = apply_thresholds_5stage(oof_probs, classes, thresholds['rem'], thresholds['n3'])
    return smooth_rolling_mode(y_pred, subject_ids, stage_to_int, int_to_stage)


def full_report(y_true, y_pred, subject_ids, classes):
    """Generate comprehensive evaluation report.

    Returns:
        dict with all metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    print(f"\nClassification Report (OOF):")
    print(classification_report(y_true, y_pred))
    print(f"Cohen's Kappa: {kappa:.4f}")

    # Per-subject F1
    subject_f1s = []
    for sid in sorted(set(subject_ids)):
        mask = np.array(subject_ids) == sid
        if mask.sum() == 0:
            continue
        y_t = np.array(y_true)[mask] if not isinstance(y_true, np.ndarray) else y_true[mask]
        y_p = y_pred[mask]
        f1 = f1_score(y_t, y_p, average='macro', zero_division=0)
        subject_f1s.append({'subject_id': sid, 'f1_macro': f1})

    subj_df = pd.DataFrame(subject_f1s)
    print(f"\nPer-Subject F1 Macro: {subj_df['f1_macro'].mean():.4f} +/- {subj_df['f1_macro'].std():.4f}")
    print(f"  Min: {subj_df['f1_macro'].min():.4f}  Max: {subj_df['f1_macro'].max():.4f}")

    return {
        'macro_f1': report['macro avg']['f1-score'],
        'accuracy': report['accuracy'],
        'kappa': kappa,
        'confusion_matrix': cm.tolist(),
        'per_class': {cls: report.get(cls, {}) for cls in classes},
        'per_subject_f1_mean': subj_df['f1_macro'].mean(),
        'per_subject_f1_std': subj_df['f1_macro'].std(),
        'per_subject_f1_min': subj_df['f1_macro'].min(),
        'per_subject_f1_max': subj_df['f1_macro'].max(),
        'per_subject_details': subject_f1s,
    }
