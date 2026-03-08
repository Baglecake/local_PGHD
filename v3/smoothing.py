"""Post-prediction smoothing: per-subject rolling mode."""

import numpy as np
import pandas as pd
from scipy.stats import mode


def smooth_rolling_mode(y_pred_strings, subject_ids_array, stage_to_int, int_to_stage, window=5):
    """Per-subject rolling mode smoothing for categorical predictions."""
    def _rolling_mode(series, w=window):
        return series.rolling(window=w, center=True, min_periods=1).apply(
            lambda x: mode(x, keepdims=False).mode
        )

    y_pred_ints = pd.Series(y_pred_strings).map(stage_to_int).values
    temp_df = pd.DataFrame({'subject_id': subject_ids_array, 'pred': y_pred_ints})
    temp_df['smoothed'] = temp_df.groupby('subject_id')['pred'].transform(
        lambda x: _rolling_mode(x)
    ).astype(int)
    return temp_df['smoothed'].map(int_to_stage).values
