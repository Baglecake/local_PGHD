"""Circadian and time-of-night features."""

import numpy as np
import pandas as pd


def add_circadian_features(epoch_df):
    """Add time-of-night and sleep cycle encoding features.

    Features:
        time_since_onset: seconds since first epoch per subject
        night_fraction: normalized position in the night (0-1)
        sin_90min: sine of ~90-minute sleep cycle encoding
        cos_90min: cosine of ~90-minute sleep cycle encoding
    """
    df = epoch_df.copy()

    # Time since sleep onset (per subject)
    df['time_since_onset'] = df.groupby('subject_id')['epoch_start'].transform(
        lambda x: x - x.min()
    )

    # Night fraction (0 = start, 1 = end of recording)
    df['night_fraction'] = df.groupby('subject_id')['epoch_start'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )

    # Sleep cycle encoding (~90 min = 5400 seconds)
    cycle_period = 5400.0
    df['sin_90min'] = np.sin(2 * np.pi * df['time_since_onset'] / cycle_period)
    df['cos_90min'] = np.cos(2 * np.pi * df['time_since_onset'] / cycle_period)

    print(f"Circadian features added. Shape: {df.shape}")
    return df


CIRCADIAN_FEATURE_COLS = ['time_since_onset', 'night_fraction', 'sin_90min', 'cos_90min']
