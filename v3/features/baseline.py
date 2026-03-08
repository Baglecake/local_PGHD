"""Baseline features matching v2 notebooks: rolling windows + time since movement."""

import numpy as np
import pandas as pd
from config import MOVEMENT_BUFFER


def add_baseline_features(epoch_df, include_extended=False):
    """Add rolling window and movement features.

    Args:
        epoch_df: DataFrame with mean_vm, std_vm, subject_id, epoch_start
        include_extended: If True, add 10m and 20m windows (for 5-stage)

    Returns:
        epoch_df with added feature columns
    """
    df = epoch_df.copy()

    # Rolling windows (per-subject)
    df['roll_mean_vm_2m'] = df.groupby('subject_id')['mean_vm'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean())
    df['roll_std_vm_2m'] = df.groupby('subject_id')['std_vm'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean())

    df['roll_mean_vm_5m'] = df.groupby('subject_id')['mean_vm'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean())
    df['roll_std_vm_5m'] = df.groupby('subject_id')['std_vm'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean())

    if include_extended:
        df['roll_std_vm_10m'] = df.groupby('subject_id')['std_vm'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean())
        df['roll_mean_vm_20m'] = df.groupby('subject_id')['mean_vm'].transform(
            lambda x: x.rolling(window=40, min_periods=1).mean())

    # Dynamic rest-activity thresholding
    def get_movement_mask(group):
        baseline = group['mean_vm'].quantile(0.05)
        return group['mean_vm'] > (baseline + MOVEMENT_BUFFER)

    df['is_movement'] = df.groupby('subject_id').apply(
        get_movement_mask, include_groups=False
    ).reset_index(level=0, drop=True)

    df['last_movement_time'] = df['epoch_start'].where(df['is_movement']).groupby(df['subject_id']).ffill()
    df['time_since_last_movement'] = df['epoch_start'] - df['last_movement_time'].fillna(df['epoch_start'])

    df.drop(columns=['is_movement', 'last_movement_time'], inplace=True)

    # Targeted fillna
    df['std_vm'] = df['std_vm'].fillna(0)
    df['time_since_last_movement'] = df['time_since_last_movement'].ffill().fillna(0)
    for col in [c for c in df.columns if c.startswith('roll_')]:
        df[col] = df[col].fillna(0)

    return df
