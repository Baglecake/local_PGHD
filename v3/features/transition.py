"""Transition and lag features for temporal dynamics."""

import pandas as pd


def add_transition_features(epoch_df):
    """Add rate-of-change and lag features.

    Features:
        delta_vm: change in mean_vm from previous epoch (per subject)
        delta_std_vm: change in std_vm from previous epoch
        lag_mean_vm_1/2/3: mean_vm at t-1, t-2, t-3
    """
    df = epoch_df.copy()

    # Delta features (per subject)
    df['delta_vm'] = df.groupby('subject_id')['mean_vm'].diff().fillna(0)
    df['delta_std_vm'] = df.groupby('subject_id')['std_vm'].diff().fillna(0)

    # Lag features (per subject)
    for lag in [1, 2, 3]:
        df[f'lag_mean_vm_{lag}'] = df.groupby('subject_id')['mean_vm'].shift(lag).fillna(
            df['mean_vm']
        )

    print(f"Transition features added. Shape: {df.shape}")
    return df


TRANSITION_FEATURE_COLS = ['delta_vm', 'delta_std_vm', 'lag_mean_vm_1', 'lag_mean_vm_2', 'lag_mean_vm_3']
