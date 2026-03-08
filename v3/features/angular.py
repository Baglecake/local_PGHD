"""Angular features from raw accelerometer axes."""

import numpy as np
import pandas as pd


def add_angular_features(epoch_df, merged_df):
    """Compute wrist angle features from raw x/y/z per epoch.

    Features:
        mean_angle: mean wrist angle (arctan2(y, z)) per epoch
        std_angle: variability of wrist angle per epoch
        angle_change: change in mean_angle from previous epoch
    """
    merged_df = merged_df.copy()
    merged_df['angle'] = np.arctan2(merged_df['y'], merged_df['z'])

    angle_df = merged_df.groupby(['subject_id', 'epoch_start'])['angle'].agg(
        mean_angle='mean',
        std_angle='std',
    ).reset_index()

    angle_df['std_angle'] = angle_df['std_angle'].fillna(0)

    # Angle change (per subject)
    angle_df = angle_df.sort_values(['subject_id', 'epoch_start'])
    angle_df['angle_change'] = angle_df.groupby('subject_id')['mean_angle'].diff().fillna(0)

    epoch_df = epoch_df.merge(angle_df, on=['subject_id', 'epoch_start'], how='left')

    for col in ['mean_angle', 'std_angle', 'angle_change']:
        epoch_df[col] = epoch_df[col].fillna(0)

    print(f"Angular features added. Shape: {epoch_df.shape}")
    return epoch_df


ANGULAR_FEATURE_COLS = ['mean_angle', 'std_angle', 'angle_change']
