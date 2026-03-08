"""Preprocessing: label mapping, temporal alignment, epoch aggregation."""

import numpy as np
import pandas as pd
from config import TOLERANCE, EPOCH_SECONDS, MOVEMENT_BUFFER


def apply_label_map(labels_df, label_map):
    """Filter valid labels and map to sleep stage names."""
    df = labels_df[labels_df['label'].isin(label_map.keys())].copy()
    df['sleep_stage'] = df['label'].map(label_map)
    return df


def synchronize(motion_df, labels_df):
    """Temporal alignment via merge_asof with tolerance guard."""
    labels_df = labels_df.copy()
    labels_df['timestamp'] = labels_df['timestamp'].astype(float)
    motion_df = motion_df.copy()
    motion_df['subject_id'] = motion_df['subject_id'].astype(str)
    labels_df['subject_id'] = labels_df['subject_id'].astype(str)

    motion_df = motion_df.sort_values('timestamp')
    labels_df = labels_df.sort_values('timestamp')

    merged = pd.merge_asof(
        motion_df, labels_df,
        on='timestamp', by='subject_id',
        direction='backward', tolerance=TOLERANCE,
    )
    merged = merged.dropna(subset=['sleep_stage'])
    return merged


def build_epoch_df(motion_df, labels_df, label_map):
    """Full pipeline: label map -> sync -> epoch aggregation.

    Returns:
        epoch_df: DataFrame with aggregated features per epoch
        merged_df: Raw-resolution DataFrame (needed for FFT/angular features)
    """
    labels_mapped = apply_label_map(labels_df, label_map)
    merged_df = synchronize(motion_df, labels_mapped)

    # VM and epoch boundaries
    merged_df['epoch_start'] = (merged_df['timestamp'] // EPOCH_SECONDS) * EPOCH_SECONDS
    merged_df['vm'] = np.sqrt(merged_df['x']**2 + merged_df['y']**2 + merged_df['z']**2)

    # Aggregate to epochs
    epoch_df = merged_df.groupby(['subject_id', 'epoch_start']).agg(
        mean_vm=('vm', 'mean'),
        std_vm=('vm', 'std'),
        sleep_stage=('sleep_stage', 'first'),
    ).reset_index()

    epoch_df = epoch_df.sort_values(['subject_id', 'epoch_start']).reset_index(drop=True)

    print(f"Built {len(epoch_df):,} epochs across {epoch_df['subject_id'].nunique()} subjects.")
    print(f"Class distribution:\n{epoch_df['sleep_stage'].value_counts().to_string()}")

    return epoch_df, merged_df
