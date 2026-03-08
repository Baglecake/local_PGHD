"""Frequency-domain features via FFT on raw acceleration per epoch."""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


def add_frequency_features(epoch_df, merged_df):
    """Compute FFT power-band features from raw VM within each 30s epoch.

    Args:
        epoch_df: Aggregated epoch DataFrame (will be augmented)
        merged_df: Raw-resolution DataFrame with vm, epoch_start, subject_id

    Returns:
        epoch_df with added FFT feature columns
    """
    # Compute FFT features per (subject_id, epoch_start) group
    fft_records = []

    groups = merged_df.groupby(['subject_id', 'epoch_start'])['vm']
    for (sid, epoch_start), vm_series in groups:
        vm = vm_series.values
        n = len(vm)
        if n < 4:
            fft_records.append({
                'subject_id': sid, 'epoch_start': epoch_start,
                'fft_power_low': 0.0, 'fft_power_high': 0.0,
                'fft_dominant_freq': 0.0, 'fft_spectral_entropy': 0.0,
            })
            continue

        # Estimate sampling rate from timestamps
        dt = merged_df.loc[vm_series.index, 'timestamp'].diff().median()
        if dt <= 0 or np.isnan(dt):
            dt = 0.01  # fallback ~100Hz
        fs = 1.0 / dt

        # FFT
        vm_centered = vm - vm.mean()
        fft_vals = np.fft.rfft(vm_centered)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0/fs)

        # Power in bands
        low_mask = (freqs >= 0.5) & (freqs < 2.0)
        high_mask = (freqs >= 2.0) & (freqs < 4.0)

        power_low = power[low_mask].sum() if low_mask.any() else 0.0
        power_high = power[high_mask].sum() if high_mask.any() else 0.0

        # Dominant frequency (excluding DC)
        if len(power) > 1:
            dom_idx = np.argmax(power[1:]) + 1
            dominant_freq = freqs[dom_idx]
        else:
            dominant_freq = 0.0

        # Spectral entropy (normalized)
        power_norm = power[1:] / (power[1:].sum() + 1e-12)
        spectral_ent = scipy_entropy(power_norm + 1e-12)

        fft_records.append({
            'subject_id': sid, 'epoch_start': epoch_start,
            'fft_power_low': power_low,
            'fft_power_high': power_high,
            'fft_dominant_freq': dominant_freq,
            'fft_spectral_entropy': spectral_ent,
        })

    fft_df = pd.DataFrame(fft_records)

    # Log-transform power features (they span many orders of magnitude)
    fft_df['fft_power_low'] = np.log1p(fft_df['fft_power_low'])
    fft_df['fft_power_high'] = np.log1p(fft_df['fft_power_high'])

    # Merge back to epoch_df
    epoch_df = epoch_df.merge(fft_df, on=['subject_id', 'epoch_start'], how='left')

    for col in ['fft_power_low', 'fft_power_high', 'fft_dominant_freq', 'fft_spectral_entropy']:
        epoch_df[col] = epoch_df[col].fillna(0)

    print(f"FFT features added. Shape: {epoch_df.shape}")
    return epoch_df


FFT_FEATURE_COLS = ['fft_power_low', 'fft_power_high', 'fft_dominant_freq', 'fft_spectral_entropy']
