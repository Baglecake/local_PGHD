"""Load raw accelerometer and PSG label data from the PhysioNet dataset."""

import os
import zipfile
import pandas as pd
from config import DATA_ZIP, EXTRACT_DIR


def load_data(zip_path=DATA_ZIP, extract_dir=EXTRACT_DIR):
    """Extract zip and load all subjects' motion and label DataFrames.

    Returns:
        motion_df: DataFrame with columns [timestamp, x, y, z, subject_id]
        labels_df: DataFrame with columns [timestamp, label, subject_id]
        subject_ids: sorted list of subject ID strings
    """
    # Cache extraction
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        print("Extraction complete.")

    # Find the nested directory (PhysioNet long name)
    base_candidates = [
        d for d in os.listdir(extract_dir)
        if os.path.isdir(os.path.join(extract_dir, d)) and not d.startswith('.')
    ]
    if base_candidates:
        base_path = os.path.join(extract_dir, base_candidates[0])
    else:
        base_path = extract_dir

    # Try both possible structures: data/motion or motion directly
    motion_dir = os.path.join(base_path, 'motion')
    labels_dir = os.path.join(base_path, 'labels')
    if not os.path.exists(motion_dir):
        motion_dir = os.path.join(extract_dir, 'data', 'motion')
        labels_dir = os.path.join(extract_dir, 'data', 'labels')
    if not os.path.exists(motion_dir):
        motion_dir = os.path.join(base_path, 'data', 'motion')
        labels_dir = os.path.join(base_path, 'data', 'labels')

    # Discover subjects
    subject_ids = sorted([
        f.split('_')[0]
        for f in os.listdir(motion_dir)
        if f.endswith('_acceleration.txt')
    ])
    print(f"Discovered {len(subject_ids)} subjects.")

    # Load all data
    motion_list = []
    labels_list = []

    for sid in subject_ids:
        motion_file = os.path.join(motion_dir, f"{sid}_acceleration.txt")
        if os.path.exists(motion_file):
            df_m = pd.read_csv(motion_file, sep=' ', header=None,
                               names=['timestamp', 'x', 'y', 'z'])
            df_m['subject_id'] = sid
            motion_list.append(df_m)

        label_file = os.path.join(labels_dir, f"{sid}_labeled_sleep.txt")
        if os.path.exists(label_file):
            df_l = pd.read_csv(label_file, sep=' ', header=None,
                               names=['timestamp', 'label'])
            df_l['subject_id'] = sid
            labels_list.append(df_l)

    motion_df = pd.concat(motion_list, ignore_index=True)
    labels_df = pd.concat(labels_list, ignore_index=True)

    print(f"Loaded {len(motion_df):,} motion samples, {len(labels_df):,} label epochs.")
    return motion_df, labels_df, subject_ids
