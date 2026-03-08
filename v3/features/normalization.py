"""Per-subject z-score feature normalization."""


def add_zscore_features(epoch_df, feature_cols):
    """Z-score normalize features within each subject.

    Replaces original feature columns with their z-scored versions.
    This removes inter-subject baseline differences.
    """
    df = epoch_df.copy()
    for col in feature_cols:
        df[col] = df.groupby('subject_id')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    return df
