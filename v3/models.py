"""Model pipeline builders."""

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from config import RANDOM_STATE, RF_PARAMS_3, RF_PARAMS_5, XGB_PARAMS_3, XGB_PARAMS_5


def build_pipeline(model_type='rf', stages=3, use_smote=True, **override_params):
    """Build an imblearn Pipeline with SMOTE and a classifier.

    Args:
        model_type: 'rf' or 'xgb'
        stages: 3 or 5 (selects default params)
        use_smote: whether to include SMOTE in pipeline
        **override_params: override any default model hyperparameters

    Returns:
        imblearn Pipeline, requires_encoding (bool)
    """
    if model_type == 'rf':
        defaults = RF_PARAMS_3 if stages == 3 else RF_PARAMS_5
        params = {**defaults, **override_params}
        model = RandomForestClassifier(**params)
        requires_encoding = False
    elif model_type == 'xgb':
        import xgboost as xgb
        defaults = XGB_PARAMS_3 if stages == 3 else XGB_PARAMS_5
        params = {**defaults, **override_params}
        if stages != 3:
            params['num_class'] = stages
        model = xgb.XGBClassifier(**params)
        requires_encoding = True
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    steps = []
    if use_smote:
        steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
    steps.append(('clf', model))

    return Pipeline(steps), requires_encoding
