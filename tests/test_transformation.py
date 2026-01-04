import pandas as pd
from scripts.data_preparation import transform_data

def test_risk_features_created():
    df = pd.DataFrame({
        "age": [60],
        "trestbps": [150],
        "chol": [260],
        "oldpeak": [2.5],
        "exang": [1],
        "ca": [1],
        "num": [1]
    })

    df_transformed = transform_data(df)

    expected_cols = [
        "age_risk",
        "bp_risk",
        "chol_risk",
        "exercise_risk",
        "st_severity",
        "vessel_risk",
        "cardiac_risk_score"
    ]

    for col in expected_cols:
        assert col in df_transformed.columns
