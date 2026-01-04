import pandas as pd
from scripts.data_preparation import prepare_data

def test_prepare_data_no_nan():
    df = pd.DataFrame({
        "age": [60, 55],
        "trestbps": [140, 130],
        "chol": [250, 200],
        "sex": ["Male", "Female"],
        "num": [1, 0]
    })

    processed = prepare_data(df)

    # No NaNs after preprocessing
    assert processed.isna().sum().sum() == 0


def test_target_column_exists():
    df = pd.DataFrame({
        "age": [50],
        "trestbps": [120],
        "chol": [180],
        "sex": ["Male"],
        "num": [0]
    })

    processed = prepare_data(df)
    assert "num" in processed.columns
