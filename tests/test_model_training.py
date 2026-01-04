import pandas as pd
from scripts.data_modeling import train_model

def test_model_training_runs():
    df = pd.DataFrame({
        "age": [60, 55, 45, 50],
        "trestbps": [140, 130, 120, 135],
        "chol": [250, 220, 180, 200],
        "cardiac_risk_score": [4, 3, 1, 2],
        "num": [1, 1, 0, 0]
    })

    # Should not raise exception
    train_model(df)
