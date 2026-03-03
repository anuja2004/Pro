import numpy as np
import pandas as pd

def build_common_features(df: pd.DataFrame):
    df = df.copy()

    df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

    X = np.column_stack([
        df["amt"].values,
        df["lat"].values,
        df["long"].values,
        df["city_pop"].values,
        df["hour"].values,
    ])

    return X
