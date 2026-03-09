import numpy as np
import pandas as pd

SELECTED_SENSORS = [
    "s2","s3","s4","s7","s8","s9","s11","s12",
    "s15","s17","s20","s21"
]

def build_engineered_features(window_df):

    df = window_df[SELECTED_SENSORS].copy()

    features = []

    # --- rolling stats over the window ---
    rolling_mean = df.mean()
    rolling_std = df.std()

    # --- delta = last - first ---
    delta = df.iloc[-1] - df.iloc[0]

    # --- slope (linear trend) ---
    x = np.arange(len(df))

    slopes = []
    for col in df.columns:
        y = df[col].values
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)

    slopes = pd.Series(slopes, index=df.columns)

    # combine all features
    engineered = pd.concat([
        rolling_mean.add_suffix("_mean"),
        rolling_std.add_suffix("_std"),
        delta.add_suffix("_delta"),
        slopes.add_suffix("_slope")
    ])

    return engineered.values.reshape(1, -1)