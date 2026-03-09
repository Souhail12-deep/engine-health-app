import pandas as pd
import numpy as np
import random
import os
from config import DATA_PATH

def load_test_data(path=None):
    """
    Load test data from file
    If path is None, use default from config
    """
    if path is None:
        path = DATA_PATH
    
    df = pd.read_csv(path, sep=r"\s+", header=None)

    columns = (
        ["unit", "cycle", "op1", "op2", "op3"] +
        [f"s{i}" for i in range(1, 22)]
    )

    df.columns = columns
    return df


def pick_random_engine_cycle(df, window_size=30):
    unit = random.choice(df["unit"].unique())
    engine_df = df[df["unit"] == unit].sort_values("cycle")

    if len(engine_df) < window_size:
        return None

    window = engine_df.tail(window_size)

    return unit, window.reset_index(drop=True)