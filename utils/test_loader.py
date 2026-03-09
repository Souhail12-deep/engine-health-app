import pandas as pd

df = pd.read_csv("data/test/test_FD001.txt", sep=r"\s+", header=None)

columns = (
    ["unit", "cycle", "op1", "op2", "op3"] +
    [f"s{i}" for i in range(1, 22)]
)

df.columns = columns

print(df.head())
print(df.shape)