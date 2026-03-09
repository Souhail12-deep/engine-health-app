import pandas as pd
import os

# Load test data
test_path = os.path.join("data", "test", "test_FD001.txt")
cols = (["unit", "cycle"] + [f"op{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)])
test_df = pd.read_csv(test_path, sep="\s+", header=None, names=cols)

# Check cycle counts per engine
cycle_counts = test_df.groupby('unit')['cycle'].count()
print("Cycle counts per engine:")
print(cycle_counts.describe())
print(f"\nEngines with <30 cycles: {(cycle_counts < 30).sum()}")
print(f"Engines with >=30 cycles: {(cycle_counts >= 30).sum()}")

# Show detailed counts
print("\nDetailed cycle counts:")
for engine_id in sorted(test_df['unit'].unique())[:20]:  # Show first 20
    count = cycle_counts[engine_id]
    print(f"Engine {engine_id}: {count} cycles")