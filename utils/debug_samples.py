import pandas as pd
import os

# Load the samples
samples_path = os.path.join("data", "test", "scenario_samples.csv")
df = pd.read_csv(samples_path)

print("=== SAMPLE DATA ===")
print(f"Total samples: {len(df)}")
print("\nFirst 5 samples:")
print(df[['engine_id', 'cycle', 'scenario']].head(10))

# Check one specific sample
sample = df.iloc[0]
print(f"\nChecking first sample: Engine {sample['engine_id']}, Cycle {sample['cycle']}")