"""
Step 1: Data Exploration
Purpose: Verify environment setup and understand dataset structure
"""

import pandas as pd
import numpy as np
import sklearn
import os

print("=" * 60)
print("STEP 1: Environment Setup and Data Exploration")
print("=" * 60)

print("\n1. Checking library versions...")
print(f"   scikit-learn version: {sklearn.__version__}")
print(f"   pandas version: {pd.__version__}")
print(f"   numpy version: {np.__version__}")

DATA_DIR = "./all_data" 

print(f"\n2. Looking for datasets in: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    print(f"   WARNING: Directory {DATA_DIR} does not exist!")
    print(f"   Please create it and place your CSV files there.")
else:
    print("   Directory found!")
    
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"\n   Found {len(csv_files)} CSV files")
    
    print("\n   Sample files:")
    for f in sorted(csv_files)[:5]:
        print(f"      - {f}")

print("\n3. Loading sample dataset: train_c300_d100.csv")
sample_file = os.path.join(DATA_DIR, "train_c300_d100.csv")

if os.path.exists(sample_file):
    df = pd.read_csv(sample_file, header=None)
    
    print(f"\n   Dataset shape: {df.shape}")
    print(f"   Number of features: {df.shape[1] - 1}")
    print(f"   Number of samples: {df.shape[0]}")
    
    labels = df.iloc[:, -1]
    print("\n   Class distribution:")
    print(f"      Class 0: {(labels == 0).sum()} samples")
    print(f"      Class 1: {(labels == 1).sum()} samples")
    
    print("\n   First 3 rows (showing first 10 columns + label):")
    print(df.iloc[:3, :11])
    
    all_values = df.values.flatten()
    unique_values = np.unique(all_values)
    print(f"\n   Unique values in dataset: {unique_values}")
    
else:
    print("\n   ERROR: Sample file not found!")
    print(f"   Please ensure your datasets are in {DATA_DIR}")

print("\n" + "=" * 60)
print("Step 1 Complete!")
print("=" * 60)