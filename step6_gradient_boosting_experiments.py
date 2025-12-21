"""
Step 6: Gradient Boosting Experiments
Purpose: Run Gradient Boosting classifier on all 15 datasets
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import json
import time
from step2_experiment_pipeline import run_experiment

print("="*70)
print("STEP 6: GRADIENT BOOSTING EXPERIMENTS")
print("="*70)

CLAUSE_COUNTS = [300, 500, 1000, 1500, 1800]
DATA_SIZES = [100, 1000, 5000]

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'subsample': [0.8, 1.0]
}

print(f"Parameter grid: {param_grid}")
print(f"Total combinations: {3*2*3*2*2*2} = 144 per dataset")
print(f"Total experiments: 15 datasets × 144 combinations = 2160 fits\n")

results = []
start_time = time.time()

for clause_count in CLAUSE_COUNTS:
    for data_size in DATA_SIZES:
        classifier = GradientBoostingClassifier(random_state=42)
        
        result = run_experiment(
            classifier=classifier,
            param_grid=param_grid,
            clause_count=clause_count,
            data_size=data_size,
            classifier_name="GradientBoosting"
        )
        
        results.append(result)

elapsed_time = time.time() - start_time

results_df = pd.DataFrame(results)
results_df.to_csv('results/gradient_boosting_results.csv', index=False)

with open('results/gradient_boosting_detailed_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"Gradient Boosting experiments complete in {elapsed_time/60:.2f} minutes")
print("Results saved to results/gradient_boosting_results.csv")
print(f"{'='*70}")

print("\nResults Summary:")
summary_cols = ['clause_count', 'data_size', 'train_accuracy', 'test_accuracy', 
                'train_f1', 'test_f1', 'train_time']
print(results_df[summary_cols].to_string(index=False))