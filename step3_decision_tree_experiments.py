"""
Step 3: Decision Tree Experiments
Purpose: Run Decision Tree classifier on all 15 datasets
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import json
import time
from step2_experiment_pipeline import run_experiment

print("="*70)
print("STEP 3: DECISION TREE EXPERIMENTS")
print("="*70)

CLAUSE_COUNTS = [300, 500, 1000, 1500, 1800]
DATA_SIZES = [100, 1000, 5000]

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4]
}

print(f"Parameter grid: {param_grid}")
print(f"Total combinations: {2*3*2*2} = 24 per dataset")
print(f"Total experiments: 15 datasets × 24 combinations = 360 fits\n")

results = []
start_time = time.time()

for clause_count in CLAUSE_COUNTS:
    for data_size in DATA_SIZES:
        classifier = DecisionTreeClassifier(random_state=42)
        
        result = run_experiment(
            classifier=classifier,
            param_grid=param_grid,
            clause_count=clause_count,
            data_size=data_size,
            classifier_name="DecisionTree"
        )
        
        results.append(result)

elapsed_time = time.time() - start_time

results_df = pd.DataFrame(results)
results_df.to_csv('results/decision_tree_results.csv', index=False)

with open('results/decision_tree_detailed_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"Decision Tree experiments complete in {elapsed_time/60:.2f} minutes")
print("Results saved to results/decision_tree_results.csv")
print(f"{'='*70}")

print("\nResults Summary:")
summary_cols = ['clause_count', 'data_size', 'train_accuracy', 'test_accuracy', 
                'train_f1', 'test_f1', 'train_time']
print(results_df[summary_cols].to_string(index=False))