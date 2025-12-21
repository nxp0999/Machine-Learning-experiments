"""
Master Script: Run All Experiments
Purpose: Execute all experiments (Steps 3-8) in sequence
"""

import subprocess
import sys
import time

print("="*70)
print("MASTER SCRIPT: RUNNING ALL EXPERIMENTS")
print("="*70)

experiments = [
    ("Step 3: Decision Tree", "step3_decision_tree_experiments.py"),
    ("Step 4: Bagging", "step4_bagging_experiments.py"),
    ("Step 5: Random Forest", "step5_random_forest_experiments.py"),
    ("Step 6: Gradient Boosting", "step6_gradient_boosting_experiments.py"),
    ("Step 7: MNIST", "step7_mnist_experiments.py"),
    ("Step 8: Analysis", "step8_analysis.py")
]

total_start = time.time()

for i, (name, script) in enumerate(experiments, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/6] Starting: {name}")
    print(f"{'='*70}\n")
    
    step_start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=False
        )
        
        step_time = time.time() - step_start
        print(f"\n✓ {name} completed in {step_time/60:.2f} minutes")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR in {name}")
        print(f"Script failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR in {name}: {e}")
        sys.exit(1)

total_time = time.time() - total_start

print(f"\n{'='*70}")
print("ALL EXPERIMENTS COMPLETE!")
print(f"{'='*70}")
print(f"Total runtime: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
print("\nResults available in results/ folder:")
print("  - decision_tree_results.csv")
print("  - bagging_results.csv")
print("  - random_forest_results.csv")
print("  - gradient_boosting_results.csv")
print("  - mnist_results.csv")
print("  - final_accuracy_table.csv")
print("  - final_f1_table.csv")