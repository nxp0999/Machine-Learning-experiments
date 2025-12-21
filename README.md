# CS 6375 Machine Learning - Project 2

This project evaluates four tree-based machine learning classifiers—Decision Tree, Bagging, Random Forest, and Gradient Boosting—on synthetic Boolean satisfiability (CNF) datasets and the MNIST handwritten digit dataset. The goal is to compare classifier performance across varying problem complexities and data sizes through comprehensive hyperparameter tuning and rigorous evaluation.

The project implements a systematic experimental framework using scikit-learn's GridSearchCV for hyperparameter optimization with 3-fold cross-validation. Each classifier is evaluated on 15 CNF datasets (5 clause counts × 3 data sizes) to assess how training data size and problem complexity affect classification accuracy and F1 score. Additionally, all four classifiers are evaluated on MNIST to compare performance across different problem domains.

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy

## Installation

```bash
pip install scikit-learn pandas numpy

```

## Project Structure

````
project 2/
└── src/
    ├── check_setup.py                          # Environment verification
    ├── step1_data_exploration.py               # Step 1: Data exploration
    ├── step2_experiment_pipeline.py            # Step 2: Core pipeline functions
    ├── step3_decision_tree_experiments.py      # Step 3: Decision Tree experiments
    ├── step4_bagging_experiments.py            # Step 4: Bagging experiments
    ├── step5_random_forest_experiments.py      # Step 5: Random Forest experiments
    ├── step6_gradient_boosting_experiments.py  # Step 6: Gradient Boosting experiments
    ├── step7_mnist_experiments.py              # Step 7: MNIST evaluation
    ├── step8_analysis.py                       # Step 8: Comparative analysis
    ├── master.py                               # Master script (runs all)
    ├── all_data/                               # Dataset folder (45 CSV files)
    │   ├── train_c300_d100.csv
    │   ├── valid_c300_d100.csv
    │   ├── test_c300_d100.csv
    │   └── ... (42 more files)
    └── results/                                # Output folder (generated)
        ├── decision_tree_results.csv
        ├── decision_tree_detailed_results.json
        ├── bagging_results.csv
        ├── bagging_detailed_results.json
        ├── random_forest_results.csv
        ├── random_forest_detailed_results.json
        ├── gradient_boosting_results.csv
        ├── gradient_boosting_detailed_results.json
        ├── mnist_results.csv
        ├── final_accuracy_table.csv
        └── final_f1_table.csv


## ENVIRONMENT CHECK

python3 check_setup.py
```
Expected output:
```
============================================================
CHECKING YOUR SETUP
============================================================
1. Python Version: 3.13.x
   ✓ Python version is compatible
2. Checking Required Libraries:
   ✓ scikit-learn (version 1.7.2)
   ✓ pandas (version 2.3.3)
   ✓ numpy (version 2.3.3)
   ✓ matplotlib (version 3.10.7)
============================================================
✓ ALL REQUIREMENTS MET!
============================================================


## Check Setup
```bash
python3 check_setup.py
````

## Run the setup scripts (./src)
```bash
python3 step1_data_exploration.py               
python3 step2_experiment_pipeline.py
```

## Run All Experiments (with the single master script) (./src)
```bash
python3 master.py
```

## Run All Experiments (separately in order) (./src)

```bash
python3 step3_decision_tree_experiments.py
python3 step4_bagging_experiments.py
python3 step5_random_forest_experiments.py
python3 step6_gradient_boosting_experiments.py
python3 step7_mnist_experiments.py
python3 step8_analysis.py
```

## Output Files (./src/results)

- `decision_tree_results.csv` - Decision Tree results
- `bagging_results.csv` - Bagging results
- `random_forest_results.csv` - Random Forest results
- `gradient_boosting_results.csv` - Gradient Boosting results
- `mnist_results.csv` - MNIST results
- `final_accuracy_table.csv` - Combined accuracy table
- `final_f1_table.csv` - Combined F1 score table

