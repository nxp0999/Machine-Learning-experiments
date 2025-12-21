"""
Step 8: Comparative Analysis
Purpose: Create final tables and analysis for Experiment 5
"""

import pandas as pd

print("="*70)
print("STEP 8: COMPARATIVE ANALYSIS")
print("="*70)

dt_df = pd.read_csv('results/decision_tree_results.csv')
bag_df = pd.read_csv('results/bagging_results.csv')
rf_df = pd.read_csv('results/random_forest_results.csv')
gb_df = pd.read_csv('results/gradient_boosting_results.csv')

accuracy_data = []
f1_data = []

for _, row in dt_df.iterrows():
    dataset = f"c{row['clause_count']}_d{row['data_size']}"
    
    dt_acc = row['test_accuracy']
    bag_acc = bag_df[(bag_df['clause_count']==row['clause_count']) & 
                     (bag_df['data_size']==row['data_size'])]['test_accuracy'].values[0]
    rf_acc = rf_df[(rf_df['clause_count']==row['clause_count']) & 
                   (rf_df['data_size']==row['data_size'])]['test_accuracy'].values[0]
    gb_acc = gb_df[(gb_df['clause_count']==row['clause_count']) & 
                   (gb_df['data_size']==row['data_size'])]['test_accuracy'].values[0]
    
    accuracy_data.append({
        'Dataset': dataset,
        'DecisionTree': dt_acc,
        'Bagging': bag_acc,
        'RandomForest': rf_acc,
        'GradientBoosting': gb_acc
    })
    
    dt_f1 = row['test_f1']
    bag_f1 = bag_df[(bag_df['clause_count']==row['clause_count']) & 
                    (bag_df['data_size']==row['data_size'])]['test_f1'].values[0]
    rf_f1 = rf_df[(rf_df['clause_count']==row['clause_count']) & 
                  (rf_df['data_size']==row['data_size'])]['test_f1'].values[0]
    gb_f1 = gb_df[(gb_df['clause_count']==row['clause_count']) & 
                  (gb_df['data_size']==row['data_size'])]['test_f1'].values[0]
    
    f1_data.append({
        'Dataset': dataset,
        'DecisionTree': dt_f1,
        'Bagging': bag_f1,
        'RandomForest': rf_f1,
        'GradientBoosting': gb_f1
    })

accuracy_table = pd.DataFrame(accuracy_data)
f1_table = pd.DataFrame(f1_data)

accuracy_table.to_csv('results/final_accuracy_table.csv', index=False)
f1_table.to_csv('results/final_f1_table.csv', index=False)

print("\n" + "="*70)
print("CLASSIFICATION ACCURACY TABLE")
print("="*70)
print(accuracy_table.to_string(index=False))

print("\n" + "="*70)
print("F1 SCORE TABLE")
print("="*70)
print(f1_table.to_string(index=False))

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print("\nAverage Test Accuracy by Classifier:")
print(f"  DecisionTree: {accuracy_table['DecisionTree'].mean():.4f}")
print(f"  Bagging: {accuracy_table['Bagging'].mean():.4f}")
print(f"  RandomForest: {accuracy_table['RandomForest'].mean():.4f}")
print(f"  GradientBoosting: {accuracy_table['GradientBoosting'].mean():.4f}")

print("\nAverage Test F1 Score by Classifier:")
print(f"  DecisionTree: {f1_table['DecisionTree'].mean():.4f}")
print(f"  Bagging: {f1_table['Bagging'].mean():.4f}")
print(f"  RandomForest: {f1_table['RandomForest'].mean():.4f}")
print(f"  GradientBoosting: {f1_table['GradientBoosting'].mean():.4f}")

print("\nTables saved to:")
print("  - results/final_accuracy_table.csv")
print("  - results/final_f1_table.csv")