"""
Step 7: MNIST Experiments
Purpose: Evaluate all 4 classifiers on MNIST dataset
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time

print("="*70)
print("STEP 7: MNIST EXPERIMENTS")
print("="*70)

print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X / 255.0

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

results = []

classifiers = {
    'DecisionTree': DecisionTreeClassifier(
        criterion='entropy',
        max_depth=20,
        min_samples_split=10,
        random_state=42
    ),
    'Bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=50,
        max_samples=0.7,
        random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

for name, clf in classifiers.items():
    print(f"\n{'='*70}")
    print(f"Training {name}...")
    print(f"{'='*70}")
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Training time: {train_time/60:.2f} minutes")
    print("Evaluating on test set...")
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    results.append({
        'classifier': name,
        'accuracy': accuracy,
        'train_time_minutes': train_time/60
    })

print("\n" + "="*70)
print("MNIST EXPERIMENTS COMPLETE!")
print("="*70)

results_df = pd.DataFrame(results)
results_df.to_csv('results/mnist_results.csv', index=False)

print("\nResults Summary:")
print(results_df.to_string(index=False))
print("\nResults saved to results/mnist_results.csv")