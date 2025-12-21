"""
Step 2: Experiment Pipeline
Purpose: Reusable functions for training and evaluating classifiers
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import os
import time

DATA_DIR = "./all_data"

def load_dataset(clause_count, data_size):
    train_file = f"train_c{clause_count}_d{data_size}.csv"
    valid_file = f"valid_c{clause_count}_d{data_size}.csv"
    test_file = f"test_c{clause_count}_d{data_size}.csv"
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, train_file), header=None)
    valid_df = pd.read_csv(os.path.join(DATA_DIR, valid_file), header=None)
    test_df = pd.read_csv(os.path.join(DATA_DIR, test_file), header=None)
    
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    
    X_valid = valid_df.iloc[:, :-1].values
    y_valid = valid_df.iloc[:, -1].values
    
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def run_experiment(classifier, param_grid, clause_count, data_size, classifier_name):
    print(f"\n{'='*60}")
    print(f"Running {classifier_name} on c{clause_count}_d{data_size}")
    print(f"{'='*60}")
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
    
    print(f"Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}, Test: {X_test.shape[0]}")
    
    print("Tuning hyperparameters...")
    train_start = time.time()
    
    grid_search = GridSearchCV(
        classifier, 
        param_grid, 
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    X_train_full = np.vstack([X_train, X_valid])
    y_train_full = np.concatenate([y_train, y_valid])
    
    print(f"Retraining on train+valid ({X_train_full.shape[0]} samples)...")
    
    best_classifier = grid_search.best_estimator_
    best_classifier.fit(X_train_full, y_train_full)
    
    train_time = time.time() - train_start
    
    y_train_pred = best_classifier.predict(X_train_full)
    train_accuracy = accuracy_score(y_train_full, y_train_pred)
    train_f1 = f1_score(y_train_full, y_train_pred)
    
    y_pred = best_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
    print(f"Test Acc: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
    print(f"Training time: {train_time:.2f}s")
    
    return {
        'classifier': classifier_name,
        'clause_count': clause_count,
        'data_size': data_size,
        'best_params': grid_search.best_params_,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_time': train_time
    }