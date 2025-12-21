import sys
import os
sys.path.append('src')

import numpy as np
from naive_bayes import MultinomialNaiveBayes, BernoulliNaiveBayes, load_dataset
from logistic_regression import LogisticRegression, train_test_split, tune_hyperparameters

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def main():
    """Evaluate all algorithms"""
    
    # Check if results files exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("Results directory not found! Running data preprocessing first...")
        return
    
    # List all files in results directory
    print("Files in results directory:")
    try:
        result_files = os.listdir(results_dir)
        for f in sorted(result_files):
            print(f"  {f}")
        print()
    except Exception as e:
        print(f"Error reading results directory: {e}")
        return
    
    datasets = ['enron1', 'enron2', 'enron4']
    all_results = {}
    
    
    print("COMPREHENSIVE EVALUATION RESULTS")
    
    
    for dataset in datasets:
        print(f"\n=== {dataset.upper()} ===")
        
        # Multinomial Naive Bayes (BoW)
        bow_train_file = f'{results_dir}/{dataset}_bow_train.csv'
        bow_test_file = f'{results_dir}/{dataset}_bow_test.csv'
        
        if os.path.exists(bow_train_file) and os.path.exists(bow_test_file):
            print("Multinomial Naive Bayes (Bag of Words):")
            try:
                X_train, y_train = load_dataset(bow_train_file)
                X_test, y_test = load_dataset(bow_test_file)
                
                mnb = MultinomialNaiveBayes(alpha=1.0)
                mnb.fit(X_train, y_train)
                y_pred = mnb.predict(X_test)
                
                metrics = calculate_metrics(y_test, y_pred)
                all_results[f'{dataset}_multinomial'] = metrics
                
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"Multinomial NB: Files not found ({bow_train_file}, {bow_test_file})")
        
        # Bernoulli Naive Bayes
        bernoulli_train_file = f'{results_dir}/{dataset}_bernoulli_train.csv'
        bernoulli_test_file = f'{results_dir}/{dataset}_bernoulli_test.csv'
        
        if os.path.exists(bernoulli_train_file) and os.path.exists(bernoulli_test_file):
            print("\nBernoulli Naive Bayes (Binary features):")
            try:
                X_train, y_train = load_dataset(bernoulli_train_file)
                X_test, y_test = load_dataset(bernoulli_test_file)
                
                bnb = BernoulliNaiveBayes(alpha=1.0)
                bnb.fit(X_train, y_train)
                y_pred = bnb.predict(X_test)
                
                metrics = calculate_metrics(y_test, y_pred)
                all_results[f'{dataset}_bernoulli'] = metrics
                
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"Bernoulli NB: Files not found ({bernoulli_train_file}, {bernoulli_test_file})")
        
        # Logistic Regression (BoW)
        if os.path.exists(bow_train_file) and os.path.exists(bow_test_file):
            print("\nLogistic Regression (Bag of Words):")
            try:
                X_train, y_train = load_dataset(bow_train_file)
                X_test, y_test = load_dataset(bow_test_file)
                
                # Split for hyperparameter tuning
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
                )
                
                # Tune hyperparameters
                print("  Tuning hyperparameters...")
                best_params = tune_hyperparameters(X_train_split, y_train_split, X_val_split, y_val_split)
                print(f"  Best params: lr={best_params['learning_rate']:.4f}, lambda={best_params['regularization_lambda']:.4f}")
                
                # Train final model
                print("  Training final model on full training set...")
                lr_bow = LogisticRegression(
                    learning_rate=best_params['learning_rate'],
                    max_iterations=2000,
                    regularization_lambda=best_params['regularization_lambda']
                )
                lr_bow.fit(X_train, y_train)
                y_pred = lr_bow.predict(X_test)
                
                metrics = calculate_metrics(y_test, y_pred)
                all_results[f'{dataset}_logistic_bow'] = metrics
                all_results[f'{dataset}_logistic_bow']['best_params'] = best_params
                
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        # Logistic Regression (Bernoulli)
        if os.path.exists(bernoulli_train_file) and os.path.exists(bernoulli_test_file):
            print("\nLogistic Regression (Bernoulli features):")
            try:
                X_train, y_train = load_dataset(bernoulli_train_file)
                X_test, y_test = load_dataset(bernoulli_test_file)
                
                # Split for hyperparameter tuning
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
                )
                
                # Tune hyperparameters
                print("  Tuning hyperparameters...")
                best_params = tune_hyperparameters(X_train_split, y_train_split, X_val_split, y_val_split)
                print(f"  Best params: lr={best_params['learning_rate']:.4f}, lambda={best_params['regularization_lambda']:.4f}")
                
                # Train final model
                print("  Training final model on full training set...")
                lr_bern = LogisticRegression(
                    learning_rate=best_params['learning_rate'],
                    max_iterations=2000,
                    regularization_lambda=best_params['regularization_lambda']
                )
                lr_bern.fit(X_train, y_train)
                y_pred = lr_bern.predict(X_test)
                
                metrics = calculate_metrics(y_test, y_pred)
                all_results[f'{dataset}_logistic_bernoulli'] = metrics
                all_results[f'{dataset}_logistic_bernoulli']['best_params'] = best_params
                
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    # Create comprehensive summary table
    if all_results:
        print("\n" + "=" * 100)
        print("SUMMARY TABLE - ALL ALGORITHMS AND METRICS")
        print("=" * 100)
        print(f"{'Dataset':<12} {'Algorithm':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 100)
        
        for dataset in datasets:
            if f'{dataset}_multinomial' in all_results:
                m = all_results[f'{dataset}_multinomial']
                print(f"{dataset:<12} {'Multinomial NB (BoW)':<35} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1_score']:<10.4f}")
            
            if f'{dataset}_bernoulli' in all_results:
                m = all_results[f'{dataset}_bernoulli']
                print(f"{dataset:<12} {'Bernoulli NB (Bernoulli)':<35} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1_score']:<10.4f}")
            
            if f'{dataset}_logistic_bow' in all_results:
                m = all_results[f'{dataset}_logistic_bow']
                print(f"{dataset:<12} {'Logistic Regression (BoW)':<35} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1_score']:<10.4f}")
            
            if f'{dataset}_logistic_bernoulli' in all_results:
                m = all_results[f'{dataset}_logistic_bernoulli']
                print(f"{dataset:<12} {'Logistic Regression (Bernoulli)':<35} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1_score']:<10.4f}")
            
            print()
        
        # Analysis
        print("=" * 100)
        print("\nCOMPARATIVE ANALYSIS")
        print("=" * 100)
        
        print("\nBest algorithm per dataset (by accuracy):")
        print("-" * 60)
        for dataset in datasets:
            dataset_results = {}
            if f'{dataset}_multinomial' in all_results:
                dataset_results['Multinomial NB'] = all_results[f'{dataset}_multinomial']['accuracy']
            if f'{dataset}_bernoulli' in all_results:
                dataset_results['Bernoulli NB'] = all_results[f'{dataset}_bernoulli']['accuracy']
            if f'{dataset}_logistic_bow' in all_results:
                dataset_results['Logistic Regression (BoW)'] = all_results[f'{dataset}_logistic_bow']['accuracy']
            if f'{dataset}_logistic_bernoulli' in all_results:
                dataset_results['Logistic Regression (Bernoulli)'] = all_results[f'{dataset}_logistic_bernoulli']['accuracy']
            
            if dataset_results:
                best_algo = max(dataset_results, key=dataset_results.get)
                best_acc = dataset_results[best_algo]
                print(f"  {dataset}: {best_algo} ({best_acc:.4f})")
        
        # Overall averages
        print("\nAverage accuracy by algorithm:")
        print("-" * 60)
        
        mnb_accs = [all_results[f'{d}_multinomial']['accuracy'] for d in datasets if f'{d}_multinomial' in all_results]
        bnb_accs = [all_results[f'{d}_bernoulli']['accuracy'] for d in datasets if f'{d}_bernoulli' in all_results]
        lr_bow_accs = [all_results[f'{d}_logistic_bow']['accuracy'] for d in datasets if f'{d}_logistic_bow' in all_results]
        lr_bern_accs = [all_results[f'{d}_logistic_bernoulli']['accuracy'] for d in datasets if f'{d}_logistic_bernoulli' in all_results]
        
        if mnb_accs:
            print(f"  Multinomial NB:                   {np.mean(mnb_accs):.4f}")
        if bnb_accs:
            print(f"  Bernoulli NB:                     {np.mean(bnb_accs):.4f}")
        if lr_bow_accs:
            print(f"  Logistic Regression (BoW):        {np.mean(lr_bow_accs):.4f}")
        if lr_bern_accs:
            print(f"  Logistic Regression (Bernoulli):  {np.mean(lr_bern_accs):.4f}")
        
        # Best overall result
        print("\nBest individual result:")
        all_scores = []
        for key, metrics in all_results.items():
            dataset = key.rsplit('_', 1)[0] if '_' in key else key
            algo_map = {
                'multinomial': 'Multinomial NB',
                'bernoulli': 'Bernoulli NB',
                'logistic_bow': 'Logistic Regression (BoW)',
                'logistic_bernoulli': 'Logistic Regression (Bernoulli)'
            }
            algo_name = None
            for k, v in algo_map.items():
                if key.endswith(k):
                    algo_name = v
                    break
            if algo_name:
                all_scores.append((f"{algo_name} - {dataset}", metrics['accuracy']))
        
        if all_scores:
            best_result = max(all_scores, key=lambda x: x[1])
            print(f"  {best_result[0]}: {best_result[1]:.4f}")
        
        # Display best hyperparameters for Logistic Regression
        print("\n" + "=" * 100)
        print("BEST HYPERPARAMETERS (LOGISTIC REGRESSION)")
        print("=" * 100)
        
        for dataset in datasets:
            print(f"\n{dataset.upper()}:")
            if f'{dataset}_logistic_bow' in all_results and 'best_params' in all_results[f'{dataset}_logistic_bow']:
                params = all_results[f'{dataset}_logistic_bow']['best_params']
                print(f"  BoW:       lr={params['learning_rate']:.4f}, lambda={params['regularization_lambda']:.4f}")
            if f'{dataset}_logistic_bernoulli' in all_results and 'best_params' in all_results[f'{dataset}_logistic_bernoulli']:
                params = all_results[f'{dataset}_logistic_bernoulli']['best_params']
                print(f"  Bernoulli: lr={params['learning_rate']:.4f}, lambda={params['regularization_lambda']:.4f}")
    
    else:
        print("\nNo results to analyze - check if CSV files exist and are accessible.")

if __name__ == "__main__":
    main()