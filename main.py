import os
import numpy as np
from data_preprocessing import DatasetLoader
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

def print_metrics_table(results_dict):
    """Print a formatted table of results."""
   
    print("EVALUATION RESULTS - ALL ALGORITHMS AND DATASETS")
    
    print(f"{'Dataset':<12} {'Algorithm':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)
    
    datasets = ['enron1', 'enron2', 'enron4']
    algorithms = ['multinomial', 'bernoulli', 'logistic_bow', 'logistic_bernoulli']
    
    for dataset in datasets:
        for algorithm in algorithms:
            key = f'{dataset}_{algorithm}'
            if key in results_dict:
                metrics = results_dict[key]
                algo_name = {
                    'multinomial': 'Multinomial NB',
                    'bernoulli': 'Bernoulli NB',
                    'logistic_bow': 'Logistic Regression (BoW)',
                    'logistic_bernoulli': 'Logistic Regression (Bernoulli)'
                }[algorithm]
                
                print(f"{dataset:<12} {algo_name:<35} "
                      f"{metrics['accuracy']:<12.4f} "
                      f"{metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<12.4f} "
                      f"{metrics['f1_score']:<12.4f}")

def print_comparison_analysis(results_dict):
    """Print comparative analysis across algorithms and datasets."""
   
    print("\nCOMPARATIVE ANALYSIS")
    
    
    datasets = ['enron1', 'enron2', 'enron4']
    
    # Compare by dataset
    print("\nBest algorithm per dataset (by accuracy):")
    print("-" * 50)
    for dataset in datasets:
        dataset_results = {k.split('_', 1)[1]: v['accuracy'] 
                          for k, v in results_dict.items() 
                          if k.startswith(dataset)}
        if dataset_results:
            best_algo = max(dataset_results, key=dataset_results.get)
            best_acc = dataset_results[best_algo]
            algo_name = {
                'multinomial': 'Multinomial NB',
                'bernoulli': 'Bernoulli NB',
                'logistic_bow': 'Logistic Regression (BoW)',
                'logistic_bernoulli': 'Logistic Regression (Bernoulli)'
            }.get(best_algo, best_algo)
            print(f"  {dataset}: {algo_name} ({best_acc:.4f})")
    
    # Average performance by algorithm
    print("\nAverage accuracy by algorithm:")
    print("-" * 50)
    mnb_accs = [v['accuracy'] for k, v in results_dict.items() if k.endswith('_multinomial')]
    bnb_accs = [v['accuracy'] for k, v in results_dict.items() if k.endswith('_bernoulli') and 'logistic' not in k]
    lr_bow_accs = [v['accuracy'] for k, v in results_dict.items() if k.endswith('_logistic_bow')]
    lr_bern_accs = [v['accuracy'] for k, v in results_dict.items() if k.endswith('_logistic_bernoulli')]

    if mnb_accs:
        print(f"  Multinomial NB: {np.mean(mnb_accs):.4f}")
    if bnb_accs:
        print(f"  Bernoulli NB: {np.mean(bnb_accs):.4f}")
    if lr_bow_accs:
        print(f"  Logistic Regression (BoW): {np.mean(lr_bow_accs):.4f}")
    if lr_bern_accs:
        print(f"  Logistic Regression (Bernoulli): {np.mean(lr_bern_accs):.4f}")
    

def main():
    """Main function to run entire project"""
     
    print("CS 6375 PROJECT 1: NAIVE BAYES AND LOGISTIC REGRESSION")
    print("Spam Email Classification")
     
    
    datasets = ['enron1', 'enron2', 'enron4']
    
    # Step 1: Check/Create preprocessed data
    if not os.path.exists('results/enron1_bow_train.csv'):
        print("\nPreprocessing data...")
        loader = DatasetLoader("data")
        for dataset in datasets:
            try:
                data_dict = loader.create_feature_matrices(dataset)
                loader.save_to_csv(data_dict, dataset)
            except Exception as e:
                print(f"Error processing {dataset}: {e}")
    
    # Step 2: Train and evaluate all models
    print("\nTraining and Evaluating Models...")
    
    
    all_results = {}
    
    for dataset in datasets:
        
        # Multinomial Naive Bayes (BoW features)
        try:
            X_train, y_train = load_dataset(f'results/{dataset}_bow_train.csv')
            X_test, y_test = load_dataset(f'results/{dataset}_bow_test.csv')
            
            mnb = MultinomialNaiveBayes(alpha=1.0)
            mnb.fit(X_train, y_train)
            y_pred = mnb.predict(X_test)
            
            metrics = calculate_metrics(y_test, y_pred)
            all_results[f'{dataset}_multinomial'] = metrics
        except Exception as e:
            print(f"Error with {dataset} Multinomial NB: {e}")
        
        # Bernoulli Naive Bayes (Bernoulli features)
        try:
            X_train, y_train = load_dataset(f'results/{dataset}_bernoulli_train.csv')
            X_test, y_test = load_dataset(f'results/{dataset}_bernoulli_test.csv')
            
            bnb = BernoulliNaiveBayes(alpha=1.0)
            bnb.fit(X_train, y_train)
            y_pred = bnb.predict(X_test)
            
            metrics = calculate_metrics(y_test, y_pred)
            all_results[f'{dataset}_bernoulli'] = metrics
        except Exception as e:
            print(f"Error with {dataset} Bernoulli NB: {e}")
        
        # Logistic Regression (BoW features)
        try:
            X_train, y_train = load_dataset(f'results/{dataset}_bow_train.csv')
            X_test, y_test = load_dataset(f'results/{dataset}_bow_test.csv')
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
            )
            
            best_params = tune_hyperparameters(X_train_split, y_train_split, X_val_split, y_val_split, verbose=False)
            
            lr_bow = LogisticRegression(
                learning_rate=best_params['learning_rate'],
                max_iterations=2000,
                regularization_lambda=best_params['regularization_lambda'],
                verbose=False
            )
            lr_bow.fit(X_train, y_train)
            y_pred = lr_bow.predict(X_test)
            
            metrics = calculate_metrics(y_test, y_pred)
            all_results[f'{dataset}_logistic_bow'] = metrics
            all_results[f'{dataset}_logistic_bow']['best_params'] = best_params
        except Exception as e:
            print(f"Error with {dataset} LR (BoW): {e}")
        
        # Logistic Regression (Bernoulli features)
        try:
            X_train, y_train = load_dataset(f'results/{dataset}_bernoulli_train.csv')
            X_test, y_test = load_dataset(f'results/{dataset}_bernoulli_test.csv')
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
            )
            
            best_params = tune_hyperparameters(X_train_split, y_train_split, X_val_split, y_val_split, verbose=False)
            
            lr_bern = LogisticRegression(
                learning_rate=best_params['learning_rate'],
                max_iterations=2000,
                regularization_lambda=best_params['regularization_lambda'],
                verbose=False
            )
            lr_bern.fit(X_train, y_train)
            y_pred = lr_bern.predict(X_test)
            
            metrics = calculate_metrics(y_test, y_pred)
            all_results[f'{dataset}_logistic_bernoulli'] = metrics
            all_results[f'{dataset}_logistic_bernoulli']['best_params'] = best_params
        except Exception as e:
            print(f"Error with {dataset} LR (Bernoulli): {e}")
    
    
    print("\nRESULTS")
     
    
    print_metrics_table(all_results)
    print_comparison_analysis(all_results)
    
    # Overall Summary
    
    print("\nOVERALL SUMMARY")
     
    
    mnb_avg = sum([v['accuracy'] for k, v in all_results.items() if 'multinomial' in k]) / 3
    bnb_avg = sum([all_results[f'{d}_bernoulli']['accuracy'] for d in datasets if f'{d}_bernoulli' in all_results]) / 3
    lr_bow_avg = sum([v['accuracy'] for k, v in all_results.items() if 'logistic_bow' in k]) / 3
    lr_bern_avg = sum([v['accuracy'] for k, v in all_results.items() if 'logistic_bernoulli' in k]) / 3
    
    print(f"\nMultinomial Naive Bayes Average:          {mnb_avg:.4f}")
    print(f"Bernoulli Naive Bayes Average:            {bnb_avg:.4f}")
    print(f"Logistic Regression (BoW) Average:        {lr_bow_avg:.4f}")
    print(f"Logistic Regression (Bernoulli) Average:  {lr_bern_avg:.4f}")
    
    best_avg = max(mnb_avg, bnb_avg, lr_bow_avg, lr_bern_avg)
    print(f"\nBest Overall Average: {best_avg:.4f}")
    
    # Display best hyperparameters for Logistic Regression
    
    print("\nBEST HYPERPARAMETERS (LOGISTIC REGRESSION)")
     
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        if f'{dataset}_logistic_bow' in all_results and 'best_params' in all_results[f'{dataset}_logistic_bow']:
            params = all_results[f'{dataset}_logistic_bow']['best_params']
            print(f"  BoW:       lr={params['learning_rate']:.4f}, lambda={params['regularization_lambda']:.4f}")
        if f'{dataset}_logistic_bernoulli' in all_results and 'best_params' in all_results[f'{dataset}_logistic_bernoulli']:
            params = all_results[f'{dataset}_logistic_bernoulli']['best_params']
            print(f"  Bernoulli: lr={params['learning_rate']:.4f}, lambda={params['regularization_lambda']:.4f}")
    
    
    print("\nPROJECT COMPLETE!")
     

if __name__ == "__main__":
    main()