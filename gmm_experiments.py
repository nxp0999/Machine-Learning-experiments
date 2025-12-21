"""
GMM Experiments Module
Runs Gaussian Mixture Model experiments with different configurations
"""
import numpy as np
from sklearn.mixture import GaussianMixture
import json
import time
import os
from metrics import calculate_gmm_metrics


def run_gmm_single(X, n_components, covariance_type='full', random_state=42, max_iter=100):
    """
    Run a single GMM experiment
    
    Args:
        X: Feature matrix
        n_components: Number of mixture components (clusters)
        covariance_type: 'full' or 'diag'
        random_state: Random seed
        max_iter: Maximum EM iterations
    
    Returns:
        Dictionary containing results
    """
    start_time = time.time()
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=1,  # Single initialization for fair comparison
        random_state=random_state
    )
    
    gmm.fit(X)
    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)  # Soft assignments
    
    elapsed_time = time.time() - start_time
    
    # Check convergence
    if not gmm.converged_:
        print(f"\n  Did not converge within {max_iter} iterations", end=" ")
    
    # Calculate metrics
    metrics = calculate_gmm_metrics(
        X,
        labels,
        log_likelihood=gmm.score(X) * len(X),  # Total log-likelihood
        bic=gmm.bic(X),
        aic=gmm.aic(X),
        n_iterations=gmm.n_iter_
    )
    
    results = {
        'algorithm': 'gmm',
        'n_components': n_components,
        'covariance_type': covariance_type,
        'n_iterations': gmm.n_iter_,
        'converged': gmm.converged_,
        'labels': labels.tolist(),
        'probabilities': probabilities.tolist(),  # Unique to GMM - soft clustering
        'means': gmm.means_.tolist(),
        'weights': gmm.weights_.tolist(),  # Mixture weights
        'metrics': metrics,
        'time': elapsed_time
    }
    
    return results


def run_gmm_experiments(datasets, k_values=[2, 3, 4, 5], 
                       covariance_types=['full', 'diag']):
    """
    Run all GMM experiments across datasets and configurations
    
    Args:
        datasets: Dictionary of datasets {'name': {'X': array, 'y': array}}
        k_values: List of k values to test
        covariance_types: List of covariance types
    
    Returns:
        Dictionary of all results
    """
    all_results = {}
    
    for dataset_name, data in datasets.items():
        X, y_true = data['X'], data['y']
        
        print(f"\n{'='*60}")
        print(f"Running GMM on {dataset_name} dataset")
        print(f"{'='*60}")
        
        dataset_results = []
        
        for k in k_values:
            for cov_type in covariance_types:
                print(f"  k={k}, cov={cov_type:8s}...", end=" ")
                
                results = run_gmm_single(
                    X,
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=42
                )
                
                results['dataset'] = dataset_name
                results['true_labels'] = y_true.tolist()
                dataset_results.append(results)
                
                conv_str = "✓" if results['converged'] else "✗"
                print(f"{conv_str} Silhouette={results['metrics']['silhouette']:.3f}, "
                      f"BIC={results['metrics']['bic']:7.2f}, "
                      f"AIC={results['metrics']['aic']:7.2f}, "
                      f"Iters={results['n_iterations']:2d}, "
                      f"Time={results['time']:.3f}s")
        
        all_results[dataset_name] = dataset_results
    
    return all_results


def save_results(results, filename='results/gmm_results.json'):
    """
    Save results to JSON file
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved to {filename}")


def print_summary_statistics(results):
    """Print summary statistics across all experiments"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        
        for k in sorted(set(r['n_components'] for r in dataset_results)):
            print(f"  k={k}:")
            for cov_type in ['full', 'diag']:
                matching = [r for r in dataset_results 
                           if r['n_components'] == k and r['covariance_type'] == cov_type]
                if matching:
                    r = matching[0]
                    conv_str = "✓" if r['converged'] else "✗"
                    print(f"    {cov_type:8s}: {conv_str} Silhouette={r['metrics']['silhouette']:.3f}, "
                          f"BIC={r['metrics']['bic']:7.2f}, "
                          f"AIC={r['metrics']['aic']:7.2f}")


def analyze_covariance_effect(results):
    """Analyze the effect of covariance type"""
    print("\n" + "="*60)
    print("COVARIANCE TYPE COMPARISON")
    print("="*60)
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        
        full_results = [r for r in dataset_results if r['covariance_type'] == 'full']
        diag_results = [r for r in dataset_results if r['covariance_type'] == 'diag']
        
        avg_full_sil = np.mean([r['metrics']['silhouette'] for r in full_results])
        avg_diag_sil = np.mean([r['metrics']['silhouette'] for r in diag_results])
        avg_full_bic = np.mean([r['metrics']['bic'] for r in full_results])
        avg_diag_bic = np.mean([r['metrics']['bic'] for r in diag_results])
        
        print(f"  Full covariance:")
        print(f"    Avg Silhouette: {avg_full_sil:.3f}")
        print(f"    Avg BIC: {avg_full_bic:.2f}")
        print(f"  Diagonal covariance:")
        print(f"    Avg Silhouette: {avg_diag_sil:.3f}")
        print(f"    Avg BIC: {avg_diag_bic:.2f}")
        
        # BIC comparison (lower is better)
        if avg_full_bic < avg_diag_bic:
            print(f"  → Full covariance preferred (BIC is {avg_diag_bic - avg_full_bic:.2f} points lower)")
        else:
            print(f"  → Diagonal covariance preferred (BIC is {avg_full_bic - avg_diag_bic:.2f} points lower)")


if __name__ == "__main__":
    from data_generation import generate_all_datasets
    
    # Generate datasets
    datasets = generate_all_datasets()
    
    # Run experiments
    results = run_gmm_experiments(datasets)
    
    # Save results
    save_results(results)
    
    # Print summary
    print_summary_statistics(results)
    
    # Analyze covariance effect
    analyze_covariance_effect(results)