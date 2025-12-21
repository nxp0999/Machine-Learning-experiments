"""
K-Means Experiments Module
Runs K-Means clustering experiments with different configurations
"""
import numpy as np
from sklearn.cluster import KMeans
import json
import time
import os
from metrics import calculate_kmeans_metrics


def run_kmeans_single(X, n_clusters, init_method='k-means++', random_state=42, max_iter=300):
    """
    Run a single K-Means experiment
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        init_method: 'k-means++' or 'random'
        random_state: Random seed
        max_iter: Maximum iterations
    
    Returns:
        Dictionary containing results
    """
    start_time = time.time()
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        n_init=1,  # ✅ FIXED: Single run for fair comparison
        max_iter=max_iter,
        random_state=random_state
    )
    
    labels = kmeans.fit_predict(X)
    elapsed_time = time.time() - start_time
    
    # Check for convergence warning
    if kmeans.n_iter_ == max_iter:
        print(f"\n    ⚠️  Did not converge within {max_iter} iterations", end=" ")
    
    # Calculate metrics - FIXED: Include n_iterations parameter
    metrics = calculate_kmeans_metrics(X, labels, kmeans.cluster_centers_, kmeans.n_iter_)
    
    results = {
        'algorithm': 'kmeans',
        'n_clusters': n_clusters,
        'init_method': init_method,
        'n_iterations': kmeans.n_iter_,
        'converged': kmeans.n_iter_ < max_iter,
        'labels': labels.tolist(),
        'centroids': kmeans.cluster_centers_.tolist(),
        'metrics': metrics,
        'time': elapsed_time
    }
    
    return results


def run_kmeans_experiments(datasets, k_values=[2, 3, 4, 5], 
                          init_methods=['random', 'k-means++']):
    """
    Run all K-Means experiments across datasets and configurations
    
    Args:
        datasets: Dictionary of datasets {'name': {'X': array, 'y': array}}
        k_values: List of k values to test
        init_methods: List of initialization methods
    
    Returns:
        Dictionary of all results
    """
    all_results = {}
    
    for dataset_name, data in datasets.items():
        X, y_true = data['X'], data['y']
        
        print(f"\n{'='*60}")
        print(f"Running K-Means on {dataset_name} dataset")
        print(f"{'='*60}")
        
        dataset_results = []
        
        for k in k_values:
            for init in init_methods:
                print(f"  k={k}, init={init:12s}...", end=" ")
                
                results = run_kmeans_single(
                    X, 
                    n_clusters=k, 
                    init_method=init,
                    random_state=42
                )
                
                results['dataset'] = dataset_name
                results['true_labels'] = y_true.tolist()
                dataset_results.append(results)
                
                print(f"Silhouette={results['metrics']['silhouette']:.3f}, "
                      f"Inertia={results['metrics']['inertia']:7.2f}, "
                      f"Iters={results['n_iterations']:2d}, "
                      f"Time={results['time']:.3f}s")
        
        all_results[dataset_name] = dataset_results
    
    return all_results


def save_results(results, filename='results/kmeans_results.json'):
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
    print(f"\n✅ Results saved to {filename}")


def print_summary_statistics(results):
    """Print summary statistics across all experiments"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        
        for k in sorted(set(r['n_clusters'] for r in dataset_results)):
            print(f"  k={k}:")
            for init in ['random', 'k-means++']:
                matching = [r for r in dataset_results 
                           if r['n_clusters'] == k and r['init_method'] == init]
                if matching:
                    r = matching[0]
                    print(f"    {init:12s}: Silhouette={r['metrics']['silhouette']:.3f}, "
                          f"Inertia={r['metrics']['inertia']:7.2f}, "
                          f"Iters={r['n_iterations']:2d}")


if __name__ == "__main__":
    from data_generation import generate_all_datasets
    
    # Generate datasets
    datasets = generate_all_datasets()
    
    # Run experiments
    results = run_kmeans_experiments(datasets)
    
    # Save results
    save_results(results)
    
    # Print summary
    print_summary_statistics(results)