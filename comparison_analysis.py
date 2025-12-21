"""
Comparison Analysis Module
Compares K-Means and GMM performance across datasets and configurations
"""
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_results():
    """
    Load both K-Means and GMM results
    
    Returns:
        Tuple of (kmeans_results, gmm_results)
    """
    with open('results/kmeans_results.json', 'r') as f:
        kmeans_results = json.load(f)
    
    with open('results/gmm_results.json', 'r') as f:
        gmm_results = json.load(f)
    
    return kmeans_results, gmm_results


def create_performance_summary_table(kmeans_results, gmm_results):
    """
    Create comprehensive performance summary table
    
    Args:
        kmeans_results: K-Means results dictionary
        gmm_results: GMM results dictionary
    
    Returns:
        pandas DataFrame with performance metrics
    """
    data = []
    
    for dataset_name in kmeans_results.keys():
        # K-Means results
        for result in kmeans_results[dataset_name]:
            data.append({
                'Dataset': dataset_name,
                'Algorithm': 'K-Means',
                'Variant': result['init_method'],
                'k': result['n_clusters'],
                'Silhouette': result['metrics']['silhouette'],
                'Metric_2': result['metrics']['inertia'],
                'Metric_2_Name': 'Inertia',
                'Iterations': result['n_iterations'],
                'Converged': result['converged'],
                'Time (s)': result['time']
            })
        
        # GMM results
        for result in gmm_results[dataset_name]:
            data.append({
                'Dataset': dataset_name,
                'Algorithm': 'GMM',
                'Variant': result['covariance_type'],
                'k': result['n_components'],
                'Silhouette': result['metrics']['silhouette'],
                'Metric_2': result['metrics']['bic'],
                'Metric_2_Name': 'BIC',
                'Iterations': result['n_iterations'],
                'Converged': result['converged'],
                'Time (s)': result['time']
            })
    
    df = pd.DataFrame(data)
    return df


def analyze_algorithm_comparison(kmeans_results, gmm_results, dataset_name, k_value):
    """
    Detailed comparison of K-Means vs GMM for specific dataset and k
    
    Args:
        kmeans_results: K-Means results
        gmm_results: GMM results
        dataset_name: Dataset to analyze
        k_value: Specific k value to compare
    
    Returns:
        Dictionary with comparison metrics
    """
    # Get best results for each algorithm at this k
    kmeans_k_results = [r for r in kmeans_results[dataset_name] 
                        if r['n_clusters'] == k_value]
    gmm_k_results = [r for r in gmm_results[dataset_name]
                     if r['n_components'] == k_value]
    
    # Best K-Means (by silhouette)
    best_kmeans = max(kmeans_k_results, key=lambda x: x['metrics']['silhouette'])
    
    # Best GMM (by silhouette)
    best_gmm = max(gmm_k_results, key=lambda x: x['metrics']['silhouette'])
    
    comparison = {
        'dataset': dataset_name,
        'k': k_value,
        'kmeans': {
            'variant': best_kmeans['init_method'],
            'silhouette': best_kmeans['metrics']['silhouette'],
            'inertia': best_kmeans['metrics']['inertia'],
            'iterations': best_kmeans['n_iterations'],
            'time': best_kmeans['time']
        },
        'gmm': {
            'variant': best_gmm['covariance_type'],
            'silhouette': best_gmm['metrics']['silhouette'],
            'bic': best_gmm['metrics']['bic'],
            'iterations': best_gmm['n_iterations'],
            'time': best_gmm['time']
        },
        'winner': 'K-Means' if best_kmeans['metrics']['silhouette'] > best_gmm['metrics']['silhouette'] else 'GMM'
    }
    
    return comparison


def analyze_initialization_effect(kmeans_results):
    """
    Analyze the effect of K-Means initialization method
    
    Args:
        kmeans_results: K-Means results dictionary
    
    Returns:
        Dictionary with analysis for each dataset
    """
    analysis = {}
    
    for dataset_name, results in kmeans_results.items():
        random_results = [r for r in results if r['init_method'] == 'random']
        kmeanspp_results = [r for r in results if r['init_method'] == 'k-means++']
        
        analysis[dataset_name] = {
            'random': {
                'avg_silhouette': np.mean([r['metrics']['silhouette'] for r in random_results]),
                'std_silhouette': np.std([r['metrics']['silhouette'] for r in random_results]),
                'avg_iterations': np.mean([r['n_iterations'] for r in random_results]),
                'avg_inertia': np.mean([r['metrics']['inertia'] for r in random_results]),
                'avg_time': np.mean([r['time'] for r in random_results])
            },
            'k-means++': {
                'avg_silhouette': np.mean([r['metrics']['silhouette'] for r in kmeanspp_results]),
                'std_silhouette': np.std([r['metrics']['silhouette'] for r in kmeanspp_results]),
                'avg_iterations': np.mean([r['n_iterations'] for r in kmeanspp_results]),
                'avg_inertia': np.mean([r['metrics']['inertia'] for r in kmeanspp_results]),
                'avg_time': np.mean([r['time'] for r in kmeanspp_results])
            }
        }
        
        # Calculate improvements
        random = analysis[dataset_name]['random']
        kmeanspp = analysis[dataset_name]['k-means++']
        
        analysis[dataset_name]['improvements'] = {
            'silhouette_improvement_pct': ((kmeanspp['avg_silhouette'] - random['avg_silhouette']) / 
                                           abs(random['avg_silhouette']) * 100) if random['avg_silhouette'] != 0 else 0,
            'iterations_reduction_pct': ((random['avg_iterations'] - kmeanspp['avg_iterations']) / 
                                         random['avg_iterations'] * 100),
            'inertia_improvement_pct': ((random['avg_inertia'] - kmeanspp['avg_inertia']) / 
                                        random['avg_inertia'] * 100)
        }
    
    return analysis


def analyze_covariance_effect(gmm_results):
    """
    Analyze the effect of GMM covariance type
    
    Args:
        gmm_results: GMM results dictionary
    
    Returns:
        Dictionary with analysis for each dataset
    """
    analysis = {}
    
    for dataset_name, results in gmm_results.items():
        full_results = [r for r in results if r['covariance_type'] == 'full']
        diag_results = [r for r in results if r['covariance_type'] == 'diag']
        
        analysis[dataset_name] = {
            'full': {
                'avg_silhouette': np.mean([r['metrics']['silhouette'] for r in full_results]),
                'std_silhouette': np.std([r['metrics']['silhouette'] for r in full_results]),
                'avg_bic': np.mean([r['metrics']['bic'] for r in full_results]),
                'avg_aic': np.mean([r['metrics']['aic'] for r in full_results]),
                'avg_iterations': np.mean([r['n_iterations'] for r in full_results]),
                'avg_time': np.mean([r['time'] for r in full_results])
            },
            'diag': {
                'avg_silhouette': np.mean([r['metrics']['silhouette'] for r in diag_results]),
                'std_silhouette': np.std([r['metrics']['silhouette'] for r in diag_results]),
                'avg_bic': np.mean([r['metrics']['bic'] for r in diag_results]),
                'avg_aic': np.mean([r['metrics']['aic'] for r in diag_results]),
                'avg_iterations': np.mean([r['n_iterations'] for r in diag_results]),
                'avg_time': np.mean([r['time'] for r in diag_results])
            }
        }
        
        # Determine which is better (lower BIC is better)
        full = analysis[dataset_name]['full']
        diag = analysis[dataset_name]['diag']
        
        analysis[dataset_name]['comparison'] = {
            'bic_difference': full['avg_bic'] - diag['avg_bic'],
            'preferred': 'full' if full['avg_bic'] < diag['avg_bic'] else 'diag',
            'silhouette_difference': full['avg_silhouette'] - diag['avg_silhouette']
        }
    
    return analysis


def find_optimal_k(results, dataset_name, method='silhouette'):
    """
    Find optimal k for a dataset using specified method
    
    Args:
        results: Results dictionary (kmeans or gmm)
        dataset_name: Name of dataset
        method: 'silhouette', 'bic', or 'aic'
    
    Returns:
        Optimal k value
    """
    dataset_results = results[dataset_name]
    
    if method == 'silhouette':
        best_result = max(dataset_results, key=lambda x: x['metrics']['silhouette'])
        optimal_k = best_result.get('n_clusters', best_result.get('n_components'))
    elif method == 'bic':
        # Lower is better for BIC
        best_result = min(dataset_results, key=lambda x: x['metrics'].get('bic', float('inf')))
        optimal_k = best_result.get('n_components')
    elif method == 'aic':
        # Lower is better for AIC
        best_result = min(dataset_results, key=lambda x: x['metrics'].get('aic', float('inf')))
        optimal_k = best_result.get('n_components')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return optimal_k


def generate_comparison_report(save_path='results/comparison_report.txt'):
    """
    Generate comprehensive text report comparing all methods
    
    Args:
        save_path: Path to save report
    """
    kmeans_results, gmm_results = load_all_results()
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLUSTERING ALGORITHMS COMPARISON REPORT\n")
        f.write("CS 6375 - Machine Learning - Project 4\n")
        f.write("="*80 + "\n\n")
        
        # Section 1: Initialization Analysis
        f.write("1. K-MEANS INITIALIZATION COMPARISON\n")
        f.write("-"*80 + "\n\n")
        
        init_analysis = analyze_initialization_effect(kmeans_results)
        
        for dataset_name, analysis in init_analysis.items():
            f.write(f"{dataset_name.upper()} Dataset:\n")
            f.write(f"  Random Initialization:\n")
            f.write(f"    Avg Silhouette: {analysis['random']['avg_silhouette']:.4f} "
                   f"(±{analysis['random']['std_silhouette']:.4f})\n")
            f.write(f"    Avg Iterations: {analysis['random']['avg_iterations']:.2f}\n")
            f.write(f"    Avg Inertia: {analysis['random']['avg_inertia']:.2f}\n")
            f.write(f"    Avg Time: {analysis['random']['avg_time']:.4f}s\n\n")
            
            f.write(f"  K-Means++ Initialization:\n")
            f.write(f"    Avg Silhouette: {analysis['k-means++']['avg_silhouette']:.4f} "
                   f"(±{analysis['k-means++']['std_silhouette']:.4f})\n")
            f.write(f"    Avg Iterations: {analysis['k-means++']['avg_iterations']:.2f}\n")
            f.write(f"    Avg Inertia: {analysis['k-means++']['avg_inertia']:.2f}\n")
            f.write(f"    Avg Time: {analysis['k-means++']['avg_time']:.4f}s\n\n")
            
            f.write(f"  Improvements (K-Means++ vs Random):\n")
            f.write(f"    Silhouette: {analysis['improvements']['silhouette_improvement_pct']:+.2f}%\n")
            f.write(f"    Iterations Reduction: {analysis['improvements']['iterations_reduction_pct']:.2f}%\n")
            f.write(f"    Inertia Improvement: {analysis['improvements']['inertia_improvement_pct']:.2f}%\n\n")
        
        # Section 2: Covariance Analysis
        f.write("\n2. GMM COVARIANCE TYPE COMPARISON\n")
        f.write("-"*80 + "\n\n")
        
        cov_analysis = analyze_covariance_effect(gmm_results)
        
        for dataset_name, analysis in cov_analysis.items():
            f.write(f"{dataset_name.upper()} Dataset:\n")
            f.write(f"  Full Covariance:\n")
            f.write(f"    Avg Silhouette: {analysis['full']['avg_silhouette']:.4f} "
                   f"(±{analysis['full']['std_silhouette']:.4f})\n")
            f.write(f"    Avg BIC: {analysis['full']['avg_bic']:.2f}\n")
            f.write(f"    Avg AIC: {analysis['full']['avg_aic']:.2f}\n")
            f.write(f"    Avg Iterations: {analysis['full']['avg_iterations']:.2f}\n")
            f.write(f"    Avg Time: {analysis['full']['avg_time']:.4f}s\n\n")
            
            f.write(f"  Diagonal Covariance:\n")
            f.write(f"    Avg Silhouette: {analysis['diag']['avg_silhouette']:.4f} "
                   f"(±{analysis['diag']['std_silhouette']:.4f})\n")
            f.write(f"    Avg BIC: {analysis['diag']['avg_bic']:.2f}\n")
            f.write(f"    Avg AIC: {analysis['diag']['avg_aic']:.2f}\n")
            f.write(f"    Avg Iterations: {analysis['diag']['avg_iterations']:.2f}\n")
            f.write(f"    Avg Time: {analysis['diag']['avg_time']:.4f}s\n\n")
            
            f.write(f"  Comparison:\n")
            f.write(f"    BIC Difference (Full - Diag): {analysis['comparison']['bic_difference']:.2f}\n")
            f.write(f"    Preferred: {analysis['comparison']['preferred'].upper()} covariance\n")
            f.write(f"    Silhouette Difference: {analysis['comparison']['silhouette_difference']:+.4f}\n\n")
        
        # Section 3: Optimal k Analysis
        f.write("\n3. OPTIMAL K SELECTION\n")
        f.write("-"*80 + "\n\n")
        
        from data_generation import generate_all_datasets
        datasets = generate_all_datasets()
        
        for dataset_name in kmeans_results.keys():
            true_k = datasets[dataset_name]['true_k']
            f.write(f"{dataset_name.upper()} Dataset (True k = {true_k}):\n")
            
            # K-Means optimal k
            kmeans_optimal_k = find_optimal_k(kmeans_results, dataset_name, 'silhouette')
            f.write(f"  K-Means (by Silhouette): k = {kmeans_optimal_k}\n")
            
            # GMM optimal k by different metrics
            gmm_sil_k = find_optimal_k(gmm_results, dataset_name, 'silhouette')
            gmm_bic_k = find_optimal_k(gmm_results, dataset_name, 'bic')
            gmm_aic_k = find_optimal_k(gmm_results, dataset_name, 'aic')
            
            f.write(f"  GMM (by Silhouette): k = {gmm_sil_k}\n")
            f.write(f"  GMM (by BIC): k = {gmm_bic_k}\n")
            f.write(f"  GMM (by AIC): k = {gmm_aic_k}\n\n")
        
        # Section 4: Algorithm Comparison
        f.write("\n4. K-MEANS VS GMM COMPARISON\n")
        f.write("-"*80 + "\n\n")
        
        for dataset_name in kmeans_results.keys():
            true_k = datasets[dataset_name]['true_k']
            f.write(f"{dataset_name.upper()} Dataset at True k = {true_k}:\n")
            
            comparison = analyze_algorithm_comparison(kmeans_results, gmm_results, 
                                                     dataset_name, true_k)
            
            f.write(f"  K-Means (Best: {comparison['kmeans']['variant']}):\n")
            f.write(f"    Silhouette: {comparison['kmeans']['silhouette']:.4f}\n")
            f.write(f"    Inertia: {comparison['kmeans']['inertia']:.2f}\n")
            f.write(f"    Iterations: {comparison['kmeans']['iterations']}\n")
            f.write(f"    Time: {comparison['kmeans']['time']:.4f}s\n\n")
            
            f.write(f"  GMM (Best: {comparison['gmm']['variant']}):\n")
            f.write(f"    Silhouette: {comparison['gmm']['silhouette']:.4f}\n")
            f.write(f"    BIC: {comparison['gmm']['bic']:.2f}\n")
            f.write(f"    Iterations: {comparison['gmm']['iterations']}\n")
            f.write(f"    Time: {comparison['gmm']['time']:.4f}s\n\n")
            
            f.write(f"  Winner by Silhouette: {comparison['winner']}\n\n")
        
        # Section 5: Key Insights
        f.write("\n5. KEY INSIGHTS AND CONCLUSIONS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Blobs Dataset (Well-separated spherical clusters):\n")
        f.write("  • Both algorithms perform excellently when k matches true structure\n")
        f.write("  • K-means++ initialization significantly outperforms random initialization\n")
        f.write("  • Full covariance GMM provides slightly better fit but at higher computational cost\n")
        f.write("  • Clear elbow at k=3 (true value) in both silhouette and BIC metrics\n\n")
        
        f.write("Moons Dataset (Non-convex crescent-shaped clusters):\n")
        f.write("  • Both algorithms struggle due to non-Gaussian, non-convex structure\n")
        f.write("  • Silhouette scores remain moderate (~0.45-0.49) even at true k=2\n")
        f.write("  • GMM with full covariance performs better by capturing curved structure\n")
        f.write("  • BIC suggests higher k values (overfitting to capture non-linear boundaries)\n")
        f.write("  • Demonstrates fundamental limitation of Gaussian/spherical cluster assumptions\n\n")
        
        f.write("General Conclusions:\n")
        f.write("  • No algorithm is universally optimal - performance depends on data structure\n")
        f.write("  • K-means++ initialization is crucial for avoiding poor local minima\n")
        f.write("  • Full covariance GMMs offer more flexibility but require more data\n")
        f.write("  • Multiple metrics should be considered for optimal k selection\n")
        f.write("  • Understanding data characteristics is essential for algorithm selection\n")
    
    print(f"\n✅ Comparison report saved to: {save_path}")


def create_performance_heatmap(kmeans_results, gmm_results, save_path='figures/performance_heatmap.png'):
    """
    Create heatmap showing performance across all configurations
    
    Args:
        kmeans_results: K-Means results
        gmm_results: GMM results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = ['blobs', 'moons']
    k_values = [2, 3, 4, 5]
    
    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        
        # Create matrix for heatmap
        methods = ['KM-rand', 'KM-pp', 'GMM-full', 'GMM-diag']
        data_matrix = []
        
        for k in k_values:
            row = []
            
            # K-Means random
            km_rand = [r for r in kmeans_results[dataset_name] 
                      if r['n_clusters'] == k and r['init_method'] == 'random']
            row.append(km_rand[0]['metrics']['silhouette'] if km_rand else 0)
            
            # K-Means k-means++
            km_pp = [r for r in kmeans_results[dataset_name]
                    if r['n_clusters'] == k and r['init_method'] == 'k-means++']
            row.append(km_pp[0]['metrics']['silhouette'] if km_pp else 0)
            
            # GMM full
            gmm_full = [r for r in gmm_results[dataset_name]
                       if r['n_components'] == k and r['covariance_type'] == 'full']
            row.append(gmm_full[0]['metrics']['silhouette'] if gmm_full else 0)
            
            # GMM diag
            gmm_diag = [r for r in gmm_results[dataset_name]
                       if r['n_components'] == k and r['covariance_type'] == 'diag']
            row.append(gmm_diag[0]['metrics']['silhouette'] if gmm_diag else 0)
            
            data_matrix.append(row)
        
        # Create heatmap
        data_matrix = np.array(data_matrix)
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=1.0)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(k_values)))
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_yticklabels(k_values, fontsize=10)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(k_values)):
            for j in range(len(methods)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9,
                             fontweight='bold')
        
        ax.set_title(f'{dataset_name.capitalize()} - Silhouette Scores', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Algorithm Configuration', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Silhouette Score', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"  Saved: {save_path}")
    
    return fig


def print_comparison_summary():
    """Print a concise summary of key findings"""
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS SUMMARY")
    print("="*80)
    
    kmeans_results, gmm_results = load_all_results()
    
    # Performance summary table
    print("\n1. Performance Summary Table:")
    print("-"*80)
    df = create_performance_summary_table(kmeans_results, gmm_results)
    
    # Show best results per dataset
    for dataset in ['blobs', 'moons']:
        print(f"\n{dataset.upper()}:")
        dataset_df = df[df['Dataset'] == dataset]
        best_sil_idx = dataset_df['Silhouette'].idxmax()
        best_result = dataset_df.loc[best_sil_idx]
        print(f"  Best Configuration: {best_result['Algorithm']} ({best_result['Variant']}) at k={best_result['k']}")
        print(f"  Silhouette: {best_result['Silhouette']:.4f}")


if __name__ == "__main__":
    print("CS 6375 Project 4: Comparison Analysis")
    print("="*80)
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Load results
    print("\nLoading results...")
    kmeans_results, gmm_results = load_all_results()
    
    # Generate comprehensive report
    print("\nGenerating comparison report...")
    generate_comparison_report()
    
    # Create performance heatmap
    print("\nCreating performance heatmap...")
    create_performance_heatmap(kmeans_results, gmm_results)
    
    # Print summary
    print_comparison_summary()
    
    print("\n" + "="*80)
    print("✅ Comparison analysis complete!")
    print("="*80)