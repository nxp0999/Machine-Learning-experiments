"""
Visualizations Module
Creates all plots for clustering analysis
Enhanced with Seaborn for professional aesthetics
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("notebook", font_scale=1.1)

# Additional matplotlib settings
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)


def plot_clustering_result(X, labels, centroids=None, title="Clustering Result", 
                           true_labels=None, save_path=None):
    """
    Plot clustering result with cluster assignments
    
    Args:
        X: Feature matrix (n_samples, 2)
        labels: Predicted cluster assignments
        centroids: Cluster centers (optional, for K-Means)
        title: Plot title
        true_labels: Ground truth labels (optional)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, 
                             figsize=(14, 5) if true_labels is not None else (8, 6))
    
    # Use seaborn color palette
    n_clusters = len(np.unique(labels))
    colors = sns.color_palette("husl", n_clusters)
    
    if true_labels is not None:
        # Plot predicted labels
        ax1 = axes[0]
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                              s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        if centroids is not None:
            ax1.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='X', s=300, 
                       edgecolors='black', linewidths=2,
                       label='Centroids', zorder=10)
            ax1.legend()
        ax1.set_title(f'{title} - Predicted', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Feature 1', fontsize=11)
        ax1.set_ylabel('Feature 2', fontsize=11)
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        sns.despine(ax=ax1)
        
        # Plot true labels
        ax2 = axes[1]
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis',
                              s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax2.set_title(f'{title} - Ground Truth', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Feature 1', fontsize=11)
        ax2.set_ylabel('Feature 2', fontsize=11)
        plt.colorbar(scatter2, ax=ax2, label='True Cluster')
        sns.despine(ax=ax2)
    else:
        ax = axes if true_labels is not None else plt.gca()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                            s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1],
                      c='red', marker='X', s=300,
                      edgecolors='black', linewidths=2,
                      label='Centroids', zorder=10)
            ax.legend()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        sns.despine(ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_elbow_curve(results, dataset_name, metric='inertia', 
                    title_suffix="", save_path=None):
    """
    Plot elbow curve for K-Means using seaborn styling
    
    Args:
        results: List of result dictionaries for a dataset
        dataset_name: Name of dataset
        metric: Metric to plot ('inertia' or 'distortion')
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by initialization method
    init_methods = sorted(list(set([r['init_method'] for r in results])))
    
    # Use seaborn color palette
    palette = {'random': sns.color_palette("Set1")[0], 
               'k-means++': sns.color_palette("Set1")[1]}
    markers = {'random': 'o', 'k-means++': 's'}
    
    for init in init_methods:
        init_results = [r for r in results if r['init_method'] == init]
        init_results.sort(key=lambda x: x['n_clusters'])
        
        k_values = [r['n_clusters'] for r in init_results]
        metric_values = [r['metrics'][metric] for r in init_results]
        
        ax.plot(k_values, metric_values, 
               marker=markers[init], linewidth=2.5, markersize=10,
               label=init, color=palette[init])
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
    ax.set_title(f'Elbow Method - {dataset_name.capitalize()} Dataset{title_suffix}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(k_values)
    sns.despine()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_silhouette_comparison(kmeans_results, gmm_results, dataset_name, save_path=None):
    """
    Compare silhouette scores between K-Means and GMM using seaborn
    
    Args:
        kmeans_results: K-Means results for dataset
        gmm_results: GMM results for dataset
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    k_values = sorted(list(set([r['n_clusters'] for r in kmeans_results])))
    
    # K-Means data
    kmeans_random = []
    kmeans_pp = []
    for k in k_values:
        random_res = [r for r in kmeans_results if r['n_clusters'] == k and r['init_method'] == 'random']
        pp_res = [r for r in kmeans_results if r['n_clusters'] == k and r['init_method'] == 'k-means++']
        kmeans_random.append(random_res[0]['metrics']['silhouette'] if random_res else 0)
        kmeans_pp.append(pp_res[0]['metrics']['silhouette'] if pp_res else 0)
    
    # GMM data
    gmm_full = []
    gmm_diag = []
    for k in k_values:
        full_res = [r for r in gmm_results if r['n_components'] == k and r['covariance_type'] == 'full']
        diag_res = [r for r in gmm_results if r['n_components'] == k and r['covariance_type'] == 'diag']
        gmm_full.append(full_res[0]['metrics']['silhouette'] if full_res else 0)
        gmm_diag.append(diag_res[0]['metrics']['silhouette'] if diag_res else 0)
    
    # Plot using seaborn color palette
    x = np.arange(len(k_values))
    width = 0.2
    colors = sns.color_palette("Set2", 4)
    
    ax.bar(x - 1.5*width, kmeans_random, width, label='K-Means (random)', color=colors[0], alpha=0.8)
    ax.bar(x - 0.5*width, kmeans_pp, width, label='K-Means (k-means++)', color=colors[1], alpha=0.8)
    ax.bar(x + 0.5*width, gmm_full, width, label='GMM (full)', color=colors[2], alpha=0.8)
    ax.bar(x + 1.5*width, gmm_diag, width, label='GMM (diag)', color=colors[3], alpha=0.8)
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Silhouette Score Comparison - {dataset_name.capitalize()} Dataset', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(fontsize=10, loc='best')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    sns.despine()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_bic_comparison(gmm_results, dataset_name, save_path=None):
    """
    Compare BIC scores for GMM with different covariance types using seaborn
    
    Args:
        gmm_results: GMM results for dataset
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    k_values = sorted(list(set([r['n_components'] for r in gmm_results])))
    
    # Separate by covariance type
    full_results = [r for r in gmm_results if r['covariance_type'] == 'full']
    diag_results = [r for r in gmm_results if r['covariance_type'] == 'diag']
    
    full_results.sort(key=lambda x: x['n_components'])
    diag_results.sort(key=lambda x: x['n_components'])
    
    full_bic = [r['metrics']['bic'] for r in full_results]
    full_aic = [r['metrics']['aic'] for r in full_results]
    diag_bic = [r['metrics']['bic'] for r in diag_results]
    diag_aic = [r['metrics']['aic'] for r in diag_results]
    
    # Use seaborn color palette
    colors = sns.color_palette("Set1", 2)
    
    # Plot BIC
    ax1.plot(k_values, full_bic, marker='o', linewidth=2.5, markersize=10, 
            label='Full Covariance', color=colors[0])
    ax1.plot(k_values, diag_bic, marker='s', linewidth=2.5, markersize=10,
            label='Diagonal Covariance', color=colors[1])
    ax1.set_xlabel('Number of Components (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BIC (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title(f'BIC - {dataset_name.capitalize()}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xticks(k_values)
    
    # Mark minimum
    min_bic_full_idx = np.argmin(full_bic)
    min_bic_diag_idx = np.argmin(diag_bic)
    min_bic_idx = min_bic_full_idx if full_bic[min_bic_full_idx] < diag_bic[min_bic_diag_idx] else min_bic_diag_idx
    min_k = k_values[min_bic_idx]
    ax1.axvline(x=min_k, color='green', linestyle='--', alpha=0.5, linewidth=2)
    sns.despine(ax=ax1)
    
    # Plot AIC
    ax2.plot(k_values, full_aic, marker='o', linewidth=2.5, markersize=10,
            label='Full Covariance', color=colors[0])
    ax2.plot(k_values, diag_aic, marker='s', linewidth=2.5, markersize=10,
            label='Diagonal Covariance', color=colors[1])
    ax2.set_xlabel('Number of Components (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AIC (lower is better)', fontsize=12, fontweight='bold')
    ax2.set_title(f'AIC - {dataset_name.capitalize()}', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xticks(k_values)
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def create_performance_heatmap(kmeans_results, gmm_results, save_path=None):
    """
    Create a heatmap showing silhouette scores across all configurations
    Uses seaborn's heatmap functionality for better visualization
    
    Args:
        kmeans_results: K-Means results dictionary
        gmm_results: GMM results dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, dataset_name in enumerate(['blobs', 'moons']):
        k_values = sorted(list(set([r['n_clusters'] for r in kmeans_results[dataset_name]])))
        
        # Create data matrix
        data = []
        labels = []
        
        # K-Means data
        for init in ['random', 'k-means++']:
            row = []
            for k in k_values:
                res = [r for r in kmeans_results[dataset_name] 
                      if r['n_clusters'] == k and r['init_method'] == init]
                row.append(res[0]['metrics']['silhouette'] if res else 0)
            data.append(row)
            labels.append(f'KM-{init}')
        
        # GMM data
        for cov in ['full', 'diag']:
            row = []
            for k in k_values:
                res = [r for r in gmm_results[dataset_name]
                      if r['n_components'] == k and r['covariance_type'] == cov]
                row.append(res[0]['metrics']['silhouette'] if res else 0)
            data.append(row)
            labels.append(f'GMM-{cov}')
        
        # Create heatmap using seaborn
        ax = axes[idx]
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   xticklabels=k_values, yticklabels=labels,
                   cbar_kws={'label': 'Silhouette Score'},
                   vmin=-0.2, vmax=1.0, center=0.5,
                   ax=ax, linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Algorithm Configuration', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name.capitalize()} - Silhouette Scores', 
                    fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def create_clustering_grid(kmeans_results, gmm_results, datasets, 
                           dataset_name, k_value, save_path=None):
    """
    Create a grid showing clustering results for different algorithms/configs
    
    Args:
        kmeans_results: K-Means results
        gmm_results: GMM results  
        datasets: Dictionary of datasets
        dataset_name: Name of dataset to visualize
        k_value: Specific k value to visualize
        save_path: Path to save figure
    """
    X = datasets[dataset_name]['X']
    y_true = datasets[dataset_name]['y']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{dataset_name.capitalize()} Dataset - k={k_value}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Find specific results
    kmeans_random = [r for r in kmeans_results[dataset_name] 
                     if r['n_clusters'] == k_value and r['init_method'] == 'random'][0]
    kmeans_pp = [r for r in kmeans_results[dataset_name]
                 if r['n_clusters'] == k_value and r['init_method'] == 'k-means++'][0]
    gmm_full = [r for r in gmm_results[dataset_name]
                if r['n_components'] == k_value and r['covariance_type'] == 'full'][0]
    gmm_diag = [r for r in gmm_results[dataset_name]
                if r['n_components'] == k_value and r['covariance_type'] == 'diag'][0]
    
    # Plot ground truth
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_title('Ground Truth', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax)
    sns.despine(ax=ax)
    
    # Plot K-Means random
    ax = axes[0, 1]
    labels = np.array(kmeans_random['labels'])
    centroids = np.array(kmeans_random['centroids'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200,
              edgecolors='black', linewidths=2, zorder=10)
    sil = kmeans_random['metrics']['silhouette']
    ax.set_title(f'K-Means (random)\nSil={sil:.3f}', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax)
    sns.despine(ax=ax)
    
    # Plot K-Means k-means++
    ax = axes[0, 2]
    labels = np.array(kmeans_pp['labels'])
    centroids = np.array(kmeans_pp['centroids'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200,
              edgecolors='black', linewidths=2, zorder=10)
    sil = kmeans_pp['metrics']['silhouette']
    ax.set_title(f'K-Means (k-means++)\nSil={sil:.3f}', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax)
    sns.despine(ax=ax)
    
    # Plot GMM full
    ax = axes[1, 0]
    labels = np.array(gmm_full['labels'])
    means = np.array(gmm_full['means'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.scatter(means[:, 0], means[:, 1], c='red', marker='X', s=200,
              edgecolors='black', linewidths=2, zorder=10)
    sil = gmm_full['metrics']['silhouette']
    bic = gmm_full['metrics']['bic']
    ax.set_title(f'GMM (full)\nSil={sil:.3f}, BIC={bic:.1f}', 
                fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax)
    sns.despine(ax=ax)
    
    # Plot GMM diagonal
    ax = axes[1, 1]
    labels = np.array(gmm_diag['labels'])
    means = np.array(gmm_diag['means'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.scatter(means[:, 0], means[:, 1], c='red', marker='X', s=200,
              edgecolors='black', linewidths=2, zorder=10)
    sil = gmm_diag['metrics']['silhouette']
    bic = gmm_diag['metrics']['bic']
    ax.set_title(f'GMM (diag)\nSil={sil:.3f}, BIC={bic:.1f}',
                fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax)
    sns.despine(ax=ax)
    
    # Add summary in last subplot
    axes[1, 2].axis('off')
    summary_text = f"""
    Dataset: {dataset_name}
    k = {k_value}
    True k = {datasets[dataset_name]['true_k']}
    
    Best Silhouette:
    {max(
        kmeans_random['metrics']['silhouette'],
        kmeans_pp['metrics']['silhouette'],
        gmm_full['metrics']['silhouette'],
        gmm_diag['metrics']['silhouette']
    ):.3f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, 
                   verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def create_all_visualizations():
    """Generate all visualizations from saved results"""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS (Enhanced with Seaborn)")
    print("="*60)
    
    # Load results
    print("\nLoading results...")
    with open('results/kmeans_results.json', 'r') as f:
        kmeans_results = json.load(f)
    
    with open('results/gmm_results.json', 'r') as f:
        gmm_results = json.load(f)
    
    from data_generation import generate_all_datasets
    datasets = generate_all_datasets()
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Elbow curves for K-Means
    for dataset_name in ['blobs', 'moons']:
        plot_elbow_curve(
            kmeans_results[dataset_name],
            dataset_name,
            metric='inertia',
            save_path=f'figures/kmeans_elbow_{dataset_name}.png'
        )
    
    # Silhouette comparisons
    for dataset_name in ['blobs', 'moons']:
        plot_silhouette_comparison(
            kmeans_results[dataset_name],
            gmm_results[dataset_name],
            dataset_name,
            save_path=f'figures/silhouette_comparison_{dataset_name}.png'
        )
    
    # BIC comparisons for GMM
    for dataset_name in ['blobs', 'moons']:
        plot_bic_comparison(
            gmm_results[dataset_name],
            dataset_name,
            save_path=f'figures/gmm_bic_{dataset_name}.png'
        )
    
    # Clustering grids (k=3 for visual comparison)
    for dataset_name in ['blobs', 'moons']:
        true_k = datasets[dataset_name]['true_k']
        create_clustering_grid(
            kmeans_results,
            gmm_results,
            datasets,
            dataset_name,
            k_value=true_k,
            save_path=f'figures/clustering_grid_{dataset_name}_k{true_k}.png'
        )
    
    # Create performance heatmap using seaborn
    create_performance_heatmap(
        kmeans_results,
        gmm_results,
        save_path='figures/performance_heatmap.png'
    )
    
    print("\n" + "="*60)
    print("[SUCCESS] All visualizations created")
    print("="*60)
    
    plt.close('all')


if __name__ == "__main__":
    print("CS 6375 Project 4: Visualizations (Seaborn Enhanced)")
    print("="*60)
    
    create_all_visualizations()