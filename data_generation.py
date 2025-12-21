"""
Data Generation Module
Generates synthetic datasets for clustering experiments
"""
import numpy as np
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import os

def generate_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.6, random_state=42):
    """
    Generate well-separated spherical clusters (ideal for clustering)
    
    Args:
        n_samples: Total number of samples (default: 150 to match report)
        n_features: Number of features (dimensions)
        centers: Number of cluster centers
        cluster_std: Standard deviation of clusters (0.6 per report)
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,)
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y

def generate_moons(n_samples=150, noise=0.05, random_state=42):
    """
    Generate crescent-shaped non-convex clusters (challenging for clustering)
    
    Args:
        n_samples: Total number of samples (default: 150 to match report)
        noise: Standard deviation of Gaussian noise (0.05 per report)
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix (n_samples, 2)
        y: True labels (n_samples,)
    """
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    return X, y

def visualize_dataset(X, y, title="Dataset Visualization", save_path=None):
    """
    Visualize the generated dataset
    
    Args:
        X: Feature matrix
        y: True labels
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib figure object
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='k')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.colorbar(scatter, label='True Label')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return plt.gcf()

def generate_all_datasets(n_samples=150, random_state=42):
    """
    Generate all datasets needed for experiments with metadata
    
    Args:
        n_samples: Number of samples per dataset
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing datasets with metadata
    """
    datasets = {}
    
    # Generate blobs dataset
    X_blobs, y_blobs = generate_blobs(
        n_samples=n_samples, 
        centers=3, 
        cluster_std=0.6,
        random_state=random_state
    )
    datasets['blobs'] = {
        'X': X_blobs,
        'y': y_blobs,
        'true_k': 3,
        'description': 'Well-separated spherical clusters',
        'n_samples': n_samples,
        'n_features': 2
    }
    
    # Generate moons dataset
    X_moons, y_moons = generate_moons(
        n_samples=n_samples,
        noise=0.05,
        random_state=random_state
    )
    datasets['moons'] = {
        'X': X_moons,
        'y': y_moons,
        'true_k': 2,
        'description': 'Crescent-shaped non-convex clusters',
        'n_samples': n_samples,
        'n_features': 2
    }
    
    print("="*60)
    print("Generated Datasets Summary".center(60))
    print("="*60)
    for name, data in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {data['X'].shape}")
        print(f"  True k: {data['true_k']}")
        print(f"  Description: {data['description']}")
    print("="*60)
    
    return datasets

def visualize_raw_data(datasets, save_dir='figures/'):
    """
    Create and save visualizations for all raw datasets
    
    Args:
        datasets: Dictionary of datasets from generate_all_datasets()
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nGenerating dataset visualizations...")
    for name, data in datasets.items():
        save_path = os.path.join(save_dir, f'{name}_raw_dataset.png')
        title = f"{name.capitalize()} Dataset (n={data['n_samples']})"
        
        fig = visualize_dataset(data['X'], data['y'], title=title, save_path=save_path)
        plt.close(fig)  # Close to avoid memory issues
    
    print(f"✓ All visualizations saved to {save_dir}")

if __name__ == "__main__":
    print("CS 6375 Project 4: Data Generation")
    print("-" * 60)
    
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Generate all datasets
    datasets = generate_all_datasets(n_samples=150, random_state=42)
    
    # Visualize and save
    visualize_raw_data(datasets, save_dir='figures/')
    
    # Optional: Display one example
    print("\nSaving sample visualization...")
    fig = visualize_dataset(
        datasets['blobs']['X'], 
        datasets['blobs']['y'],
        title="Blobs Dataset Example"
    )
    
    
    print("\n✓ Data generation complete!")