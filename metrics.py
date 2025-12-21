"""
Metrics Module
Calculates clustering evaluation metrics
"""

import numpy as np
from sklearn.metrics import silhouette_score

def calculate_silhouette_score(X, labels):
    """
    Calculate silhouette score for clustering quality
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
    
    Returns:
        float: Silhouette score (-1 to 1, higher is better)
               Returns -1 for invalid clustering (< 2 clusters)
    
    Notes:
        - Values > 0.5 indicate good clustering
        - Values < 0.25 indicate poor clustering
        - Values near 0 indicate overlapping clusters
    """
    # Check for valid clustering
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Warning: Less than 2 clusters found. Returning -1.")
        return -1.0
    
    # Check for empty clusters or all points in one cluster
    if len(unique_labels) == len(X):
        print("Warning: Each point in its own cluster. Returning -1.")
        return -1.0
    
    try:
        score = silhouette_score(X, labels)
        return score
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return -1.0

def calculate_inertia(X, labels, centroids):
    """
    Calculate inertia (within-cluster sum of squares) for K-Means
    Used for the elbow method to determine optimal k
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        centroids: Cluster centers (n_clusters, n_features)
    
    Returns:
        float: Inertia value (lower is better)
               Sum of squared distances from each point to its centroid
    
    Formula:
        Inertia = Σ_k Σ_{x_i ∈ C_k} ||x_i - μ_k||²
    """
    inertia = 0.0
    n_clusters = len(centroids)
    
    for k in range(n_clusters):
        # Get all points assigned to cluster k
        cluster_mask = (labels == k)
        cluster_points = X[cluster_mask]
        
        if len(cluster_points) == 0:
            continue  # Skip empty clusters
        
        # Calculate squared distances to centroid
        squared_distances = np.sum((cluster_points - centroids[k]) ** 2, axis=1)
        inertia += np.sum(squared_distances)
    
    return inertia

def calculate_distortion(X, labels, centroids):
    """
    Calculate distortion (average squared distance to centroids)
    Alternative to inertia, normalized by number of points
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        centroids: Cluster centers (n_clusters, n_features)
    
    Returns:
        float: Distortion value (lower is better)
    """
    inertia = calculate_inertia(X, labels, centroids)
    return inertia / len(X)

def calculate_kmeans_metrics(X, labels, centroids, n_iterations):
    """
    Calculate all metrics for K-Means clustering result
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        centroids: Cluster centers (n_clusters, n_features)
        n_iterations: Number of iterations to convergence
    
    Returns:
        dict: Dictionary containing:
            - silhouette: Silhouette score
            - inertia: Within-cluster sum of squares
            - distortion: Average squared distance
            - n_iterations: Iterations to convergence
    """
    metrics = {
        'silhouette': calculate_silhouette_score(X, labels),
        'inertia': calculate_inertia(X, labels, centroids),
        'distortion': calculate_distortion(X, labels, centroids),
        'n_iterations': n_iterations
    }
    
    return metrics

def calculate_gmm_metrics(X, labels, log_likelihood, bic, aic, n_iterations):
    """
    Calculate all metrics for GMM clustering result
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster assignments (hard assignments from GMM)
        log_likelihood: Log-likelihood of the data under the model
        bic: Bayesian Information Criterion
        aic: Akaike Information Criterion
        n_iterations: Number of EM iterations to convergence
    
    Returns:
        dict: Dictionary containing:
            - silhouette: Silhouette score
            - log_likelihood: Log-likelihood of data
            - bic: Bayesian Information Criterion (lower is better)
            - aic: Akaike Information Criterion (lower is better)
            - n_iterations: Iterations to convergence
    
    Notes:
        BIC = -2 * log_likelihood + n_params * log(n_samples)
        AIC = -2 * log_likelihood + 2 * n_params
    """
    metrics = {
        'silhouette': calculate_silhouette_score(X, labels),
        'log_likelihood': log_likelihood,
        'bic': bic,
        'aic': aic,
        'n_iterations': n_iterations
    }
    
    return metrics

def compute_bic(log_likelihood, n_params, n_samples):
    """
    Compute Bayesian Information Criterion
    
    Args:
        log_likelihood: Log-likelihood of the data under the model
        n_params: Number of free parameters in the model
        n_samples: Number of data samples
    
    Returns:
        float: BIC value (lower is better)
    
    Formula:
        BIC = -2 * log_likelihood + n_params * log(n_samples)
    """
    return -2 * log_likelihood + n_params * np.log(n_samples)

def compute_aic(log_likelihood, n_params):
    """
    Compute Akaike Information Criterion
    
    Args:
        log_likelihood: Log-likelihood of the data under the model
        n_params: Number of free parameters in the model
    
    Returns:
        float: AIC value (lower is better)
    
    Formula:
        AIC = -2 * log_likelihood + 2 * n_params
    """
    return -2 * log_likelihood + 2 * n_params

def count_gmm_parameters(n_components, n_features, covariance_type='full'):
    """
    Count the number of free parameters in a GMM
    
    Args:
        n_components: Number of Gaussian components (k)
        n_features: Number of features (d)
        covariance_type: 'full' or 'diagonal'
    
    Returns:
        int: Number of free parameters
    
    Parameter breakdown:
        - Means: k * d
        - Covariances: 
            * Full: k * d * (d + 1) / 2  (symmetric matrices)
            * Diagonal: k * d  (only diagonal elements)
        - Mixture weights: k - 1  (sum to 1 constraint)
    """
    k, d = n_components, n_features
    
    # Means
    n_means = k * d
    
    # Covariances
    if covariance_type == 'full':
        # Each component has a d×d symmetric covariance matrix
        # Number of unique elements = d(d+1)/2
        n_covs = k * d * (d + 1) // 2
    elif covariance_type == 'diagonal':
        # Each component has d diagonal elements
        n_covs = k * d
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")
    
    # Mixture weights (k weights that sum to 1 = k-1 free parameters)
    n_weights = k - 1
    
    total_params = n_means + n_covs + n_weights
    
    return total_params

def get_centroids_from_labels(X, labels, n_clusters):
    """
    Calculate cluster centroids from data and hard cluster assignments
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        n_clusters: Number of clusters
    
    Returns:
        ndarray: Centroids array (n_clusters, n_features)
    """
    n_features = X.shape[1]
    centroids = np.zeros((n_clusters, n_features))
    
    for k in range(n_clusters):
        cluster_mask = (labels == k)
        cluster_points = X[cluster_mask]
        
        if len(cluster_points) > 0:
            centroids[k] = cluster_points.mean(axis=0)
        else:
            # Empty cluster - assign random point or keep at origin
            print(f"Warning: Cluster {k} is empty")
            centroids[k] = X[np.random.randint(len(X))]
    
    return centroids

# Utility function for printing metrics nicely
def print_metrics(metrics, algorithm='kmeans', indent=2):
    """
    Pretty print metrics dictionary
    
    Args:
        metrics: Dictionary of metric values
        algorithm: 'kmeans' or 'gmm'
        indent: Number of spaces for indentation
    """
    prefix = ' ' * indent
    
    if algorithm == 'kmeans':
        print(f"{prefix}Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"{prefix}Inertia: {metrics['inertia']:.2f}")
        print(f"{prefix}Iterations: {metrics['n_iterations']}")
    
    elif algorithm == 'gmm':
        print(f"{prefix}Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"{prefix}BIC: {metrics['bic']:.2f}")
        print(f"{prefix}Log-Likelihood: {metrics['log_likelihood']:.2f}")
        print(f"{prefix}Iterations: {metrics['n_iterations']}")

if __name__ == "__main__":
    print("Testing Metrics Module")
    print("=" * 60)
    
    # Test with dummy data
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    # Generate test data
    X, y_true = make_blobs(n_samples=150, centers=3, random_state=42)
    
    # Test K-Means metrics
    print("\nTesting K-Means Metrics:")
    print("-" * 60)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    metrics = calculate_kmeans_metrics(
        X, 
        labels, 
        kmeans.cluster_centers_,
        n_iterations=kmeans.n_iter_
    )
    print_metrics(metrics, algorithm='kmeans')
    
    # Test GMM parameter counting
    print("\n\nTesting GMM Parameter Counting:")
    print("-" * 60)
    for cov_type in ['full', 'diagonal']:
        for k in [2, 3, 4, 5]:
            n_params = count_gmm_parameters(k, n_features=2, covariance_type=cov_type)
            print(f"k={k}, covariance={cov_type:8s}: {n_params:3d} parameters")
    
    # Test BIC/AIC calculation
    print("\n\nTesting BIC/AIC Calculation:")
    print("-" * 60)
    log_likelihood = -500.0
    n_params = 11  # Example
    n_samples = 150
    
    bic = compute_bic(log_likelihood, n_params, n_samples)
    aic = compute_aic(log_likelihood, n_params)
    
    print(f"Log-Likelihood: {log_likelihood:.2f}")
    print(f"N Parameters: {n_params}")
    print(f"N Samples: {n_samples}")
    print(f"BIC: {bic:.2f}")
    print(f"AIC: {aic:.2f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests complete!")