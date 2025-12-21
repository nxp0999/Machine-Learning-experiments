# CS 6375 Project 4: K-Means Clustering and Gaussian Mixture Models

## Project Overview

This project provides a comprehensive comparative analysis of two fundamental unsupervised learning algorithms:
- **K-Means Clustering** (with random and k-means++ initialization)
- **Gaussian Mixture Models (GMM)** with Expectation-Maximization (with full and diagonal covariance)

Both algorithms are evaluated on synthetic 2D datasets with different geometric properties to understand their strengths, limitations, and appropriate use cases.

## Datasets

### 1. make_blobs
- **Characteristics:** Well-separated spherical clusters
- **Samples:** 150 points
- **True clusters:** 3
- **Purpose:** Tests performance on data matching algorithmic assumptions

### 2. make_moons
- **Characteristics:** Non-convex crescent-shaped clusters
- **Samples:** 150 points
- **True clusters:** 2
- **Purpose:** Tests robustness to violations of spherical/Gaussian assumptions

## Experiments Conducted

**Total Experiments:** 32

### K-Means (16 experiments)
- **Cluster values:** k ∈ {2, 3, 4, 5}
- **Initializations:** random vs. k-means++
- **Datasets:** make_blobs and make_moons
- **Metrics:** Silhouette score, inertia (ICSSD), iterations to convergence

### GMM (16 experiments)
- **Components:** k ∈ {2, 3, 4, 5}
- **Covariance types:** full (general ellipses) vs. diagonal (axis-aligned ellipses)
- **Datasets:** make_blobs and make_moons
- **Metrics:** Silhouette score, log-likelihood, BIC, AIC

## Key Findings

### Performance Summary

| Dataset    | Algorithm             | Best k | Silhouette Score |
|------------|-----------------------|--------|------------------|
| make_blobs | K-Means (k-means++)   | 3      | 0.907            |
| make_blobs | GMM (full covariance) | 3      | 0.907            |
| make_moons | K-Means (random)      | 2      | 0.493            |
| make_moons | GMM (full covariance) | 2      | 0.467            |

### Critical Insights

1. **Algorithm Assumptions Matter:** Both algorithms excel on spherical/Gaussian clusters (make_blobs) but struggle with non-convex shapes (make_moons), regardless of algorithmic sophistication.

2. **Initialization is Critical:** k-means++ initialization provided:
   - 37% faster convergence (fewer iterations)
   - 7.86% better silhouette scores
   - 46.32% lower inertia

3. **Full Covariance Advantage:** GMM with full covariance showed:
   - 61-point BIC advantage over diagonal covariance on blobs
   - 18.5% silhouette improvement on moons
   - Better modeling flexibility for complex cluster shapes

4. **Computational Trade-offs:** K-Means is 2-4× faster than GMM, but GMM provides probabilistic assignments and principled model selection via BIC.

5. **No Universal Solution:** Algorithm selection must be guided by data characteristics. Matching algorithmic assumptions to data structure is more important than algorithmic sophistication.

## Project Structure

```
project4/
├── README.md                         
├── CS6375_Project4_Report.pdf              # Final report 
├── AI TRANSCRIPT         \
├── src/                                    # Source code
    ├── data_generation.py                  # Dataset creation
    ├── metrics.py                          # Evaluation metrics
    ├── kmeans_experiments.py               # K-Means implementation
    ├── gmm_experiments.py                  # GMM implementation
    ├── comparison_analysis.py              # Comparative analysis
    ├── visualizations.py                   # Plotting functions
    ├── main.py                             # Main orchestration script
    └── figures/                            # Generated visualizations
        ├── blobs_raw_dataset.png           # Raw blobs data
        ├── moons_raw_dataset.png           # Raw moons data
        ├── clustering_grid_blobs_k3.png    # K-Means results on blobs
        ├── clustering_grid_moons_k2.png    # K-Means results on moons
        ├── kmeans_elbow_blobs.png          # Elbow method for blobs
        ├── kmeans_elbow_moons.png          # Elbow method for moons
        ├── gmm_bic_blobs.png               # GMM BIC analysis for blobs
        ├── gmm_bic_moons.png               # GMM BIC analysis for moons
        ├── performance_heatmap.png         # Algorithm comparison heatmap
        ├── silhouette_comparison_blobs.png # Detailed comparison (blobs)
        └── silhouette_comparison_moons.png # Detailed comparison (moons)
    └── results/ 
        ├── comparison_report.txt    
        ├── gmm_results.json
        └── kmeans_results.json        
```

## Implementation Details

### Dependencies
- Python 3.x
- NumPy (core computations)
- scikit-learn (K-Means and GMM implementations)
- Matplotlib (visualizations)
- Seaborn (enhanced visualizations) 

### Key Parameters
- **Random state:** 42 (for reproducibility)
- **Max iterations:** 100 (K-Means), 100 (GMM)
- **Convergence tolerance:** 1e-4

### Evaluation Metrics
- **Silhouette Score:** Measures cluster separation quality (range: [-1, 1], higher is better)
- **Inertia (ICSSD):** Within-cluster sum of squared distances (lower is better)
- **BIC/AIC:** Model selection criteria balancing fit and complexity (lower is better)
- **Iterations to Convergence:** Computational efficiency indicator

## How to Run

```bash
# Generate datasets and run all experiments
python3 src/main.py

# Results will be saved to:
# - figures/ directory (visualizations)
# - comparison_report.txt (detailed metrics)
```

## Results Highlights

### Best Performing Configurations

**For Spherical Clusters (make_blobs):**
- K-Means with k-means++ initialization at k=3
- GMM with full covariance at k=3
- Both achieve identical performance (0.907 silhouette)

**For Non-Convex Clusters (make_moons):**
- Neither algorithm performs well
- K-Means slightly edges out GMM
- Highlights fundamental limitation of spherical/Gaussian assumptions

### Model Selection

**Elbow Method (K-Means):**
- Clear elbow at k=3 for make_blobs
- No clear elbow for make_moons (gradual decrease)

**BIC Analysis (GMM):**
- Correctly identifies k=3 for make_blobs
- Continuously decreases for make_moons (indicates overfitting)

## Conclusions

This systematic analysis demonstrates that:

1. **Data characteristics dominate performance** - matching algorithmic assumptions to data structure is essential
2. **Initialization matters** - k-means++ provides significant benefits with negligible cost
3. **Flexibility has costs** - GMM's additional sophistication offers marginal benefits when K-Means assumptions hold
4. **Multiple metrics are essential** - relying on a single metric can be misleading
5. **No algorithm dominates universally** - practitioners must understand both algorithms and their data

