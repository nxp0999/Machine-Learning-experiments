"""
Main Execution Script
Orchestrates the complete clustering experiment pipeline
CS 6375 - Machine Learning - Project 4
"""
import os
import sys
import time
from datetime import datetime


def print_header(text, char='=', width=80):
    """Print a formatted header"""
    print("\n" + char*width)
    print(text.center(width))
    print(char*width)


def print_step(step_num, total_steps, description):
    """Print a step indicator"""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 60)


def create_directories():
    """Create necessary project directories"""
    directories = ['results', 'figures', 'results/metrics']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directories created/verified")


def run_data_generation():
    """Generate and visualize datasets"""
    from data_generation import generate_all_datasets, visualize_raw_data
    
    print("Generating synthetic datasets...")
    datasets = generate_all_datasets(n_samples=150, random_state=42)
    
    print("\nVisualizing datasets...")
    visualize_raw_data(datasets, save_dir='figures/')
    
    return datasets


def run_kmeans_experiments(datasets):
    """Run all K-Means experiments"""
    from kmeans_experiments import run_kmeans_experiments, save_results
    
    print("Running K-Means experiments...")
    results = run_kmeans_experiments(
        datasets,
        k_values=[2, 3, 4, 5],
        init_methods=['random', 'k-means++']
    )
    
    print("\nSaving K-Means results...")
    save_results(results, 'results/kmeans_results.json')
    
    return results


def run_gmm_experiments(datasets):
    """Run all GMM experiments"""
    from gmm_experiments import run_gmm_experiments, save_results
    
    print("Running GMM experiments...")
    results = run_gmm_experiments(
        datasets,
        k_values=[2, 3, 4, 5],
        covariance_types=['full', 'diag']
    )
    
    print("\nSaving GMM results...")
    save_results(results, 'results/gmm_results.json')
    
    return results


def create_visualizations():
    """Generate all visualizations"""
    from visualizations import create_all_visualizations
    
    print("Creating visualizations...")
    create_all_visualizations()


def run_comparison_analysis():
    """Run comprehensive comparison analysis"""
    from comparison_analysis import (
        generate_comparison_report,
        create_performance_heatmap,
        load_all_results,
        print_comparison_summary
    )
    
    print("Generating comparison report...")
    generate_comparison_report()
    
    print("\nCreating performance heatmap...")
    kmeans_results, gmm_results = load_all_results()
    create_performance_heatmap(kmeans_results, gmm_results)
    
    print("\nPrinting comparison summary...")
    print_comparison_summary()


def print_final_summary():
    """Print final summary of generated files"""
    print_header("PIPELINE COMPLETE!", char='=')
    
    print("\n📁 Generated Files:")
    print("\n  Results:")
    print("    ├── results/kmeans_results.json")
    print("    ├── results/gmm_results.json")
    print("    └── results/comparison_report.txt")
    
    print("\n  Figures:")
    print("    ├── figures/blobs_raw_dataset.png")
    print("    ├── figures/moons_raw_dataset.png")
    print("    ├── figures/kmeans_elbow_blobs.png")
    print("    ├── figures/kmeans_elbow_moons.png")
    print("    ├── figures/silhouette_comparison_blobs.png")
    print("    ├── figures/silhouette_comparison_moons.png")
    print("    ├── figures/gmm_bic_blobs.png")
    print("    ├── figures/gmm_bic_moons.png")
    print("    ├── figures/clustering_grid_blobs_k3.png")
    print("    ├── figures/clustering_grid_moons_k2.png")
    print("    └── figures/performance_heatmap.png")
    
    print("\n📊 Experiment Statistics:")
    import json
    
    # Count experiments
    with open('results/kmeans_results.json', 'r') as f:
        kmeans_results = json.load(f)
    with open('results/gmm_results.json', 'r') as f:
        gmm_results = json.load(f)
    
    kmeans_count = sum(len(v) for v in kmeans_results.values())
    gmm_count = sum(len(v) for v in gmm_results.values())
    
    print(f"    ├── Total K-Means experiments: {kmeans_count}")
    print(f"    ├── Total GMM experiments: {gmm_count}")
    print(f"    └── Total experiments: {kmeans_count + gmm_count}")
    
    print("\n💡 Next Steps:")
    print("    1. Review results/comparison_report.txt for detailed analysis")
    print("    2. Examine figures/ directory for all visualizations")
    print("    3. Use results for your project report and presentation")
    
    print("\n" + "="*80)


def run_full_pipeline():
    """
    Execute the complete clustering analysis pipeline
    """
    start_time = time.time()
    total_steps = 6
    
    print_header("CS 6375 - MACHINE LEARNING PROJECT 4", char='=')
    print("Unsupervised Learning: K-Means vs GMM Clustering Analysis")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Setup
        print_step(1, total_steps, "Setting up project structure")
        create_directories()
        
        # Step 2: Data Generation
        print_step(2, total_steps, "Generating datasets")
        datasets = run_data_generation()
        
        # Step 3: K-Means Experiments
        print_step(3, total_steps, "Running K-Means experiments")
        kmeans_results = run_kmeans_experiments(datasets)
        
        # Step 4: GMM Experiments
        print_step(4, total_steps, "Running GMM experiments")
        gmm_results = run_gmm_experiments(datasets)
        
        # Step 5: Visualizations
        print_step(5, total_steps, "Creating visualizations")
        create_visualizations()
        
        # Step 6: Comparison Analysis
        print_step(6, total_steps, "Running comparison analysis")
        run_comparison_analysis()
        
        # Final Summary
        elapsed_time = time.time() - start_time
        print_final_summary()
        
        print(f"\n⏱️  Total Execution Time: {elapsed_time:.2f} seconds")
        print(f"📅 Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0  # Success
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    print("\n" + "🚀 " + "Starting clustering analysis pipeline..." + " 🚀\n")
    
    exit_code = run_full_pipeline()
    
    if exit_code == 0:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()