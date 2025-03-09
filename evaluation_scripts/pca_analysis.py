import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils import *

def compute_difference_vectors(embeddings_df):
    # Compute difference vectors
    embeddings_df['diff_vector'] = embeddings_df['Sentence1_embedding'] - embeddings_df['Sentence2_embedding']
    return embeddings_df

def perform_pca(diff_vectors):
    # Perform PCA on all components
    pca = PCA()
    pca.fit(diff_vectors)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    return pca, explained_variance, cumulative_variance

def plot_elbow(cumulative_variance, results_directory):
    # Plot cumulative variance to visualize the elbow
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             marker='o', linestyle='--', color='r', label='Cumulative Variance')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Explained Variance Threshold')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Elbow Method for PCA')
    plt.legend()
    plt.grid()
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "pca_elbow_plot.png"))
    plt.close()

def find_optimal_components(cumulative_variance, threshold=0.95):
    # Find the optimal number of components to retain the specified variance
    optimal_components = np.argmax(cumulative_variance >= threshold) + 1
    return optimal_components

def aggregate_contributions(pca, optimal_components):
    # Aggregate absolute contributions across the optimal principal components
    selected_components = pca.components_[:optimal_components]
    overall_contributions = np.sum(np.abs(selected_components), axis=0)
    return overall_contributions

def report_top_dimensions(overall_contributions, top_k=20):
    # Find the top K embedding dimensions
    top_dimensions = np.argsort(overall_contributions)[-top_k:][::-1]
    return top_dimensions

def plot_contributions(overall_contributions, top_dimensions, results_directory):
    # Plot histogram of contributions with highlighted top dimensions
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(overall_contributions)), overall_contributions, alpha=0.6)
    
    # Highlight the top dimensions
    for dim in top_dimensions:
        bars[dim].set_color('red')
        bars[dim].set_alpha(1.0)
    
    # Truncate the y-axis to start at the minimum value
    plt.ylim(bottom=np.min(overall_contributions) - 0.01)
    
    # Add labels and legend
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Aggregate Contribution')
    plt.title('Overall Contributions of Embedding Dimensions to PCA\n(Top 20 Dimensions Highlighted)')
    plt.grid(axis='y')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=1.0, label='Top 20 Dimensions'),
                       Patch(facecolor='blue', alpha=0.6, label='Other Dimensions')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "pca_contributions.png"))
    plt.close()

def save_pca_results(optimal_components, top_dimensions, overall_contributions, results_directory):
    # Save optimal number of components and top dimensions to CSV
    results_df = pd.DataFrame({
        'Rank': range(1, len(top_dimensions) + 1),
        'Dimension': top_dimensions,
        'Contribution': overall_contributions[top_dimensions]
    })
    # Add a row for the optimal number of components (as metadata)
    metadata_df = pd.DataFrame({
        'Rank': ['Optimal Components'],
        'Dimension': [optimal_components],
        'Contribution': [np.nan]
    })
    # Combine metadata and results
    combined_df = pd.concat([metadata_df, results_df], ignore_index=True)
    combined_df.to_csv(os.path.join(results_directory, "pca_results.csv"), index=False)

    all_df = pd.DataFrame({
        'Dimension': [i + 1 for i in range(len(overall_contributions))],
        'Contribution': overall_contributions
    })

    all_df.to_csv(os.path.join(results_directory, "pca_all.csv"), index=False)

if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):
        embeddings_df = read_embeddings_df(embeddings_csv)

        # Compute difference vectors
        embeddings_df = compute_difference_vectors(embeddings_df)

        # Convert the diff_vector column into a numpy array for PCA
        diff_vectors = np.array(embeddings_df['diff_vector'].tolist())

        # Perform PCA
        pca, explained_variance, cumulative_variance = perform_pca(diff_vectors)

        # Get results directory
        results_directory = get_results_directory(embeddings_csv, "pca_analysis")

        # Plot elbow and determine optimal components
        plot_elbow(cumulative_variance, results_directory)
        optimal_components = find_optimal_components(cumulative_variance, threshold=0.95)
        print(f"Optimal number of components to retain 95% variance: {optimal_components}")

        # Perform PCA with optimal components
        pca_optimal = PCA(n_components=optimal_components)
        pca_optimal.fit(diff_vectors)

        # Aggregate contributions
        overall_contributions = aggregate_contributions(pca_optimal, optimal_components)

        # Report top 20 dimensions
        top_dimensions = report_top_dimensions(overall_contributions, top_k=20)
        print(f"Top 20 embedding dimensions contributing to PCA:")
        for rank, dim in enumerate(top_dimensions, start=1):
            contribution = overall_contributions[dim]
            print(f"{rank}. Dimension {dim}: Contribution {contribution:.4f}")

        # Save PCA results including optimal components and top dimensions
        save_pca_results(optimal_components, top_dimensions, overall_contributions, results_directory)

        # Plot contributions
        plot_contributions(overall_contributions, top_dimensions, results_directory)

    print("PCA analysis complete.")
