import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import *


def create_combined_graph(mutual_info_df, wilcoxon_results_df, clf_weights_df, rfe_results_df, results_directory, N=15):
    """
    Create a combined visualization showing mutual information and significant dimensions 
    from different analyses
    """
    lp = results_directory.split('/')[1]
   
    mutual_informations = mutual_info_df['Mutual_Information'].values
    top_N_wilcoxon = wilcoxon_results_df.nsmallest(N, 'wilcoxon_pvalue_bh')['dimension'].values
    rfe_dimensions = rfe_results_df['Feature'].values
    
    # Get top N MI dimensions
    top_N_mi = np.argsort(mutual_informations)[-N:]

    plt.figure(figsize=(12, 6))

    # Create bar colors array - default darker gray for most bars
    bar_colors = ['#BDBDBD'] * len(mutual_informations)  # Darker gray
    # Set blue color for top Wilcoxon dimensions
    for dim in top_N_wilcoxon:
        bar_colors[dim] = '#1565C0'  # Darker blue

    # Plot mutual information bars with colors
    plt.bar(np.arange(len(mutual_informations)), mutual_informations, 
           color=bar_colors, label='Mutual Information', alpha=0.8)

    # Add a blue bar to legend for Wilcoxon
    plt.bar([-1], [0], color='#1565C0', alpha=0.8, label=f'Top {N} (Wilcoxon)')

    # Add threshold line for top N mutual information values
    threshold = np.sort(mutual_informations)[-N]
    plt.axhline(y=threshold, color='#AD6ED1', linestyle='--', label=f'Top {N} MI Threshold')  # Green

    # Plot RFE dimensions
    plt.scatter(rfe_dimensions, mutual_informations[rfe_dimensions], 
               color='#00C853', marker='^', label=f'Top {N} (RFE)', s=50, alpha=0.8)  # Bright green

    # Find dimensions that are in all three categories
    triple_overlap = set(top_N_wilcoxon) & set(rfe_dimensions) & set(top_N_mi)
    if triple_overlap:
        plt.scatter(list(triple_overlap), mutual_informations[list(triple_overlap)],
                   facecolors='none', edgecolors='red', s=200, linewidth=2,
                   label='Top 25 in All 3 Analyses', marker='o')

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Mutual Information")
    plt.title(f"Mutual Information of Embedding Dimensions: {lp}")
    plt.legend()

    graph_filepath = os.path.join(results_directory, "combined_graph.png")
    plt.tight_layout()
    plt.savefig(graph_filepath)
    plt.close()

    print(f"Combined graph saved at: {graph_filepath}")
    if triple_overlap:
        print(f"Dimensions present in all analyses: {sorted(triple_overlap)}")


if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):
        results_directory = get_results_directory(embeddings_csv, "combined_analysis")

        # Load all the analysis results
        mutual_info_df = pd.read_csv(os.path.join(
            get_results_directory(embeddings_csv, "mutual_information"), 
            "mutual_information_all.csv"))
        
        wilcoxon_results_df = pd.read_csv(os.path.join(
            get_results_directory(embeddings_csv, "t_test_analysis"), 
            "wilcoxon_results.csv"))
        
        clf_weights_df = pd.read_csv(os.path.join(
            get_results_directory(embeddings_csv, "logistic_classifier"), 
            "classifier_weights.csv"))
        
        rfe_results_df = pd.read_csv(os.path.join(
            get_results_directory(embeddings_csv, "rfe_analysis"), 
            "rfe_results.csv"))

        create_combined_graph(mutual_info_df, wilcoxon_results_df, 
                            clf_weights_df, rfe_results_df, results_directory, N=25)

