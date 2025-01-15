import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import *

def perform_t_test(embeddings_df, dim):
    
    sentence1_values = [emb[dim] for emb in embeddings_df['Sentence1_embedding']]
    sentence2_values = [emb[dim] for emb in embeddings_df['Sentence2_embedding']]
    t_statistic, p_value = stats.ttest_ind(sentence1_values, sentence2_values)
    return t_statistic, p_value


def save_t_test_results(results_df, results_directory):
    
    csv_filepath = os.path.join(results_directory, "t_test_results.csv")
    results_df.to_csv(csv_filepath, index=False)


def apply_multiple_testing_correction(results_df):
    """
    Apply both Bonferroni and Benjamini-Hochberg corrections to p-values
    """
    # Bonferroni correction
    results_df['p_value_bonferroni'] = multipletests(
        results_df['p_value'], 
        method='bonferroni'
    )[1]
    
    # Benjamini-Hochberg correction
    results_df['p_value_bh'] = multipletests(
        results_df['p_value'], 
        method='fdr_bh'
    )[1]
    
    return results_df


def plot_top_and_bottom_p_values(results_df, embeddings_df, results_directory):
    
    # Get top-4 and bottom-4 dimensions based on p-values
    top_4_dims = results_df.nsmallest(4, 'p_value_bh')['dimension'].values
    bottom_4_dims = results_df.nlargest(4, 'p_value_bh')['dimension'].values

    # Plot top-4 dimensions
    plt.figure(figsize=(15, 10))
    for i, dim in enumerate(top_4_dims):
        plt.subplot(2, 2, i + 1)
        sentence1_values = [emb[dim] for emb in embeddings_df['Sentence1_embedding']]
        sentence2_values = [emb[dim] for emb in embeddings_df['Sentence2_embedding']]
        plt.hist(sentence1_values, alpha=0.5, label='Sentence 1', bins=30)
        plt.hist(sentence2_values, alpha=0.5, label='Sentence 2', bins=30)
        plt.xlabel(f'Dimension {dim}')
        plt.ylabel('Frequency')
        plt.title(f'Top {i+1}: Dimension {dim}')
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "top_4_p_values.png"))
    plt.close()

    # Plot bottom-4 dimensions
    plt.figure(figsize=(15, 10))
    for i, dim in enumerate(bottom_4_dims):
        plt.subplot(2, 2, i + 1)
        sentence1_values = [emb[dim] for emb in embeddings_df['Sentence1_embedding']]
        sentence2_values = [emb[dim] for emb in embeddings_df['Sentence2_embedding']]
        plt.hist(sentence1_values, alpha=0.5, label='Sentence 1', bins=30)
        plt.hist(sentence2_values, alpha=0.5, label='Sentence 2', bins=30)
        plt.xlabel(f'Dimension {dim}')
        plt.ylabel('Frequency')
        plt.title(f'Bottom {i+1}: Dimension {dim}')
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "bottom_4_p_values.png"))
    plt.close()


if __name__ == "__main__":

    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):
        embeddings_df = read_embeddings_df(embeddings_csv)

        results = []
        for dim in range(len(embeddings_df['Sentence1_embedding'][0])):
            t_statistic, p_value = perform_t_test(embeddings_df, dim)
            results.append({'dimension': dim, 't_statistic': t_statistic, 'p_value': p_value})

        results_df = pd.DataFrame(results)
        
        # Apply multiple testing correction
        results_df = apply_multiple_testing_correction(results_df)

        results_directory = get_results_directory(embeddings_csv, "t_test_analysis")

        save_t_test_results(results_df, results_directory)

        plot_top_and_bottom_p_values(results_df, embeddings_df, results_directory)
        

    print("T-test analysis complete.")