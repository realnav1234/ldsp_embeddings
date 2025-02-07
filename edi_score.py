import numpy as np
import pandas as pd
from utils import *
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
from skopt import gp_minimize
from skopt.space import Real


def calculate_edi_scores(mutual_info_df, wilcoxon_results_df, clf_weights_df, rfe_results_df, N=20):
    """
    Calculate EDI (Embedding Dimension Importance) scores incorporating:
    - Mutual Information scores
    - Wilcoxon test p-values (negative log likelihood)
    - RFE-selected logistic regression weights
    """
    # Define weights for different analyses
    WEIGHTS = {
        'mutual_info': 0.33,
        'wilcoxon': 0.33,
        'rfe_logistic': 0.34
    }
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-10, "Weights must sum to 1"

    # Initialize scores arrays
    n_dims = 768
    mi_scores = np.zeros(n_dims)
    wilcoxon_scores = np.zeros(n_dims)
    rfe_logistic_scores = np.zeros(n_dims)

    # Calculate Mutual Information scores
    mi_values = mutual_info_df['Mutual_Information'].values
    mi_scores = (mi_values - mi_values.min()) / (mi_values.max() - mi_values.min())

    # Calculate Wilcoxon scores using negative log likelihood of p-values
    epsilon = 1e-15  # To avoid log(0)
    p_values = wilcoxon_results_df['wilcoxon_pvalue_bh'].values
    wilcoxon_scores = -np.log(p_values + epsilon)
    wilcoxon_scores = (wilcoxon_scores - wilcoxon_scores.min()) / (wilcoxon_scores.max() - wilcoxon_scores.min())

    # Calculate RFE-Logistic scores
    rfe_dimensions = rfe_results_df['Feature'].values
    clf_weights = clf_weights_df['Weight'].values
    # Only use logistic weights for RFE-selected features
    rfe_logistic_scores[rfe_dimensions] = clf_weights[rfe_dimensions]
    # Normalize RFE-logistic scores
    if rfe_logistic_scores.max() != rfe_logistic_scores.min():
        rfe_logistic_scores = (rfe_logistic_scores - rfe_logistic_scores.min()) / (rfe_logistic_scores.max() - rfe_logistic_scores.min())

    # Combine scores using weights
    final_scores = (
        WEIGHTS['mutual_info'] * mi_scores +
        WEIGHTS['wilcoxon'] * wilcoxon_scores +
        WEIGHTS['rfe_logistic'] * rfe_logistic_scores
    )

    # Create and sort DataFrame
    scores_df = pd.DataFrame({
        'dimension': range(n_dims),
        'edi_score': final_scores,
        'mi_score': mi_scores,
        'wilcoxon_score': wilcoxon_scores,
        'rfe_logistic_score': rfe_logistic_scores
    })
    
    scores_df = scores_df.sort_values('edi_score', ascending=False)
    
    return scores_df


def save_edi_scores(scores_df, results_directory):
    """Save EDI scores to CSV file"""
    output_path = os.path.join(results_directory, "edi_score.csv")
    scores_df.to_csv(output_path, index=False)
    print(f"EDI scores saved to: {output_path}")


if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):
        results_directory = get_results_directory(embeddings_csv, "edi_scores")
        
        # Load analysis results
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

        # Calculate and save EDI scores
        edi_scores = calculate_edi_scores(
            mutual_info_df, wilcoxon_results_df, clf_weights_df, rfe_results_df
            )
        
        save_edi_scores(edi_scores, results_directory)

    print("\nEDI score calculation complete.")


    

    