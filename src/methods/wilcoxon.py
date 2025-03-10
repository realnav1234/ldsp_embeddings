import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from src.utils import *
from src.constants import MODEL


def differences_t_test(embeddings_df):
    s1 = embeddings_df["Sentence1_embedding"]
    s2 = embeddings_df["Sentence2_embedding"]

    s1, s2 = np.vstack(s1), np.vstack(s2)
    diff = s1 - s2
    diff_means = np.mean(diff, axis=0)

    t_stats, p_values = np.empty(768), np.empty(768)

    for d in range(768):
        t_stat, p_val = stats.wilcoxon(diff[:, d])
        t_stats[d] = t_stat
        p_values[d] = p_val

    results = []
    for dim in range(768):
        results.append(
            {
                "dimension": dim,
                "wilcoxon_statistic": t_stats[dim],
                "wilcoxon_pvalue": p_values[dim],
                "mean_difference": diff_means[dim],
            }
        )

    results_df = pd.DataFrame(results)

    return t_stats, p_values, results_df


def perform_t_test(embeddings_df, dim):

    sentence1_values = [emb[dim] for emb in embeddings_df["Sentence1_embedding"]]
    sentence2_values = [emb[dim] for emb in embeddings_df["Sentence2_embedding"]]
    t_statistic, p_value = stats.ttest_ind(
        sentence1_values, sentence2_values, equal_var=False
    )
    return t_statistic, p_value


def save_t_test_results(results_df, results_directory):

    csv_filepath = os.path.join(results_directory, "t_test_results.csv")
    results_df.to_csv(csv_filepath, index=False)


# def apply_multiple_testing_correction(results_df):
#     """
#     Apply Benjamini-Hochberg or Bonferroni corrections to p-values
#     """
#     # Bonferroni correction
#     # results_df['p_value_bonferroni'] = multipletests(
#     #     results_df['p_value'],
#     #     method='bonferroni'
#     # )[1]

#     # Benjamini-Hochberg correction
#     results_df["p_value_bh"] = multipletests(results_df["p_value"], method="fdr_bh")[1]

#     return results_df


def plot_top_and_bottom_p_values(
    results_df, embeddings_df, results_directory, test_type="t_test", numplots=2
):
    """
    Plot distributions for dimensions with most and least significant differences.

    Parameters:
        test_type: Either 't_test' or 'wilcoxon' to determine which p-values to use
    """
    pval_col = "t_pvalue_bh" if test_type == "t_test" else "wilcoxon_pvalue_bh"

    top_dims = results_df.nsmallest(numplots, pval_col)["dimension"].values
    bottom_dims = results_df.nlargest(numplots, pval_col)["dimension"].values

    # Plot top dimensions
    plt.figure(figsize=(15, 10))
    for i, dim in enumerate(top_dims):
        plt.subplot(2, 2, i + 1)
        sentence1_values = [emb[dim] for emb in embeddings_df["Sentence1_embedding"]]
        sentence2_values = [emb[dim] for emb in embeddings_df["Sentence2_embedding"]]
        plt.hist(sentence1_values, alpha=0.5, label="Sentence 1", bins=30)
        plt.hist(sentence2_values, alpha=0.5, label="Sentence 2", bins=30)
        plt.xlabel(f"Dimension {dim}")
        plt.ylabel("Frequency")
        plt.title(f"Top {i+1}: Dimension {dim}")
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_directory, f"top_{numplots}_p_values_{test_type}.png")
    )
    plt.close()

    # Plot bottom dimensions
    plt.figure(figsize=(15, 5))
    for i, dim in enumerate(bottom_dims):
        plt.subplot(2, 2, i + 1)
        sentence1_values = [emb[dim] for emb in embeddings_df["Sentence1_embedding"]]
        sentence2_values = [emb[dim] for emb in embeddings_df["Sentence2_embedding"]]
        plt.hist(sentence1_values, alpha=0.5, label="Sentence 1", bins=30)
        plt.hist(sentence2_values, alpha=0.5, label="Sentence 2", bins=30)
        plt.xlabel(f"Dimension {dim}")
        plt.ylabel("Frequency")
        plt.title(f"Bottom {i+1}: Dimension {dim}")
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_directory, f"bottom_{numplots}_p_values_{test_type}.png")
    )
    plt.close()

    plt.figure(figsize=(12, 10))

    top1_dim = top_dims[0]
    plt.subplot(2, 1, 1)
    sentence1_values = [emb[top1_dim] for emb in embeddings_df["Sentence1_embedding"]]
    sentence2_values = [emb[top1_dim] for emb in embeddings_df["Sentence2_embedding"]]
    plt.hist(sentence1_values, alpha=0.5, label="Sentence 1", bins=30)
    plt.hist(sentence2_values, alpha=0.5, label="Sentence 2", bins=30)
    plt.xlabel(f"Dimension {top1_dim}")
    plt.ylabel("Frequency")
    plt.title(f"Top-1: Dimension {top1_dim}")
    plt.legend(loc="upper right")

    bottom1_dim = bottom_dims[0]
    plt.subplot(2, 1, 2)
    sentence1_values = [
        emb[bottom1_dim] for emb in embeddings_df["Sentence1_embedding"]
    ]
    sentence2_values = [
        emb[bottom1_dim] for emb in embeddings_df["Sentence2_embedding"]
    ]
    plt.hist(sentence1_values, alpha=0.5, label="Sentence 1", bins=30)
    plt.hist(sentence2_values, alpha=0.5, label="Sentence 2", bins=30)
    plt.xlabel(f"Dimension {bottom1_dim}")
    plt.ylabel("Frequency")
    plt.title(f"Bottom-1: Dimension {bottom1_dim}")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_directory, f"top1_bottom1_p_values_{test_type}.png")
    )
    plt.close()


def plot_difference_distributions(t_stats, p_vals, results_directory):
    """
    Create a bar plot showing the significance of differences across all dimensions.
    Taller bars indicate more significant differences from zero.
    """
    plt.figure(figsize=(20, 6))

    # Convert p-values to -log10 scale for better visualization
    # Add small constant to avoid log(0)
    log_p_values = -np.log10(p_vals + 1e-300)

    plt.bar(range(len(log_p_values)), log_p_values, alpha=0.6)

    plt.axhline(
        y=-np.log10(0.05),
        color="r",
        linestyle="--",
        alpha=0.5,
        label="p=0.05 threshold",
    )

    plt.xlabel("Embedding Dimensions")
    plt.ylabel("-log10(p-value)")
    plt.title("Significance of Differences Across Embedding Dimensions")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "dimension_differences.png"))
    plt.close()


if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths(model_name=MODEL)

    for embeddings_csv in tqdm(embedding_filepaths):
        embeddings_df = read_embeddings_df(embeddings_csv)
        results_directory = get_results_directory(
            embeddings_csv, "t_test_analysis", model_name=MODEL
        )

        # Run Wilcoxon analysis on differences
        t_stats, p_vals, wilcoxon_df = differences_t_test(embeddings_df)

        # Apply BH correction to Wilcoxon results
        # wilcoxon_df["wilcoxon_pvalue_bh"] = multipletests(
        #     wilcoxon_df["wilcoxon_pvalue"], method="fdr_bh"
        # )[1]

        wilcoxon_df.to_csv(
            os.path.join(results_directory, "wilcoxon_results.csv"), index=False
        )

        # Generate and save plots
        plot_difference_distributions(t_stats, p_vals, results_directory)

        plot_top_and_bottom_p_values(
            wilcoxon_df,
            embeddings_df,
            results_directory,
            test_type="wilcoxon",
            numplots=4,
        )

    print("\nT-test analysis complete.")
