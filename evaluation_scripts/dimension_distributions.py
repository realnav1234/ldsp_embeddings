import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from utils import *

def get_stats(embedding_column):
    
    emb_arr = np.array([emb for emb in embedding_column])
    means = np.mean(emb_arr, axis=0)
    stds = np.std(emb_arr, axis=0)
    return means, stds


def plot_dimension_distributions(embeddings_df, sampled_dimensions, results_directory, plot_name = "random_dimension_distributions.png"):
    
    plt.figure(figsize=(15, 10))

    for i, dim in enumerate(sampled_dimensions):
        plt.subplot(3, 2, i + 1)

        sentence1_values = [emb[dim] for emb in embeddings_df['Sentence1_embedding']]
        sentence2_values = [emb[dim] for emb in embeddings_df['Sentence2_embedding']]

        plt.hist(sentence1_values, alpha=0.5, label='Sentence 1', bins=30)
        plt.hist(sentence2_values, alpha=0.5, label='Sentence 2', bins=30)

        plt.xlabel(f'Dimension {dim}')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of values in dimension {dim}')
        plt.legend(loc='upper right')

    plt.tight_layout()  
    plt.savefig(os.path.join(results_directory, plot_name))
    plt.close()


if __name__ == "__main__":

    embedding_filepaths = get_embeddings_filepaths()
    num_dimensions_to_sample = 6

    for embeddings_csv in tqdm(embedding_filepaths):
        embeddings_df = read_embeddings_df(embeddings_csv)

        mean_sentence1, std_sentence1 = get_stats(embeddings_df['Sentence1_embedding'])
        mean_sentence2, std_sentence2 = get_stats(embeddings_df['Sentence2_embedding'])

        sampled_dimensions = random.sample(range(len(mean_sentence1)), num_dimensions_to_sample)

        results_directory = get_results_directory(embeddings_csv, "dimension_distributions")

        plot_dimension_distributions(embeddings_df, sampled_dimensions, results_directory)

        stats_df = pd.DataFrame({
            "Dimension": np.arange(len(mean_sentence1)) + 1,
            "Mean_Sentence1": mean_sentence1,
            "Std_Sentence1": std_sentence1,
            "Mean_Sentence2": mean_sentence2,
            "Std_Sentence2": std_sentence2
        })

        stats_df.to_csv(os.path.join(results_directory, "embedding_dimensions_stats.csv"), index=False)
        

    print("Dimension distribution analysis complete.")