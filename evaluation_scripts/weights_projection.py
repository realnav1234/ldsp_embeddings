import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import *


def create_projection_plot(X, y, weights, results_directory):
   
    projections = X @ weights

    plt.figure(figsize=(10, 6))
    plt.hist(projections[y == 1], bins=30, alpha=0.7, label='Negation Present (y=1)', color='blue')
    plt.hist(projections[y == 0], bins=30, alpha=0.7, label='Negation Absent (y=0)', color='orange')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, label='Decision Boundary')
    plt.xlabel('Projection onto Classifier Weight Vector')
    plt.ylabel('Frequency')
    plt.title('Projection of Embeddings onto Classifier Weight Vector')
    plt.legend()
    plt.grid(alpha=0.3)

    plot_filepath = os.path.join(results_directory, "projections_plot.png")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.close()

    print(f"Projection plot saved at: {plot_filepath}")


if __name__ == "__main__":

    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):

        results_directory = get_results_directory(embeddings_csv, "logistic_classifier")
        os.makedirs(results_directory, exist_ok=True)


        embeddings_df = read_embeddings_df(embeddings_csv)
        X_sentence1 = np.array(embeddings_df['Sentence1_embedding'].tolist())
        X_sentence2 = np.array(embeddings_df['Sentence2_embedding'].tolist())
        y = np.concatenate([np.zeros(len(X_sentence1)), np.ones(len(X_sentence2))])
        X = np.concatenate([X_sentence1, X_sentence2], axis=0)


        clf_weights_df = pd.read_csv(os.path.join(results_directory, "classifier_weights.csv"))
        weights = clf_weights_df['Weight'].values

        create_projection_plot(X, y, weights, results_directory)