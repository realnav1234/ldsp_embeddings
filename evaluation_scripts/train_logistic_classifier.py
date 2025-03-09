import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import *

def train_logistic_classifier(embeddings_df, results_directory):
    
    sentence1_df = pd.DataFrame({'embedding': embeddings_df['Sentence1_embedding'].tolist(), 'label': 0})
    sentence2_df = pd.DataFrame({'embedding': embeddings_df['Sentence2_embedding'].tolist(), 'label': 1})

    df = pd.concat([sentence1_df, sentence2_df], ignore_index=True)

    X = np.array(df['embedding'].tolist())
    y = np.array(df['label'].tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    weights = clf.coef_[0]
    most_influential_dimensions = np.argsort(np.abs(weights))[::-1]

    # Save accuracy and top 10 dimensions to a .txt file
    txt_filepath = os.path.join(results_directory, "classifier_results.txt")
    with open(txt_filepath, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Top 10 Most Influential Dimensions:\n")
        for dim in most_influential_dimensions[:10]:
            f.write(f"Dimension {dim}: Weight {weights[dim]}\n")

    # Save weights to a CSV file
    weights_df = pd.DataFrame({
        "Dimension": np.arange(len(weights)),
        "Weight": weights
    })
    weights_csv_filepath = os.path.join(results_directory, "classifier_weights.csv")
    weights_df.to_csv(weights_csv_filepath, index=False)

    model_filepath = os.path.join(results_directory, "logistic_classifier.pkl")
    with open(model_filepath, "wb") as model_file:
        pickle.dump(clf, model_file)

    print("Classifier training complete.")
    print(f"Accuracy: {accuracy}")
    print(f"Results saved in: {results_directory}")


if __name__ == "__main__":

    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):
        embeddings_df = read_embeddings_df(embeddings_csv)

        results_directory = get_results_directory(embeddings_csv, "logistic_classifier")

        train_logistic_classifier(embeddings_df, results_directory)
        

    print("Logistic Classifier Training Complete")