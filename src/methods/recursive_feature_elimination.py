from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from src.utils import *
from src.constants import MODEL


def perform_rfe(embeddings_df, n_features=20):

    sentence1_df = pd.DataFrame(
        {"embedding": embeddings_df["Sentence1_embedding"].tolist(), "label": 0}
    )
    sentence2_df = pd.DataFrame(
        {"embedding": embeddings_df["Sentence2_embedding"].tolist(), "label": 1}
    )
    df = pd.concat([sentence1_df, sentence2_df], ignore_index=True)

    X = np.array(df["embedding"].tolist())
    y = np.array(df["label"].tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize RFE with the Logistic Regression classifier
    rfe = RFE(
        estimator=LogisticRegression(random_state=0, max_iter=1000),
        n_features_to_select=n_features,
    )

    # Fit RFE to the training data
    rfe.fit(X_train, y_train)

    # Get the selected features
    selected_features = np.where(rfe.support_)[0]

    # Transform the data to keep only selected features
    X_train_rfe = X_train[:, selected_features]
    X_test_rfe = X_test[:, selected_features]

    # Train a logistic regression model on the selected features
    clf_rfe = LogisticRegression(random_state=0, max_iter=1000)
    clf_rfe.fit(X_train_rfe, y_train)

    # Evaluate the model
    y_pred_rfe = clf_rfe.predict(X_test_rfe)
    accuracy_rfe = accuracy_score(y_test, y_pred_rfe)

    return accuracy_rfe, selected_features, clf_rfe


def save_rfe_results(selected_features, accuracy, model, results_directory):
    # Save selected features and their importance to CSV
    feature_importance = np.abs(
        model.coef_[0]
    )  # Get feature importance from logistic regression
    results_df = pd.DataFrame(
        {
            "Rank": range(1, len(selected_features) + 1),
            "Feature": selected_features,
            "Importance": feature_importance,
        }
    )
    results_df.to_csv(os.path.join(results_directory, "rfe_results.csv"), index=False)

    # Save accuracy and details to a text file
    with open(os.path.join(results_directory, "rfe_summary.txt"), "w") as f:
        f.write(f"RFE Analysis Results\n")
        f.write(f"Number of selected features: {len(selected_features)}\n")
        f.write(f"Model accuracy: {accuracy:.4f}\n\n")
        f.write("Selected features and their importance:\n")
        for rank, feature, importance in zip(
            results_df["Rank"], results_df["Feature"], results_df["Importance"]
        ):
            f.write(f"{rank}. Feature {feature}: Importance {importance:.4f}\n")

    # Create and save visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(selected_features)), feature_importance)
    plt.xlabel("Selected Features")
    plt.ylabel("Feature Importance")
    plt.title("Feature Importance from RFE Analysis")
    plt.xticks(range(len(selected_features)), selected_features, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "rfe_feature_importance.png"))
    plt.close()


if __name__ == "__main__":

    embedding_filepaths = get_embeddings_filepaths(model_name=MODEL)
    for embeddings_csv in tqdm(embedding_filepaths):
        # Get results directory
        results_directory = get_results_directory(
            embeddings_csv, "rfe_analysis", model_name=MODEL
        )

        # Perform RFE
        accuracy, selected_features, model = perform_rfe(
            read_embeddings_df(embeddings_csv)
        )

        # Save results
        save_rfe_results(selected_features, accuracy, model, results_directory)
