import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from tqdm import tqdm
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import shap

from constants import *


def load_and_process_embeddings(embedding_filepaths):
    """
    Load embeddings from all properties and create difference vectors with labels
    """
    all_diff_vectors = []
    all_labels = []
    property_names = []

    for filepath in embedding_filepaths:

        # Extract property name from filepath
        property_name = get_linguistic_property(filepath)
        property_names.append(property_name)

        # Load embeddings
        embeddings_df = read_embeddings_df(filepath)

        # Calculate difference vectors
        sent1_embeddings = np.array(embeddings_df["Sentence1_embedding"].tolist())
        sent2_embeddings = np.array(embeddings_df["Sentence2_embedding"].tolist())
        diff_vectors = sent2_embeddings - sent1_embeddings

        # Add to collections
        all_diff_vectors.extend(diff_vectors)
        all_labels.extend([property_name] * len(diff_vectors))

    all_diff_vectors, all_labels = np.array(all_diff_vectors), np.array(all_labels)
    return all_diff_vectors, all_labels, property_names


def train_property_classifier(X, y):
    """
    Train a multiclass logistic regression classifier with progress tracking
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    start_time = time.time()
    print("Starting training...")

    train_sizes, train_scores, val_scores = learning_curve(
        clf,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    clf.fit(X_train_final, y_train_final)

    train_pred = clf.predict(X_train_final)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train_final, train_pred),
        "train_f1": f1_score(y_train_final, train_pred, average="weighted"),
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_f1": f1_score(y_val, val_pred, average="weighted"),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred, average="weighted"),
        "training_time": time.time() - start_time,
        "learning_curves": {
            "train_sizes": train_sizes,
            "train_scores": train_scores,
            "val_scores": val_scores,
        },
    }

    return clf, X_test, y_test, test_pred, metrics


def analyze_classifier_weights(clf, property_names, n_top_dimensions=10):
    """
    Analyze classifier weights to identify important dimensions for each property
    """
    weights_by_property = {}

    for idx, property_name in enumerate(property_names):
        property_weights = clf.coef_[idx]

        # Get top dimensions by absolute weight
        top_dims = np.argsort(np.abs(property_weights))[-n_top_dimensions:]
        top_weights = property_weights[top_dims]

        weights_by_property[property_name] = {
            "dimensions": top_dims,
            "weights": top_weights,
        }

    return weights_by_property


def plot_learning_curves(metrics, results_directory):
    """
    Plot learning curves showing training progress
    """
    train_sizes = metrics["learning_curves"]["train_sizes"]
    train_scores = metrics["learning_curves"]["train_scores"]
    val_scores = metrics["learning_curves"]["val_scores"]

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training score")
    plt.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1
    )
    plt.plot(train_sizes, val_mean, label="Cross-validation score")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "learning_curves.png"))
    plt.close()


def save_results(
    clf,
    y_test,
    y_pred,
    weights_by_property,
    property_names,
    results_directory,
    metrics,
):
    """
    Save analysis results and visualizations
    """
    os.makedirs(results_directory, exist_ok=True)

    # Save training metrics
    with open(os.path.join(results_directory, "training_metrics.txt"), "w") as f:
        f.write("Training Metrics:\n")
        f.write(f"Training Time: {metrics['training_time']:.2f} seconds\n\n")
        f.write("Training Set Metrics:\n")
        f.write(f"Accuracy: {metrics['train_accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['train_f1']:.4f}\n\n")
        f.write("Validation Set Metrics:\n")
        f.write(f"Accuracy: {metrics['val_accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['val_f1']:.4f}\n\n")
        f.write("Test Set Metrics:\n")
        f.write(f"Accuracy: {metrics['test_accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['test_f1']:.4f}\n\n")

    # Plot learning curves
    plot_learning_curves(metrics, results_directory)

    # Save classification report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(results_directory, "classification_report.txt"), "w") as f:
        f.write(report)

    # Save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true", labels=unique_labels)
    sns.heatmap(
        cm, annot=True, fmt=".2f", xticklabels=unique_labels, yticklabels=unique_labels
    )
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "confusion_matrix.png"))
    plt.close()

    with open(os.path.join(results_directory, "important_dimensions.txt"), "w") as f:
        for property_name in property_names:
            f.write(f"\nTop dimensions for {property_name}:\n")
            dims = weights_by_property[property_name]["dimensions"]
            weights = weights_by_property[property_name]["weights"]
            for dim, weight in zip(dims, weights):
                f.write(f"Dimension {dim}: {weight:.4f}\n")

    weights_df = pd.DataFrame(clf.coef_, index=property_names)
    weights_df.to_csv(os.path.join(results_directory, "classifier_weights.csv"))

    with open(os.path.join(results_directory, "property_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)


def create_weight_heatmap(clf, property_names, results_directory):
    """
    Create heatmap showing the importance of each dimension for each property
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        clf.coef_,
        cmap="RdBu",
        center=0,
        xticklabels=range(clf.coef_.shape[1]),
        yticklabels=property_names,
    )
    plt.title("Dimension Importance by Linguistic Property")
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Linguistic Properties")
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "weight_heatmap.png"))
    plt.close()



def main():
    # Get embedding filepaths
    embedding_filepaths = get_embeddings_filepaths(model_name=MODEL)

    # Create results directory
    results_directory = os.path.join("results", MODEL, "linguistic_property_analysis")
    os.makedirs(results_directory, exist_ok=True)

    # Load and process data
    print("Loading and processing embeddings...")
    X, y, property_names = load_and_process_embeddings(embedding_filepaths)

    # Train classifier
    print("Training classifier...")
    clf, X_test, y_test, y_pred, metrics = train_property_classifier(X, y)

    # Analyze weights
    weights_by_property = analyze_classifier_weights(clf, property_names)

    # Save results
    print("Saving results...")
    save_results(
        clf,
        X_test,
        y_test,
        y_pred,
        weights_by_property,
        property_names,
        results_directory,
        metrics,
    )

    # Create weight heatmap
    create_weight_heatmap(clf, property_names, results_directory)

    print(f"Analysis complete. Results saved in {results_directory}")
    print("\nFinal Metrics:")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['test_f1']:.4f}")


if __name__ == "__main__":
    main()
