import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from src.utils import *

from src.constants import *


def load_data_for_property(embeddings_csv):
    """Load and prepare data for binary classification"""
    embeddings_df = read_embeddings_df(embeddings_csv)

    # Create binary classification dataset
    sentence1_df = pd.DataFrame(
        {"embedding": embeddings_df["Sentence1_embedding"].tolist(), "label": 0}
    )
    sentence2_df = pd.DataFrame(
        {"embedding": embeddings_df["Sentence2_embedding"].tolist(), "label": 1}
    )
    df = pd.concat([sentence1_df, sentence2_df], ignore_index=True)

    X = np.array(df["embedding"].tolist())
    y = np.array(df["label"].tolist())

    # Split into train, val, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train logistic classifier and return accuracies"""
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    # val_accuracy = accuracy_score(y_val, clf.predict(X_val))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))

    return test_accuracy


def get_subset_data(X, selected_dims):
    """Get data subset using only selected dimensions"""
    return X[:, selected_dims]


def evaluate_property(embeddings_csv, other_property_csvs):
    """Run all evaluations for a linguistic property"""
    # Load data
    X_train, X_test, y_train, y_test = load_data_for_property(embeddings_csv)

    # Load EDI scores
    results_dir = get_results_directory(embeddings_csv, "edi_scores", model_name=MODEL)
    edi_scores_df = pd.read_csv(os.path.join(results_dir, "edi_score.csv"))

    # Get evaluation directory
    eval_dir = get_results_directory(embeddings_csv, "evaluation", model_name=MODEL)
    os.makedirs(eval_dir, exist_ok=True)

    results = {"baseline": [], "incremental": [], "low_edi": [], "cross_property": []}

    # 1. Baseline (all dimensions)
    print("Computing baseline...")
    baseline_test_acc = train_and_evaluate(X_train, X_test, y_train, y_test)
    results["baseline"] = baseline_test_acc

    # 2. Incremental evaluation with high EDI scores
    print("Running incremental evaluation...")
    top_dimensions = edi_scores_df.sort_values("edi_score", ascending=False)[
        "dimension"
    ].values

    for n_dims in tqdm(range(1, len(top_dimensions) + 1)):
        selected_dims = top_dimensions[:n_dims]

        # Get data subset
        X_train_subset = get_subset_data(X_train, selected_dims)
        X_test_subset = get_subset_data(X_test, selected_dims)

        test_acc = train_and_evaluate(X_train_subset, X_test_subset, y_train, y_test)

        results["incremental"].append(
            {
                "n_dimensions": n_dims,
                "test_accuracy": test_acc,
            }
        )

        # Stop if we reach close to baseline accuracy
        if test_acc >= baseline_test_acc * 0.95:
            break

    # 3. Evaluation with low EDI scores
    print("Evaluating low EDI score dimensions...")
    bottom_dimensions = edi_scores_df.sort_values("edi_score")["dimension"].values[:100]
    X_train_bottom = get_subset_data(X_train, bottom_dimensions)
    X_test_bottom = get_subset_data(X_test, bottom_dimensions)

    low_edi_test_acc = train_and_evaluate(
        X_train_bottom, X_test_bottom, y_train, y_test
    )
    results["low_edi"] = low_edi_test_acc

    # 4. Cross-property evaluation
    results["cross_property"] = []  

    if other_property_csvs:
        print("Running cross-property evaluations...")
        # Use top 25 dimensions for cross-property evaluation
        N_CROSS_DIMS = 25

        for other_csv in other_property_csvs:
            property_name = get_linguistic_property(other_csv)
            print(f"Evaluating against {property_name}...")

            # Load other property's EDI scores to get its top dimensions
            other_results_dir = get_results_directory(
                other_csv, "edi_scores", model_name=MODEL
            )
            other_edi_scores_df = pd.read_csv(
                os.path.join(other_results_dir, "edi_score.csv")
            )

            # Get top dimensions from other property
            other_top_dims = other_edi_scores_df.sort_values(
                "edi_score", ascending=False
            )["dimension"].values[:N_CROSS_DIMS]

            # Use other property's top dimensions on current property's data
            X_train_cross = get_subset_data(X_train, other_top_dims)
            X_test_cross = get_subset_data(X_test, other_top_dims)

            # Train and evaluate using current property's labels
            test_accuracy = train_and_evaluate(
                X_train_cross, X_test_cross, y_train, y_test
            )

            results["cross_property"].append(
                {"property": property_name, "accuracy": test_accuracy}
            )

    # Get property name for plot title
    current_property = get_linguistic_property(embeddings_csv)

    # Plot results with property name
    plot_results(results, eval_dir, current_property, best_cross_only=False)
    plot_results(results, eval_dir, current_property, best_cross_only=True)
    save_results(results, eval_dir)

    return results


def plot_results(results, eval_dir, property_name, best_cross_only=False):
    """Create visualization plots"""
    # Make figure wider
    plt.figure(figsize=(15, 8))

    # Plot incremental results with darker line
    incremental_df = pd.DataFrame(results["incremental"])
    if len(incremental_df) == 1:
        # If only one dimension, plot a single point
        plt.scatter(
            incremental_df["n_dimensions"],
            incremental_df["test_accuracy"],
            color="#1f77b4",  # Darker blue
            s=100,  # Size of the point
            label="High EDI Score Dimensions",
            zorder=3,
        )
    else:
        plt.plot(
            incremental_df["n_dimensions"],
            incremental_df["test_accuracy"],
            label="High EDI Score Dimensions",
            linewidth=2.5,
            color="#1f77b4",
        ) 

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.axhline(
        y=results["baseline"],
        color="#2ca02c",  
        linestyle="--",
        label="Baseline (All Dimensions)",
        linewidth=2.5,
    )

    plt.axhline(
        y=results["low_edi"],
        color="#d62728", 
        linestyle="--",
        label="Low EDI Score Dimensions",
        linewidth=2.5,
    )

    # Plot cross-property accuracies
    colors = plt.cm.Dark2(np.linspace(0, 1, len(results["cross_property"])))

    best_cross_result = None
    best_acc = float("-inf")
    control_result = None

    # First pass - find control and best non-control result
    for cross_result in results["cross_property"]:
        if cross_result["property"].lower() == "control":
            control_result = cross_result
        elif cross_result["accuracy"] > best_acc:
            best_acc = cross_result["accuracy"]
            best_cross_result = cross_result

    # Second pass - plot results
    if best_cross_only:
        # Plot only control and best non-control result
        if control_result:
            plt.axhline(
                y=control_result["accuracy"],
                color="#7D7C7C",  # Grey
                linestyle="--",
                label=f"Control Top Dimensions",
                linewidth=2.5,
            )

        if best_cross_result:
            plt.axhline(
                y=best_cross_result["accuracy"],
                color="#743cd4",  # Purple
                linestyle=":",
                label=f"Best Cross-Property: {best_cross_result['property']}",
                linewidth=2.5,
            )
    else:
        # Plot all results
        for i, cross_result in enumerate(results["cross_property"]):
            if cross_result["property"].lower() == "control":
                style = "--"
                color = "#d62728"  # Red
                label = "Control Top Dimensions"
            else:
                style = ":"
                color = colors[i]
                label = f"Cross-Property: {cross_result['property']}"

            plt.axhline(
                y=cross_result["accuracy"],
                color=color,
                linestyle=style,
                label=label,
                linewidth=2.5,
            )

    plt.xlabel("Number of Dimensions", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title(f"EDI Score Evaluation Results for {property_name.title()}", fontsize=14)

    # Make legend more compact and place it to the right
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=10,
        frameon=True,
        framealpha=1,
        edgecolor="black",
    )

    plt.grid(True, alpha=0.3)
    plt.subplots_adjust(right=0.85)  

    fname = (
        "evaluation_results.png"
        if best_cross_only
        else "all_props_evaluation_results.png"
    )

    plt.savefig(os.path.join(eval_dir, fname), bbox_inches="tight", dpi=300)
    plt.close()


def save_results(results, eval_dir):
    """Save numerical results to file"""
    with open(os.path.join(eval_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Baseline Accuracy: {results['baseline']:.4f}\n")
        f.write(f"Low EDI Score Dimensions Accuracy: {results['low_edi']:.4f}\n")

        f.write("\nCross-Property Accuracies:\n")
        for cross_result in results["cross_property"]:
            f.write(f"{cross_result['property']}: {cross_result['accuracy']:.4f}\n")

        f.write("\nIncremental Results:\n")
        for result in results["incremental"]:
            f.write(
                f"Dimensions: {result['n_dimensions']}, "
                f"Test Accuracy: {result['test_accuracy']:.4f}\n"
            )


if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths(model_name=MODEL)

    # For each property, evaluate against all other properties
    for i, primary_property in enumerate(tqdm(embedding_filepaths)):
        print(f"\nEvaluating property: {os.path.basename(primary_property)}")

        # Get all other properties for cross-property evaluation
        other_properties = [p for p in embedding_filepaths if p != primary_property]

        evaluate_property(primary_property, other_properties)

    print("\nEvaluation complete.")
