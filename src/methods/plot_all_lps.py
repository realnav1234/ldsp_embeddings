import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.utils import *
from src.constants import MODEL


def create_edi_comparison_table(embedding_dim):
    # Define the linguistic properties and their file paths
    properties = {
        "Control": f"results/{MODEL}/control/edi_scores/edi_score.csv",
        "Negation": f"results/{MODEL}/negation/edi_scores/edi_score.csv",
        "Intensifier": f"results/{MODEL}/intensifier/edi_scores/edi_score.csv",
        "Tense": f"results/{MODEL}/tense/edi_scores/edi_score.csv",
        "Voice": f"results/{MODEL}/voice/edi_scores/edi_score.csv",
        "Polarity": f"results/{MODEL}/polarity/edi_scores/edi_score.csv",
        "Quantity": f"results/{MODEL}/quantity/edi_scores/edi_score.csv",
        "Factuality": f"results/{MODEL}/factuality/edi_scores/edi_score.csv",
        "Definiteness": f"results/{MODEL}/definiteness/edi_scores/edi_score.csv",
        "Synonym": f"results/{MODEL}/synonym/edi_scores/edi_score.csv",
    }

    # Initialize an empty DataFrame to store all scores
    all_scores = pd.DataFrame(columns=range(embedding_dim))

    # Read each file and add it to the DataFrame
    for property_name, filepath in properties.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Sort by dimension to ensure correct ordering
            df = df.sort_values("dimension")
            # Set property name as the row name for these scores
            all_scores.loc[property_name] = df["edi_score"].values

    # Rename columns to be dimension numbers
    all_scores.columns = [f"Dim_{i}" for i in range(len(all_scores.columns))]

    # Save to CSV
    output_path = f"results/{MODEL}/combined_edi_scores.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_scores.to_csv(output_path)

    return all_scores


def create_highlighted_heatmap(df):
    plt.figure(figsize=(20, 8))

    # Create mask for missing values
    mask = df.isna()

    plt.rcParams.update({"font.size": 22})

    # Create the base heatmap
    ax = sns.heatmap(
        df,
        cmap="YlOrRd",
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "EDI Score"},
        mask=mask,
    )

    ax.set_title("EDI Scores Across Dimensions", fontsize=44, fontweight="bold")
    plt.xlabel("Dimensions (0-767)")
    plt.tight_layout()

    plt.savefig(f"results/{MODEL}/edi_scores_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_colored_top_values(df, threshold=0.5):
    plt.figure(figsize=(20, 14))

    colors = [
        "#FF5252",  # Red
        "#FF7B52",  # Red-Orange
        "#FFB347",  # Orange
        "#FFEB3B",  # Yellow
        "#9CCC65",  # Yellow-Green
        "#66BB6A",  # Green
        "#4DB6AC",  # Blue-Green
        "#4FC3F7",  # Light Blue
        "#5C6BC0",  # Blue
        "#7E57C2",  # Blue-Violet
        "#D81B60",  # Pink-Violet
    ]

    plt.imshow(np.zeros_like(df), cmap="binary", aspect="auto")

    # For each row (property), highlight values above threshold
    for idx, (property_name, row) in enumerate(df.iterrows()):
        high_value_idx = row[row > threshold].index
        high_values = row[row > threshold]

        x_positions = [int(col.split("_")[1]) for col in high_value_idx]
        y_positions = [idx] * len(x_positions)

        plt.scatter(
            x_positions,
            y_positions,
            c=[colors[idx]],
            s=100,
            label=f"{property_name} ({len(high_values)} dims)",
        )

    # Customize the plot
    plt.yticks(range(len(df)), df.index)
    plt.xlabel("Dimensions (0-767)")
    plt.title(f"EDI Scores Above {threshold} for Each Linguistic Property")

    plt.tight_layout()
    plt.savefig(
        f"results/{MODEL}/colored_top_edi_scores.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_colored_grid(df, threshold=0.8):
    plt.figure(figsize=(20, 12))

    colors = sns.color_palette("husl", n_colors=len(df))

    plt.pcolor(
        np.zeros((len(df), len(df.columns))),
        cmap="binary",
        edgecolors="lightgray",
        linewidths=0.1,
    )

    # For each row (property), highlight values above threshold
    for idx, (property_name, row) in enumerate(df.iterrows()):
        high_value_idx = row[row > threshold].index
        high_values = row[row > threshold]

        x_positions = [int(col.split("_")[1]) for col in high_value_idx]

        for x, value in zip(x_positions, high_values):
            alpha = min(1.0, value / threshold) 
            plt.fill(
                [x, x + 1, x + 1, x],
                [idx, idx, idx + 1, idx + 1],
                color=colors[idx],
                alpha=alpha,
            )

    plt.yticks(np.arange(0.5, len(df), 1), df.index, fontsize=20)
    plt.xlabel("Dimensions (0-767)", fontsize=20)
    plt.title(f"EDI Scores Above {threshold} for Each Linguistic Property", fontsize=20)

    plt.tight_layout()
    plt.savefig(
        f"results/{MODEL}/colored_grid_edi_scores.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    df = create_edi_comparison_table(768)
    print(f"Created table with shape: {df.shape}")
    print("\nFirst few columns of the table:")
    print(df.iloc[:, :5])

    create_highlighted_heatmap(df)
    print(f"Created heatmap at 'results/{MODEL}/edi_scores_heatmap.png'")

    create_colored_grid(df, threshold=0.8)
    print(f"Created visualization at 'results/{MODEL}/colored_grid_edi_scores.png'")
