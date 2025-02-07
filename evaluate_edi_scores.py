import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import *

def load_data_for_property(embeddings_csv):
    """Load and prepare data for binary classification"""
    embeddings_df = read_embeddings_df(embeddings_csv)
    
    # Create binary classification dataset
    sentence1_df = pd.DataFrame({'embedding': embeddings_df['Sentence1_embedding'].tolist(), 'label': 0})
    sentence2_df = pd.DataFrame({'embedding': embeddings_df['Sentence2_embedding'].tolist(), 'label': 1})
    df = pd.concat([sentence1_df, sentence2_df], ignore_index=True)
    
    X = np.array(df['embedding'].tolist())
    y = np.array(df['label'].tolist())
    
    # Split into train, val, test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train logistic classifier and return accuracies"""
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    val_accuracy = accuracy_score(y_val, clf.predict(X_val))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))
    
    return val_accuracy, test_accuracy

def get_subset_data(X, selected_dims):
    """Get data subset using only selected dimensions"""
    return X[:, selected_dims]

def evaluate_property(embeddings_csv, other_property_csvs):
    """Run all evaluations for a linguistic property"""
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_for_property(embeddings_csv)
    
    # Load EDI scores
    results_dir = get_results_directory(embeddings_csv, "edi_scores")
    edi_scores_df = pd.read_csv(os.path.join(results_dir, "edi_score.csv"))
    
    # Get evaluation directory
    eval_dir = get_results_directory(embeddings_csv, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    results = {
        'baseline': [],
        'incremental': [],
        'low_edi': [],
        'cross_property': []
    }
    
    # 1. Baseline (all dimensions)
    print("Computing baseline...")
    baseline_val_acc, baseline_test_acc = train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test)
    results['baseline'] = baseline_test_acc
    
    # 2. Incremental evaluation with high EDI scores
    print("Running incremental evaluation...")
    top_dimensions = edi_scores_df.sort_values('EDI Score', ascending=False)['Dimension'].values
    
    for n_dims in tqdm(range(1, len(top_dimensions) + 1)):
        selected_dims = top_dimensions[:n_dims]
        
        # Get data subset
        X_train_subset = get_subset_data(X_train, selected_dims)
        X_val_subset = get_subset_data(X_val, selected_dims)
        X_test_subset = get_subset_data(X_test, selected_dims)
        
        val_acc, test_acc = train_and_evaluate(
            X_train_subset, X_val_subset, X_test_subset, 
            y_train, y_val, y_test)
        
        results['incremental'].append({
            'n_dimensions': n_dims,
            'test_accuracy': test_acc,
            'val_accuracy': val_acc
        })
        
        # Stop if we reach close to baseline accuracy
        if val_acc >= baseline_val_acc * 0.95:
            break
    
    # 3. Evaluation with low EDI scores
    print("Evaluating low EDI score dimensions...")
    bottom_dimensions = edi_scores_df.sort_values('EDI Score')['Dimension'].values[:100]
    X_train_bottom = get_subset_data(X_train, bottom_dimensions)
    X_val_bottom = get_subset_data(X_val, bottom_dimensions)
    X_test_bottom = get_subset_data(X_test, bottom_dimensions)
    
    _, low_edi_test_acc = train_and_evaluate(
        X_train_bottom, X_val_bottom, X_test_bottom,
        y_train, y_val, y_test)
    results['low_edi'] = low_edi_test_acc
    
    # 4. Cross-property evaluation
    results['cross_property'] = []  # Change to list to store multiple results
    
    if other_property_csvs:
        print("Running cross-property evaluations...")
        # Get number of dimensions we needed to reach 95% baseline accuracy
        n_dims_needed = len(results['incremental'])
        top_dims = top_dimensions[:n_dims_needed]
        
        for other_csv in other_property_csvs:

            if "control" in other_csv.lower() or "synonym" in other_csv.lower(): 
                continue

            property_name = os.path.basename(other_csv)[:-21]  # Get property name from filename
            print(f"Evaluating against {property_name}...")
            
            X_train_other, X_val_other, X_test_other, y_train_other, y_val_other, y_test_other = \
                load_data_for_property(other_csv)
            
            X_train_cross = get_subset_data(X_train_other, top_dims)
            X_val_cross = get_subset_data(X_val_other, top_dims)
            X_test_cross = get_subset_data(X_test_other, top_dims)
            
            _, cross_property_test_acc = train_and_evaluate(
                X_train_cross, X_val_cross, X_test_cross,
                y_train_other, y_val_other, y_test_other)
            
            results['cross_property'].append({
                'property': property_name,
                'accuracy': cross_property_test_acc
            })
    
    # Get property name for plot title
    current_property = os.path.basename(embeddings_csv)[:-21]
    
    # Plot results with property name
    plot_results(results, eval_dir, current_property)
    save_results(results, eval_dir)
    
    return results

def plot_results(results, eval_dir, property_name):
    """Create visualization plots"""
    # Make figure wider
    plt.figure(figsize=(15, 8))
    
    # Plot incremental results with darker line
    incremental_df = pd.DataFrame(results['incremental'])
    plt.plot(incremental_df['n_dimensions'], 
             incremental_df['test_accuracy'], 
             label='High EDI Score Dimensions',
             linewidth=2.5,
             color='#1f77b4')  # Darker blue
    
    # Plot baseline with darker line
    plt.axhline(y=results['baseline'], 
                color='#d62728',  # Darker red
                linestyle='--', 
                label='Baseline (All Dimensions)',
                linewidth=2.5)
    
    # Plot low EDI score accuracy with darker line
    plt.axhline(y=results['low_edi'], 
                color='#2ca02c',  # Darker green
                linestyle='--', 
                label='Low EDI Score Dimensions',
                linewidth=2.5)
    
    # Plot cross-property accuracies with darker colors
    # Use a different colormap with more distinct colors
    colors = plt.cm.Dark2(np.linspace(0, 1, len(results['cross_property'])))
    for i, cross_result in enumerate(results['cross_property']):
        plt.axhline(y=cross_result['accuracy'],
                   color=colors[i],
                   linestyle=':',
                   label=f"Cross-Property: {cross_result['property']}",
                   linewidth=2.5)
    
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title(f'EDI Score Evaluation Results for {property_name.title()}', fontsize=14)
    
    # Make legend more compact and place it to the right
    plt.legend(bbox_to_anchor=(1.02, 1), 
              loc='upper left',
              borderaxespad=0,
              fontsize=10,
              frameon=True,
              framealpha=1,
              edgecolor='black')
    
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff while keeping plot wide
    plt.subplots_adjust(right=0.85)  # Make more room for legend while keeping plot wide
    
    plt.savefig(os.path.join(eval_dir, 'all_props_evaluation_results.png'), 
                bbox_inches='tight',
                dpi=300)
    plt.close()

def save_results(results, eval_dir):
    """Save numerical results to file"""
    with open(os.path.join(eval_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Baseline Accuracy: {results['baseline']:.4f}\n")
        f.write(f"Low EDI Score Dimensions Accuracy: {results['low_edi']:.4f}\n")
        
        f.write("\nCross-Property Accuracies:\n")
        for cross_result in results['cross_property']:
            f.write(f"{cross_result['property']}: {cross_result['accuracy']:.4f}\n")
        
        f.write("\nIncremental Results:\n")
        for result in results['incremental']:
            f.write(f"Dimensions: {result['n_dimensions']}, "
                   f"Test Accuracy: {result['test_accuracy']:.4f}\n")

if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths()
    
    # For each property, evaluate against all other properties
    for i, primary_property in enumerate(tqdm(embedding_filepaths)):
        print(f"\nEvaluating property: {os.path.basename(primary_property)}")
        
        # Get all other properties for cross-property evaluation
        other_properties = [p for p in embedding_filepaths if p != primary_property]
        
        evaluate_property(primary_property, other_properties)
    
    print("\nEvaluation complete.") 