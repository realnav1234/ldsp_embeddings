# Disentangling Linguistic Features with Dimension-Wise Analysis of Vector Embeddings

## Overview
This repository contains the code and resources associated with the paper *Disentangling Linguistic Features with Dimension-Wise Analysis of Vector Embeddings*, authored by Saniya Karwa and Navpreet Singh from MIT. The project introduces a framework for analyzing vector embeddings in NLP models, specifically focusing on identifying embedding dimensions responsible for encoding linguistic properties.

## Key Contributions
- **LDSP-10 Dataset:** A curated dataset of linguistically distinct sentence pairs isolating key linguistic features such as synonymy, negation, tense, and polarity.
- **Embedding Dimension Impact (EDI) Score:** A novel metric for quantifying the importance of each embedding dimension in encoding linguistic properties.
- **Interpretability Insights:** Findings suggest that certain linguistic properties (e.g., negation, polarity) are robustly encoded in specific dimensions, whereas others (e.g., synonymy) are more diffusely represented.

## Repository Structure
```
├── datasets/             # Contains the LDSP-10 dataset
├── embeddings/                   # The generated embeddings for each linguistic property and language model.
├── src/
    ├── utils.py                               # Utility functions
    ├── constants.py                           # Constants used throughout the project
    ├── dataset/
        ├── generate_embeddings.py
        └── create_control_group.py
    ├── evaluations/                           # Core scripts used for statistical analysis
        ├── linguistic_property_classifier.py  # Classifier for linguistic properties
        └── evaluate_edi_scores.py  # Evaluation of EDI scores
    ├── methods/              # Contains various methods for analysis
        ├── mutual_information.py  # Mutual information calculations
        ├── recursive_feature_elimination.py  # RFE implementation
        ├── wilcoxon.py       # Wilcoxon test implementation
        ├── edi_score.py 
        ├── plot_all_lps.py 
        └──generate_combined_scores.py 
├── results/              # Stores experimental results and figures
└── run.sh                # Used to generate EDI scores for one of the built-in models.

