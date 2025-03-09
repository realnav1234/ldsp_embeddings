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
├── embeddings/           # The generated embeddings for each linguistic property and language model.
├── evaluation scripts /  # Core scripts used for statistical analysis
│   ├── dataset.py        # Code for dataset preprocessing
│   ├── analysis.py       # Scripts for statistical analysis
│   ├── visualization.py  # Tools for plotting and result visualization
├── results/              # Stores experimental results and figures
├── README.md             # Project documentation
├── run.sh                # Used to generate EDI scores for one of the built-in models.
```
