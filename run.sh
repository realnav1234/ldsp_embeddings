#!/bin/bash

# Change the model name in constants.py
MODEL_NAME="gpt"  # Replace with the desired model name
sed -i '' "s/MODEL = .*/MODEL = \"$MODEL_NAME\"/" constants.py

# Run the Python scripts in the specified order
python3 ./evaluation_scripts/t_test.py
python3 ./evaluation_scripts/mutual_information.py
python3 ./evaluation_scripts/recursive_feature_elimination.py
python3 ./evaluation_scripts/generate_combined_graph.py
python3 ./evaluation_scripts/edi_score.py
python3 ./evaluation_scripts/plot_all_lps.py
python3 ./evaluation_scripts/evaluate_edi_scores.py
python3 ./evaluation_scripts/linguistic_property_classifier.py