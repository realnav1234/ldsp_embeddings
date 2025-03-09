#!/bin/bash

# Change the model name in constants.py
MODEL_NAME="gpt"  # Replace with the desired model name
sed -i '' "s/MODEL = .*/MODEL = \"$MODEL_NAME\"/" constants.py

# Run the Python scripts in the specified order
# python3 t_test.py
# python3 mutual_information.py
# python3 recursive_feature_elimination.py
# python3 generate_combined_graph.py
python3 edi_score.py
python3 plot_all_lps.py
python3 evaluate_edi_scores.py
# python3 linguistic_property_classifier.py