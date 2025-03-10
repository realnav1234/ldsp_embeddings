import os
import pandas as pd
import random


if __name__ == "__main__":

    dataset_folder = "./datasets"

    sentences = []

    for file_name in os.listdir(dataset_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(dataset_folder, file_name)
            df = pd.read_csv(file_path)
            sentences.extend(df["Sentence1"].tolist())
            sentences.extend(df["Sentence2"].tolist())

    control_group = []
    for _ in range(1000):
        sentence1 = random.choice(sentences)
        sentence2 = random.choice(sentences)
        control_group.append([sentence1, sentence2])

    control_df = pd.DataFrame(control_group, columns=["Sentence1", "Sentence2"])
    control_df.to_csv(os.path.join(dataset_folder, "control_ldsps.csv"), index=False)
