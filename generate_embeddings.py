import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import os
from utils import get_dataset_filepaths
from tqdm import tqdm

def get_embedding(text):
  inputs = tokenizer(text, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)
  return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


if __name__ == "__main__":

    dataset_filepaths = get_dataset_filepaths()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    for dataset_csv in tqdm(dataset_filepaths):
       
       df = pd.read_csv(dataset_csv)
       
       df['Sentence1_embedding'] = df['Sentence1'].apply(get_embedding)
       df['Sentence2_embedding'] = df['Sentence2'].apply(get_embedding)
       
       df.to_csv(os.path.join("embeddings", os.path.basename(dataset_csv).replace(".csv", "_embeddings.csv")))


    print("Embedding Generation Complete")