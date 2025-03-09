from utils import get_embeddings_filepaths
from constants import MODEL
import os

if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths(model_name=MODEL)

    for csv in embedding_filepaths:

        property_name = os.path.basename(csv).split("_")[0]
        print(property_name)