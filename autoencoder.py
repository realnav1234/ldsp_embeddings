import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

class AutoencoderWithSkip(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        # Initialize skip connection weight as ones (identity)
        # self.skip_weight = nn.Parameter(torch.ones(input_size))
        
    def forward(self, x):
        # Learned transformation path
        encoded = self.encoder(x)
        transformed = self.decoder(encoded)
        
        # Skip connection path (element-wise multiplication)
        # skip_connection = x * self.skip_weight
        
        # Combine both paths
        return transformed + x

def train_autoencoder(sent1_embeddings, sent2_embeddings, hidden_size=16, epochs=100):
    """Train a neural network to project sentence1 embeddings to sentence2 embeddings"""
    input_size = sent1_embeddings.shape[1]
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(sent1_embeddings)
    y = torch.FloatTensor(sent2_embeddings)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define model with skip connection
    model = AutoencoderWithSkip(input_size, hidden_size)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Track losses
    epoch_losses = []
    
    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        # Record average loss for this epoch
        avg_epoch_loss = total_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
            
    return model, epoch_losses

def plot_training_curve(losses, results_directory):
    """Plot and save the training curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add final loss value annotation
    final_loss = losses[-1]
    plt.annotate(f'Final Loss: {final_loss:.6f}', 
                xy=(len(losses)-1, final_loss),
                xytext=(len(losses)-20, final_loss*1.2),
                arrowprops=dict(facecolor='black', shrink=0.05),
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, 'training_curve.png'))
    plt.close()

if __name__ == "__main__": 
    embeddings_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embeddings_filepaths):
        print(f"\nProcessing {embeddings_csv}")
        embeddings_df = read_embeddings_df(embeddings_csv)
        
        # Get embeddings as numpy arrays
        sent1_embeddings = np.array(embeddings_df['Sentence1_embedding'].tolist())
        sent2_embeddings = np.array(embeddings_df['Sentence2_embedding'].tolist())
        
        # Train autoencoder
        results_directory = get_results_directory(embeddings_csv, "autoencoder")
        model, losses = train_autoencoder(sent1_embeddings, sent2_embeddings)
        
        # Plot training curve
        plot_training_curve(losses, results_directory)
        
        # Save model
        model_path = os.path.join(results_directory, "autoencoder.pt")
        torch.save(model.state_dict(), model_path)
        
        # Save loss values
        loss_path = os.path.join(results_directory, "training_losses.txt")
        with open(loss_path, 'w') as f:
            f.write("Epoch,Loss\n")
            for epoch, loss in enumerate(losses):
                f.write(f"{epoch},{loss}\n")
        
        print(f"Model and training curves saved in {results_directory}")

    print("\nAutoencoder training complete")
    