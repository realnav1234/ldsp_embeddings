import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from generate_embeddings import get_embedding

class AutoencoderWithSkip(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.act = nn.ReLU()
        self.decoder = nn.Linear(hidden_size, input_size)

        
    def forward(self, x):
        # Learned transformation path
        encoded = self.act(self.encoder(x))
        transformed = self.decoder(encoded)
        
        # Combine both paths
        return transformed + x

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_weight=0.01):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_weight = sparsity_weight
        
    def forward(self, x):
        encoded = self.encoder(x)
        activated = F.relu(encoded)
        decoded = self.decoder(activated)
        return decoded, activated
    
    def get_l1_loss(self, activations):
        return self.sparsity_weight * torch.mean(torch.abs(activations))

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

def train_sparse_autoencoder(diff_vectors, hidden_size=64, epochs=100, sparsity_weight=0.01):
    """Train a sparse autoencoder on difference vectors between sentence pairs"""
    input_size = diff_vectors.shape[1]
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(diff_vectors)
    dataset = TensorDataset(X, X)  # Autoencoder reconstructs its input
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define sparse autoencoder
    model = SparseAutoencoder(input_size, hidden_size, sparsity_weight)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Track losses
    epoch_losses = []
    
    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_l1_loss = 0
        batch_count = 0
        
        for batch_X, _ in loader:
            optimizer.zero_grad()
            reconstructed, activations = model(batch_X)
            
            # Compute reconstruction loss
            recon_loss = criterion(reconstructed, batch_X)
            
            # Compute L1 sparsity loss
            l1_loss = model.get_l1_loss(activations)
            
            # Total loss
            loss = recon_loss + l1_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()
            batch_count += 1
        
        # Record average losses for this epoch
        avg_epoch_loss = total_loss / batch_count
        avg_l1_loss = total_l1_loss / batch_count
        epoch_losses.append((avg_epoch_loss, avg_l1_loss))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Recon Loss: {avg_epoch_loss:.4f}, '
                  f'L1 Loss: {avg_l1_loss:.4f}')
            
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


def analyze_weights(model): 

    emb1 = torch.tensor(get_embedding("This is a house."))
    print(model.encoder(emb1))
    


    emb2 = torch.tensor(get_embedding("This is not a house."))
    print(model.encoder(emb2))

    enc_weights = model.encoder.weight
    enc_weights = torch.abs(enc_weights)
    # print(torch.min(enc_weights))
    # print(torch.topk(enc_weights, 20))
   



def analyze_latent_dimensions(model, diff_vectors, property_labels):
    """Analyze which latent dimensions correspond to specific linguistic properties"""
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(diff_vectors)
        _, activations = model(X)
        
    # Compute correlation between activations and property labels
    correlations = []
    for dim in range(activations.shape[1]):
        dim_activations = activations[:, dim].numpy()
        correlation = np.corrcoef(dim_activations, property_labels)[0, 1]
        correlations.append((dim, abs(correlation)))
    
    # Sort dimensions by correlation strength
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    return correlations

if __name__ == "__main__": 
    embeddings_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embeddings_filepaths):
        if "negation" not in embeddings_csv:
            continue

        print(f"\nProcessing {embeddings_csv}")
        embeddings_df = read_embeddings_df(embeddings_csv)
        
        # Get embeddings as numpy arrays
        sent1_embeddings = np.array(embeddings_df['Sentence1_embedding'].tolist())
        sent2_embeddings = np.array(embeddings_df['Sentence2_embedding'].tolist())
        
        # Train autoencoder
        results_directory = get_results_directory(embeddings_csv, "autoencoder")
        model, losses = train_autoencoder(sent1_embeddings, sent2_embeddings, hidden_size=1)

        analyze_weights(model)
        
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

        # Compute difference vectors (sentence with property - sentence without property)
        diff_vectors = sent2_embeddings - sent1_embeddings
        
        # Create property labels
        # For negation dataset: sent2 has negation, sent1 doesn't
        property_labels = np.zeros(len(diff_vectors))
        for i in range(len(property_labels)):
            # 1 for dimensions that encode the property (active in sent2)
            # -1 for dimensions that encode the absence of the property (active in sent1)
            property_labels[i] = 1
        
        # Train sparse autoencoder with more hidden dimensions to capture fine-grained features
        sparse_model, losses = train_sparse_autoencoder(
            diff_vectors, 
            hidden_size=64,  # Increased from 64
            epochs=100,       # Train longer
            sparsity_weight=0.1  # Reduced sparsity for finer feature detection
        )
        
        # Save sparse autoencoder model
        sparse_model_path = os.path.join(results_directory, "sparse_autoencoder.pt")
        torch.save(sparse_model.state_dict(), sparse_model_path)
        
        # Analyze dimensions
        correlations = analyze_latent_dimensions(sparse_model, diff_vectors, property_labels)
        
        # Save correlation results
        correlation_path = os.path.join(results_directory, "dimension_correlations.csv")
        with open(correlation_path, 'w') as f:
            f.write("Dimension,Correlation\n")
            for dim, corr in correlations:
                f.write(f"{dim},{corr:.6f}\n")
        
        # Print top correlated dimensions
        print("\nTop 10 dimensions correlated with negation:")
        for dim, corr in correlations[:10]:
            print(f"Dimension {dim}: correlation = {corr:.3f}")

        break

    print("\nAutoencoder training complete")
    