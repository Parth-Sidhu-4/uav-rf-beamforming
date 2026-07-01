import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

LR = 1e-3
BATCH_SIZE = 256

def positional_encoding(x, L=6):
    # x: [batch, 3]  ->  output: [batch, 3*(1 + 2L)] = [batch, 39]
    encoded = [x]
    for i in range(L):
        encoded.append(torch.sin((2.0 ** i) * torch.pi * x))
        encoded.append(torch.cos((2.0 ** i) * torch.pi * x))
    return torch.cat(encoded, dim=-1)

class ShadowNet(nn.Module):
    def __init__(self):
        super(ShadowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(39, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        x_enc = positional_encoding(x)
        return self.net(x_enc)

def train_model(model, train_loader, val_loader, epochs, patience, save_path=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (Best Val Loss: {best_val_loss:.6f})")
                break
                
    return best_val_loss

def main():
    dataset_path = "dataset_shadow_30k.npz"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return
        
    print("Loading dataset...")
    data = np.load(dataset_path)
    inputs = data['inputs']
    labels = data['labels']
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    N = inputs.shape[0]
    indices = np.random.permutation(N)
    split_idx = int(N * 0.8)
    
    X_train = torch.tensor(inputs[indices[:split_idx]], dtype=torch.float32)
    y_train = torch.tensor(labels[indices[:split_idx]], dtype=torch.float32)
    
    X_val = torch.tensor(inputs[indices[split_idx:]], dtype=torch.float32)
    y_val = torch.tensor(labels[indices[split_idx:]], dtype=torch.float32)
    
    print(f"--- Running Phase A PE Full Training (30k samples, 300 epochs) ---")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    final_model = ShadowNet()
    train_model(final_model, train_loader, val_loader, epochs=300, patience=50, save_path="shadow_net_pe.pt")
    print("Full training complete. Saved to shadow_net_pe.pt")

if __name__ == '__main__':
    main()
