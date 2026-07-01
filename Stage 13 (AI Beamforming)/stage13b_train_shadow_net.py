import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

LR = 1e-3
BATCH_SIZE = 256

class ShadowNet(nn.Module):
    def __init__(self):
        super(ShadowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        return self.net(x)

class AsymmetricPolarLoss(nn.Module):
    def __init__(self, penalty_multiplier):
        super().__init__()
        self.penalty = penalty_multiplier
        
    def forward(self, outputs, labels):
        mag_pred = torch.sigmoid(outputs[:, 0::2])
        phase_pred = outputs[:, 1::2]
        
        mag_true = labels[:, 0::2]
        phase_true = labels[:, 1::2]
        
        mag_err = mag_true - mag_pred
        is_transition = (mag_true > 0.2) & (mag_true < 0.8)
        is_underpred = mag_err > 0
        
        weight = torch.ones_like(mag_true)
        weight[is_transition & is_underpred] *= self.penalty
        
        mag_loss = torch.mean(weight * (mag_err ** 2))
        phase_loss = torch.mean(mag_true * (1.0 - torch.cos(phase_pred - phase_true)))
        
        return mag_loss + phase_loss

def train_model(model, train_loader, val_loader, penalty, epochs, patience, save_path=None):
    criterion = AsymmetricPolarLoss(penalty_multiplier=penalty)
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
    dataset_path = "dataset_shadow_100k_polar.npz"
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
    
    # 1. Pilot Sweep on 10k subset
    print("\n--- Running Pilot Hyperparameter Sweep (10k samples, 30 epochs) ---")
    pilot_idx_t = np.random.permutation(X_train.shape[0])[:8000]
    pilot_idx_v = np.random.permutation(X_val.shape[0])[:2000]
    
    p_train_loader = DataLoader(TensorDataset(X_train[pilot_idx_t], y_train[pilot_idx_t]), batch_size=BATCH_SIZE, shuffle=True)
    p_val_loader = DataLoader(TensorDataset(X_val[pilot_idx_v], y_val[pilot_idx_v]), batch_size=BATCH_SIZE, shuffle=False)
    
    penalties = [5.0, 10.0, 20.0]
    best_penalty = 10.0
    best_pilot_loss = float('inf')
    
    for p in penalties:
        print(f"\nEvaluating Penalty Multiplier: {p}x")
        model = ShadowNet()
        val_loss = train_model(model, p_train_loader, p_val_loader, penalty=p, epochs=30, patience=30)
        print(f"Final Val Loss for {p}x: {val_loss:.6f}")
        if val_loss < best_pilot_loss:
            best_pilot_loss = val_loss
            best_penalty = p
            
    print(f"\n*** Selected Optimal Penalty: {best_penalty}x ***\n")
    
    # 2. Full Training
    print(f"--- Running Full Training (100k samples, up to 500 epochs) ---")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    final_model = ShadowNet()
    train_model(final_model, train_loader, val_loader, penalty=best_penalty, epochs=500, patience=50, save_path="shadow_net_polar.pt")
    print("Full training complete. Saved to shadow_net_polar.pt")

if __name__ == '__main__':
    main()
