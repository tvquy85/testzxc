import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from mtan_model import MTAN

class TimeframeDataset(Dataset):
    def __init__(self, data, labels, num_timeframes):
        self.data = data  # shape: (num_samples, num_timeframes, seq_len, input_size)
        self.labels = labels  # shape: (num_samples,)
        self.num_timeframes = num_timeframes
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, num_epochs, device, optimizer=None):
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    patience = 5  # Early stopping patience
    no_improve = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs['prediction'], labels)
            
            # Thêm L1 regularization
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs['prediction'].max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs['prediction'], labels)
                
                val_loss += loss.item()
                _, predicted = outputs['prediction'].max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    input_size = 5  # OHLCV
    hidden_size = 64
    num_timeframes = 4  # D1, H4, H1, M15
    
    # Create model
    model = MTAN(input_size, hidden_size, num_timeframes).to(device)
    
    # Generate dummy data for example
    num_samples = 1000
    seq_len = 100
    data = torch.randn(num_samples, num_timeframes, seq_len, input_size)
    labels = torch.randint(0, 3, (num_samples,))  # 0: down, 1: sideway, 2: up
    
    # Split data
    train_size = int(0.8 * num_samples)
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    val_data = data[train_size:]
    val_labels = labels[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = TimeframeDataset(train_data, train_labels, num_timeframes)
    val_dataset = TimeframeDataset(val_data, val_labels, num_timeframes)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=10, device=device)

if __name__ == "__main__":
    main() 