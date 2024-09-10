from tqdm import tqdm
import pandas as pd
import numpy as np

import os
import torch
import torch.nn as nn
import torch.optim as optim

def create_network(input_size, hidden_layers_sizes, activations, output_size):

    # Dictionary to map activation function names to their corresponding PyTorch activation functions
    activation_dict = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(),
        'softmax': nn.Softmax(dim=-1),
        'log_softmax': nn.LogSoftmax(dim=-1),
        'linear': nn.Identity()  # No activation, just the identity mapping
    }
    
    layers = []
    # Input layer
    in_size = input_size
    
    for i, hidden_size in enumerate(hidden_layers_sizes):
        # Fully connected layer
        layers.append(nn.Linear(in_size, hidden_size))
        # Corresponding activation function
        if activations[i] in activation_dict:
            layers.append(activation_dict[activations[i]])
        else:
            raise ValueError(f"Unknown activation function: {activations[i]}")
        # Update input size for next layer
        in_size = hidden_size
    
    # Output layer (linear activation by default for regression)
    layers.append(nn.Linear(in_size, output_size))
    # No activation for the output layer, as it's for regression (linear output)

    # Combine all layers into a sequential model
    model = nn.Sequential(*layers)
    
    return model

def to_dataloader(
    X: pd.DataFrame,
    Y: pd.Series,
    batch_size: int = 32,
    shuffle = True):

    X_tensor = torch.tensor(X.values, dtype = torch.float32)
    Y_tensor = torch.tensor(Y, dtype = torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    return dataloader

def train_network(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    epochs: int, 
    learning_rate: float = 0.001, 
    device: str = 'check', 
    patience: int = 10,
    start_from_epoch: int = 10):

    history = {'train_loss': [], 'val_loss': []}

    if device == 'check':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    criterion = nn.MSELoss()  # Assuming a regression task. Use nn.CrossEntropyLoss for classification.
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc = f"Epoch {epoch+1}/{epochs}")
        
        for inputs, targets in train_progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view_as(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_progress_bar.set_postfix(train_loss=running_loss/len(train_loader))
        
        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_network(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Check for early stopping
        if epoch >= start_from_epoch:
            if epoch == start_from_epoch:
                torch.save(model.state_dict(), "./temp.pt")  # Save the current model the first time
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience counter
                torch.save(model.state_dict(), "./temp.pt")  # Save the best model
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered")
                model.load_state_dict(torch.load("./temp.pt"))  # Restore the best model
                # Delete the temporary file
                try:
                    os.remove("./temp.pt")
                except:
                    pass
                break

    return history, model

def evaluate_network(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion, device: str = 'cpu'):

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        eval_progress_bar = tqdm(dataloader, desc = "Evaluation")
        for inputs, targets in eval_progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets = targets.view_as(outputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(dataloader)