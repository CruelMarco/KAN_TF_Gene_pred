#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:17:59 2024

@author: msolanki
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
import os
from kan import *
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# # Check if CUDA is available
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

#print(device)

# Load data
data_set_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp"
os.chdir(data_set_dir)

tf_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/tf.csv"
gene_exp_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/gene_exp.csv"

tf_csv = pd.read_csv(tf_data_dir, sep=",").drop(["Unnamed: 0"], axis=1)
gene_exp_csv = pd.read_csv(gene_exp_data_dir, sep=",").drop(["Unnamed: 0"], axis=1)

X_data = tf_csv.drop(["tissue_code"], axis=1)
y_data = gene_exp_csv.drop(["Unnamed:"], axis=1).iloc[:, 0]

# Dataset class
class TFGeneDataset(Dataset):
    def __init__(self, X , y):
        self.X = torch.tensor(X.values , dtype=torch.float32)
        self.y = torch.tensor(y.values , dtype = torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        feature = self.X[idx]
        target = self.y[idx]
        return feature , target

# Create datasets and loaders
dataset = TFGeneDataset(X_data, y_data)
train_set, val_set , test_set = random_split(dataset , [0.8 , 0.1 , 0.1])

train_loader = DataLoader(train_set , shuffle=True , batch_size=10)
val_loader = DataLoader(val_set, shuffle=False, batch_size=10)
test_loader = DataLoader(test_set, shuffle=False, batch_size=10)

# Initialize model
KAN_width = [X_data.shape[1], 8, 1]
model = KAN(width=KAN_width, grid=3, k=3, seed=1)
#model.to(device)  # Ensure model is on the correct device

# Loss function and optimizer
n_epochs = 5
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Store losses for plotting
train_losses = []
val_losses = []

# Directory to save model and weights
save_dir = "/BS/SparseExplainability/nobackup/KANSysbio/codes/models"  # Modify this path as per your environment
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Training loop
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        # Move inputs and targets to the device
        X_batch, y_batch = X_batch, y_batch
        
        y_pred = model(X_batch)
        loss = loss_fn(y_pred.squeeze(), y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            # Move inputs and targets to the device
            X_val, y_val = X_val, y_val
            
            y_pred_val = model(X_val)
            loss_val = loss_fn(y_pred_val.squeeze(), y_val)
            val_loss += loss_val.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Print losses for each epoch
    print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the entire model
model_file = os.path.join(save_dir, "KAN_model.pth")
torch.save(model, model_file)
print(f"Model saved at: {model_file}")

# Save only the model's state_dict (weights and biases)
weights_file = os.path.join(save_dir, "KAN_model_weights.pth")
torch.save(model.state_dict(), weights_file)
print(f"Weights and biases saved at: {weights_file}")

# Plot losses
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss Per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Test the model and compute RMSE
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for X_test, y_test in test_loader:
        # Move inputs and targets to the device
        X_test, y_test = X_test, y_test
        
        output = model(X_test)
        y_true.extend(y_test.cpu().numpy())
        y_pred.extend(output.squeeze().cpu().numpy())

# Compute RMSE on test set
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Test RMSE: {rmse:.4f}")
