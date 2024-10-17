#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:42:40 2024

@author: msolanki
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tqdm as tqdm
import copy
import wandb
import matplotlib.pyplot as plt

# Initialize Weights and Biases
wandb.init(project="TF_MLP", name="Regression_Model_Training", config={
    "learning_rate": 0.0001,
    "epochs": 100,
    "batch_size": 32
})

def process_data():
    data_set_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp"
    os.chdir(data_set_dir)

    tf_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/tf.csv"
    gene_exp_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/gene_exp.csv"

    tf_csv = pd.read_csv(tf_data_dir, sep=",").drop(["Unnamed: 0"], axis=1)
    gene_exp_csv = pd.read_csv(gene_exp_data_dir, sep=",").drop(["Unnamed: 0"], axis=1)

    X_data = tf_csv.drop(["tissue_code"], axis=1)
    y_data = gene_exp_csv.drop(["Unnamed:"], axis=1).iloc[:, 0]

    X_test = X_data.sample(n=200, random_state=42)
    test_rows_indices = X_test.index

    X_data_remaining = X_data.drop(test_rows_indices)
    y_test = y_data.iloc[test_rows_indices]
    y_data_remaining = y_data.drop(test_rows_indices)

    X_cal, X_val, y_cal, y_val = train_test_split(X_data_remaining, y_data_remaining, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_cal = scaler.fit_transform(X_cal)
    X_val = scaler.transform(X_val)

    X_cal = torch.tensor(X_cal, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_cal = torch.tensor(y_cal.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    return X_cal, X_val, X_test, y_cal, y_val, y_test

class RegressionDataset(Dataset):
    def __init__(self, variant):
        X_cal, X_val, X_test, y_cal, y_val, y_test = process_data()
        if variant == 'train':
            self.data_X, self.data_Y = X_cal, y_cal
        elif variant == 'val':
            self.data_X, self.data_Y = X_val, y_val
        elif variant == 'test':
            self.data_X, self.data_Y = X_test, y_test

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_Y[idx]

# Create datasets and dataloaders
train_dataset = RegressionDataset(variant='train')
val_dataset = RegressionDataset(variant='val')
test_dataset = RegressionDataset(variant='test')

train_dl = DataLoader(train_dataset, batch_size=32, num_workers=2)
val_dl = DataLoader(val_dataset, batch_size=32)
test_dl = DataLoader(test_dataset, batch_size=32)

class RegressionModel(nn.Module):
    def __init__(self, in_features):
        super(RegressionModel, self).__init__()
        self.linear_layer1 = nn.Linear(in_features, 24)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(24, 12)
        self.linear_layer3 = nn.Linear(12, 6)
        self.linear_layer4 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.relu(self.linear_layer1(x))
        x = self.relu(self.linear_layer2(x))
        x = self.relu(self.linear_layer3(x))
        x = self.linear_layer4(x)
        return x

# Hyperparameters
num_epochs = wandb.config.epochs
batch_size = wandb.config.batch_size

# Load data and model
X_cal, X_val, X_test, y_cal, y_val, y_test = process_data()
loss_fn = nn.MSELoss()
model = RegressionModel(X_cal.shape[1])
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# Track best model
best_mse = np.inf
best_weights = None

history = []
train_history = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_mse_accum = []
    with tqdm.tqdm(train_dl, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for data, gt in bar:
            y_pred = model(data)
            loss = loss_fn(y_pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse_accum.append(float(loss))
            bar.set_postfix(mse=float(loss))

    # Validation
    model.eval()
    val_mse_accum = []
    with torch.no_grad():
        for data, gt in val_dl:
            y_pred = model(data)
            mse = loss_fn(y_pred, gt)
            val_mse_accum.append(float(mse))

    val_mse = np.mean(val_mse_accum)
    train_epoch_mse = np.mean(train_mse_accum)

    # Save best model
    if val_mse < best_mse:
        best_mse = val_mse
        best_weights = copy.deepcopy(model.state_dict())

    # Logging to wandb in real-time
    wandb.log({
        "Train MSE": train_epoch_mse,
        "Validation MSE": val_mse,
        "epoch": epoch + 1
    })

    train_history.append(train_epoch_mse)
    history.append(val_mse)

# Load best model weights
model.load_state_dict(best_weights)

# Final evaluation
print(f"Best Validation MSE: {best_mse:.2f}")
print(f"Best Validation RMSE: {np.sqrt(best_mse):.2f}")

# Log best results to wandb
wandb.log({"Best Validation MSE": best_mse})

# Plot training history
plt.plot(train_history, label='Training MSE')
plt.plot(history, label='Validation MSE')
plt.title("Training and Validation MSE per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()

# Finish the wandb run
wandb.finish()
