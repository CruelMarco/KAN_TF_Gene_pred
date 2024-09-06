#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:13:05 2024

@author: msolanki
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from kan import *
from sklearn.preprocessing import StandardScaler
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import tqdm as tqdm
import copy
import wandb  # Import wandb

# Initialize wandb
wandb.init(project="KAN_tf_to_gene_exp", entity="your-username")  # Change "your-username" to your Wandb username
wandb.config = {
    "learning_rate": 0.0001,
    "epochs": 5,
    "batch_size": 10,
    "model_width": [X_data.shape[1], 8, 1],
}

#wandb.login(key=["cec77c120fd0c482c535deeeb54879448f826151"])

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

data_set_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp"
os.chdir(data_set_dir)

tf_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/tf.csv"
gene_exp_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/gene_exp.csv"

tf_csv = pd.read_csv(tf_data_dir, sep=",").drop(["Unnamed: 0"], axis=1)
gene_exp_csv = pd.read_csv(gene_exp_data_dir, sep=",").drop(["Unnamed: 0"], axis=1)

X_data = tf_csv.drop(["tissue_code"], axis=1)
y_data = gene_exp_csv.drop(["Unnamed:"], axis=1).iloc[:, 0]

X_cal, X_val, y_cal, y_val = train_test_split(X_data, y_data, test_size=0.20, random_state=42)

scaler = StandardScaler()

X_cal = scaler.fit_transform(X_cal)
X_val = scaler.transform(X_val)

KAN_width = [X_cal.shape[1], 8, 1]

y_cal = y_cal.values
y_val = y_val.values

X_cal = torch.tensor(X_cal, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_cal = torch.tensor(y_cal, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

model = KAN(width=KAN_width, grid=3, k=3, seed=1)

# loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 5
batch_size = 10
batch_start = torch.arange(0, len(X_cal), batch_size)

# Hold the best model
best_mse = np.inf
best_weights = None
history = []
train_history = []

# Training Loop
for epoch in range(n_epochs):
    
    model.train()
    train_mse_accum = []
    
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        
        bar.set_description(f"Epoch {epoch}")
        
        for start in bar:
            # Take a batch
            X_batch = X_cal[start: start + batch_size]
            y_batch = y_cal[start: start + batch_size]
            
            # Forward Pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update Weights
            optimizer.step()
            
            train_mse_accum.append(float(loss))
            bar.set_postfix(mse=float(loss))
    
    model.eval()
    y_pred = model(X_val)
    mse = loss_fn(y_pred, y_val)
    mse = float(mse)
    
    history.append(mse)
    train_epoch_mse = np.mean(train_mse_accum)
    train_history.append(train_epoch_mse)
    
    # Log training and validation metrics to Wandb
    wandb.log({
        "epoch": epoch,
        "train_mse": train_epoch_mse,
        "val_mse": mse,
    })
    
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_weights)

# Save the best model to Wandb
torch.save(model.state_dict(), "best_model.pth")
wandb.save("best_model.pth")

print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))

# Plot and log to Wandb
plt.plot(train_history, label='Training MSE')
plt.plot(history, label='Validation MSE')
plt.title("Training and Validation MSE per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()

# Log the final plot to Wandb
wandb.log({"Training and Validation MSE Plot": wandb.Image(plt)})
wandb.finish()
