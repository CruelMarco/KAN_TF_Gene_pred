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
import time
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

gene = ["TP53", "BRAF", "PIK3CA", "KRAS", "PTEN", "E2F1", "FOXM1"]

gene = "BRAF"

wandb.init(project="TF_KAN", name="Regression_Model_KAN" + "_" + gene, config={
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 32
})

# Initialize Weights and Biases
#wandb.init(project="TF_KAN", name="Regression_Model_KAN")

def process_data():
    data_set_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp"
    os.chdir(data_set_dir)

    tf_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/tf.csv"
    gene_exp_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/gene_exp.csv"

    tf_csv = pd.read_csv(tf_data_dir, sep=",").drop(["Unnamed: 0"], axis=1)
    gene_exp_csv = pd.read_csv(gene_exp_data_dir, sep=",").drop(["Unnamed: 0"], axis=1)

    X_data = tf_csv.drop(["tissue_code"], axis=1)
    y_data = gene_exp_csv.drop(["Unnamed:"], axis=1)["TP53"]

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

# Hyperparameters
num_epochs = wandb.config.epochs
batch_size = wandb.config.batch_size

# Load data and model
X_cal, X_val, X_test, y_cal, y_val, y_test = process_data()

loss_fn = nn.MSELoss()

KAN_width = [X_cal.shape[1] , 8, 1]

model = KAN(width=KAN_width, grid=3, k=3, seed=1)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

#n_epochs = 10

#batch_size = 10

#batch_start = torch.arange(0 , len(X_cal) , batch_size)

##Hold the best model##

best_mse = np.inf

best_weights = None

history = []

train_history = []

save_dir = "/BS/SparseExplainability/nobackup/KANSysbio/codes/models"

os.makedirs(save_dir, exist_ok = True)

current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

model_save_path = os.path.join(save_dir, f"best_model_{current_time}.pth")

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_mse_accum = []
    with tqdm.tqdm(train_dl, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for data, gt in bar:
            gt = gt.unsqueeze(1)
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
            gt = gt.unsqueeze(1)
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
        "Validation MSE": val_mse
        #"epoch": epoch + 1
    })

    train_history.append(train_epoch_mse)
    history.append(val_mse)


            
model.load_state_dict(best_weights)

#print("MSE: %.2f" % best_mse)
#print("RMSE: %.2f" % np.sqrt(best_mse))

model.eval()

y_pred_list_test = []

y_true_list_test = []

y_true_list = []
with torch.no_grad():
    for data_test, gt_test in test_dl:
        y_pred_test = model(data_test)
        y_pred_list_test.append(y_pred_test.numpy())
        y_true_list_test.append(gt_test.numpy())

# Convert lists to numpy arrays for plotting and metric calculation
y_pred_test_array = np.concatenate(y_pred_list_test)
y_true_test_array = np.concatenate(y_true_list_test)

# Calculate MSE and RMSE for the test set
test_mse = np.mean((y_pred_test_array - y_true_test_array) ** 2)
test_rmse = np.sqrt(test_mse)

# Print MSE and RMSE
print(f"Test Set MSE: {test_mse:.4f}")
print(f"Test Set RMSE: {test_rmse:.4f}")

# Log MSE and RMSE to wandb
wandb.log({"Test Set MSE": test_mse, "Test Set RMSE": test_rmse})

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

fig_sav_dir = "/BS/SparseExplainability/nobackup/KANSysbio/plots"

file_name = filename = f"KAN_test_true_vs_predicted_scatter_{current_time}.png"

fig_save_name = os.path.join(fig_sav_dir , file_name)

# Scatter plot of predicted vs true values on the test set
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_true_test_array)), y_true_test_array, label="True Values", color="blue", alpha=0.6)
plt.scatter(range(len(y_pred_test_array)), y_pred_test_array, label="Predicted Values", color="orange", alpha=0.6)
plt.title("Predicted vs True Values on Test Set (Scatter Plot)")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig(fig_save_name)

# Log the scatter plot to wandb for visualization
wandb.log({"KAN Test True vs Predicted Scatter Plot" + "_" + gene: wandb.Image(fig_save_name)})

wandb.finish()
# Log the scatter plot to wandb for visualization
#wandb.log({"Test True vs Predicted Scatter Plot": plt})

