#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:28:55 2024

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
y_data = gene_exp_csv.drop(["Unnamed:"], axis=1).iloc[:,0]

X_cal, X_val, y_cal, y_val = train_test_split(X_data, y_data, test_size=0.20, random_state=42)

scaler = StandardScaler()

X_cal = scaler.fit_transform(X_cal)

X_val = scaler.transform(X_val)

KAN_width = [X_cal.shape[1], 5, 1]

dtype = torch.get_default_dtype()
# dataset = {
#     'train_input': torch.from_numpy(X_cal.values).float(),
#     'train_label': torch.from_numpy(y_cal.values[:, None]).float(),  
#     'test_input': torch.from_numpy(X_val.values).float(),
#     'test_label': torch.from_numpy(y_val.values[:, None]).float()}

dataset = {
    'train_input': torch.from_numpy(X_cal).float(),
    'train_label': torch.from_numpy(y_cal.values[:, None]).float(),  
    'test_input': torch.from_numpy(X_val).float(),
    'test_label': torch.from_numpy(y_val.values[:, None]).float()}


def train_mse():
    with torch.no_grad():
        predictions = model(dataset["train_input"])
        mse = torch.nn.functional.mse_loss(predictions, dataset['train_label'])
    return mse

def test_mse():
    with torch.no_grad():
        predictions = model(dataset['test_input'])
        mse = torch.nn.functional.mse_loss(predictions, dataset['test_label'])
    return mse

lam = [0.00001]

for i in lam:
    
    model = KAN(width=KAN_width, grid=5, k=5, seed=1)

    results = model.fit(dataset, opt="LBFGS" , metrics=(train_mse, test_mse),
                        loss_fn=torch.nn.MSELoss(), steps=5, lamb=i, lamb_entropy=2.)

# Plotting the training and testing losses
    plt.figure(figsize=(10, 5))
    plt.plot(results['train_mse'], label='Train MSE')
    plt.plot(results['test_mse'], label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Train and Test MSE Loss During Training for Lamb = {i}')
    plt.legend()
    plt.show()
    
    # Save the model
    #model_save_path = '/BS/SparseExplainability/nobackup/KANSysbio/codes/models/model_642_10_1.pth'
    #torch.save(model.state_dict(), model_save_path)
    
    #print("Model saved to", model_save_path)
    print("Final Train MSE:", results['train_mse'][-1])
    print("Final Test MSE:", results['test_mse'][-1])

#model.plot(beta=50, scale=1, out_vars=['GENE'])
