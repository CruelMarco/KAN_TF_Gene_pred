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

class TFGeneDataset(Dataset):
    
    def __init__(self, X , y):
        
        self.X = torch.tensor(X.values , dtype=torch.float32)
    
        self.y = torch.tensor(y.values , dtype = torch.float32)
        
    def __len__(self):
        
        return len(self.X)
    
    def __getitem__(self, idx):
        
        ##this should return one sample from dataset##
        
        feature = self.X[idx]
        
        target = self.y[idx]
        
        return feature , target
    
dataset = TFGeneDataset(X_data, y_data)

train_set, val_set , test_set = random_split(dataset , [0.8 , 0.1 , 0.1])

loader = DataLoader(train_set , shuffle = True , batch_size = 10)

KAN_width = [X_data.shape[1], 8, 1]

model = KAN(width=KAN_width, grid=3, k=3, seed=1)

#Train model##

n_epochs = 5

loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.train()

for epoch in range(n_epochs):
    
    for X_batch , y_batch in loader:
        
        y_pred = model(X_batch)
        
        loss = loss_fn(y_pred , y_batch)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
X_val, y_val = default_collate(val_set)

model.eval()

y_pred = model(X_test)



        
        
                    


    
    