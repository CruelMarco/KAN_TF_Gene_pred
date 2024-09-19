#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:34:20 2024

@author: msolanki
"""

import pandas as pd
import numpy as np
from kan import KAN
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import torch

data_set_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp"

os.chdir(data_set_dir)

tf_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/tf.csv"

gene_exp_data_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/gene_exp.csv"

tf_csv = pd.read_csv(tf_data_dir , sep = ",").drop(["Unnamed: 0"] , axis = 1)

gene_exp_csv = pd.read_csv(gene_exp_data_dir , sep = ",").drop(["Unnamed: 0"] , axis = 1)

X_data = tf_csv.drop(["tissue_code"] , axis = 1)

y_data = gene_exp_csv.drop(["Unnamed:"] , axis = 1)

X_cal , X_val , y_cal , y_val = train_test_split(X_data , y_data , test_size = 0.20, random_state=42)

KAN_width =  [X_cal.shape[1] , 50  , y_cal.shape[1]]

model = KAN(width = KAN_width , grid = 3 , k = 3)

dataset = {}

dtype = torch.get_default_dtype()

dataset['train_input'] = torch.from_numpy(X_cal.values).float()

dataset['train_label'] = torch.from_numpy(y_cal.values).float()

dataset['test_input'] = torch.from_numpy(X_val.values).float()

dataset['test_label'] = torch.from_numpy(y_val.values).float()

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

#model.fit(dataset, lamb=0.005, batch=1024, loss_fn = nn.CrossEntropyLoss(), metrics=[train_acc, test_acc], display_metrics=['train_loss', 'reg', 'train_acc', 'test_acc']);

model(dataset['train_input'])

model.plot()

#results = model.fit(dataset, lamb=0.005, opt = "LBFGS" , loss_fn = nn.MSELoss(), metrics=[train_acc, test_acc] , steps = 20)
