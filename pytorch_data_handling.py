#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:30:27 2024

@author: msolanki
"""

import torch
import torchvision
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


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

#Convert into Pytorch Tensors

X_data_tensor = torch.tensor(X_data.values , dtype= torch.float32)

y_data_tensor = torch.tensor(y_data.values , dtype= torch.float32)

##Create DataLoader, then take one batch##

loader = DataLoader(list(zip(X_data_tensor , y_data_tensor)), shuffle = True , batch_size= 16)

for X_batch , y_batch in loader: 
    
    print(X_batch , y_batch)
    
    break


