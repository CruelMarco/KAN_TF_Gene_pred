#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:02:34 2024

@author: msolanki
"""

import pandas as pd
import os

gene_exp_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/gene_exp.csv"

tf_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/tf.csv"

gene_exp_csv = pd.read_csv(gene_exp_dir , sep = ",").drop(columns = ["Unnamed: 0"])

tf_csv = pd.read_csv(tf_dir , sep = ",").drop(columns = ["Unnamed: 0"])



