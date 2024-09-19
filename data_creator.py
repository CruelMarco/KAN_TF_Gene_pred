#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:23:14 2024

@author: msolanki
"""

import os
import pandas as pd

data_set_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/"

prot_ab_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/CCLE_RPPA_20181003.csv"

gene_exp_dir = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv"

annot_file_dir = '/BS/SparseExplainability/nobackup/KANSysbio/data_sets/Cell_lines_annotations_20181226.txt'

human_tf = "/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tissues_motif.txt"

os.chdir(data_set_dir)

prot_ab_csv = pd.read_csv(prot_ab_dir , sep = ',')

gene_exp_csv = pd.read_csv(gene_exp_dir , sep = ',')

with open(annot_file_dir, 'r') as f:
    
    first_line = f.readline().strip()
    
    columns = first_line.split()[:2]
    
    lines = f.readlines()
    
with open(human_tf , "r") as t:
    
    col = t.readlines()

tf = [i.split()[0] for i in col]

#tf = set(tf)

#tf = list(tf)

unique_tf = []

for item in tf:
    
    if item not in unique_tf:
        
        unique_tf.append(item)

gene_cols = gene_exp_csv.columns

gene_cols_split = [i.split(" ")[0] for i in gene_cols]

common_elements = [element for element in unique_tf if element in gene_cols_split]

common_elements_with_indices = [(element, gene_cols_split.index(element)) for element in unique_tf if element in gene_cols_split]

sorted_common_elements_with_indices = sorted(common_elements_with_indices, key=lambda x: x[1])

sorted_common_elements = [element for element, index in sorted_common_elements_with_indices]

gene_exp_csv_split = gene_exp_csv

gene_exp_csv_split.columns = gene_cols_split

tf_csv = gene_exp_csv_split[sorted_common_elements]

tissue_col = gene_exp_csv_split.iloc[:,0]

tf_csv.insert(0 , "tissue_code" , tissue_col)

tf_csv.to_csv("/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/tf.csv" , sep = ",")

gene_exp_csv_split.to_csv("/BS/SparseExplainability/nobackup/KANSysbio/data_sets/tf_to_gene_exp/gene_exp.csv" , sep = ",")




