import pandas as pd
import numpy as np

metaData1 = pd.read_csv(filepath_or_buffer="scRNA_metadata.csv", header='infer')

cellType1 = metaData1["cellType"]
subIDo1 = metaData1["subIDo"]

for i in range(0, len(cellType1)):
    if cellType1.iloc[i] == 'oligo':
        if subIDo1.iloc[i] == 'o1':
            subIDo1.iloc[i] = 'Oligo_1'
        elif subIDo1.iloc[i] == 'o2':
            subIDo1.iloc[i] = 'Oligo_2'
        elif subIDo1.iloc[i] == 'o3':
            subIDo1.iloc[i] = 'Oligo_3'
        elif subIDo1.iloc[i] == 'o4':
            subIDo1.iloc[i] = 'Oligo_4'
        elif subIDo1.iloc[i] == 'o5':
            subIDo1.iloc[i] = 'Oligo_5'
        elif subIDo1.iloc[i] == 'o6':
            subIDo1.iloc[i] = 'Oligo_6'
    elif cellType1.iloc[i] == 'unID' and subIDo1.iloc[i]=='ZZZ':
            subIDo1.iloc[i] = 'unID'
    elif cellType1.iloc[i] == 'astro' and subIDo1.iloc[i]=='ZZZ':
            subIDo1.iloc[i] = 'astro'
    elif cellType1.iloc[i] == 'OPC' and subIDo1.iloc[i]=='ZZZ':
            subIDo1.iloc[i] = 'OPC'
    elif cellType1.iloc[i] == 'neuron' and subIDo1.iloc[i]=='ZZZ':
            subIDo1.iloc[i] = 'neuron'
    elif cellType1.iloc[i] == 'endo' and subIDo1.iloc[i]=='ZZZ':
            subIDo1.iloc[i] = 'endo'
    elif cellType1.iloc[i] == 'mg' and subIDo1.iloc[i]=='ZZZ':
            subIDo1.iloc[i] = 'mg'
    
    

cellType_unique = cellType1.unique()
subIDo_unique = subIDo1.unique()

# Generating the true cell type using Cell Type Metadata
true_cell_arr = np.zeros(len(cellType1)).reshape(len(cellType1), 1)
true_cellType = pd.DataFrame(true_cell_arr, columns=['True Labels'])

for i in range(0, len(cellType1)):
    if cellType1.iloc[i]=='oligo':
        true_cellType.iloc[i] = 0
    elif cellType1.iloc[i] == 'unID':
        true_cellType.iloc[i] = 1
    elif cellType1.iloc[i] == 'astro':
        true_cellType.iloc[i] = 2
    elif cellType1.iloc[i] == 'OPC':
        true_cellType.iloc[i] = 3
    elif cellType1.iloc[i] == 'neuron':
        true_cellType.iloc[i] = 4
    elif cellType1.iloc[i] == 'endo':
        true_cellType.iloc[i] = 5
    elif cellType1.iloc[i] == 'mg':
        true_cellType.iloc[i] = 6
    elif cellType1.iloc[i] == 'doublet':
        true_cellType.iloc[i] = 7


# saving the true cell type dataframe in a csv
true_cellType.to_csv('scRNA_true_label_considering_unique_cellType.csv')
