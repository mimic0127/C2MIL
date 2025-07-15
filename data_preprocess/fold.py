# %%
import joblib
import os  
import random 
import torch 
import numpy as np
from sklearn.model_selection import StratifiedKFold
import h5py

patients = joblib.load('./data/Feat_result/TCGA_BLCA_input/blca_patients_386.pkl')
print(len(patients))
# package features
all_features = {}
for f in os.listdir('./data/Feature/BLCA_UNI/h5_files/'):
    print(f[:12])
    if f[:12] in patients:
        h5_file_path = './data/Feature/BLCA_UNI/h5_files/'+f 
        with h5py.File(h5_file_path, 'r') as file:
            data = file['features'][:]
            # print(torch.tensor(data))
        all_features[f[:12]] = torch.tensor(data)

joblib.dump(all_features,'./data/Feat_result/TCGA_BLCA_input/BLCA_features.pkl')

all_features = {}
for f in os.listdir('./data/Feature/BLCA_UNI/graph_files/'):
    print(f[:12])
    if f[:12] in patients:
        all_features[f[:12]]=torch.load('./data/Feature/BLCA_UNI/graph_files/'+f,map_location='cpu')

joblib.dump(all_features,'./data/Feat_result/TCGA_BLCA_input/BLCA_graphs.pkl')