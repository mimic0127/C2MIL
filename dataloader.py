import numpy as np
import torch
import torch.utils.data as data_utils
import torch_geometric.utils as utils
import h5py
import os
import csv
import pandas as pd

import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_scatter import scatter, segment_csr, gather_csr
import warnings

warnings.filterwarnings("ignore", category=Warning)

class CRCyjDataset(Dataset):
    def __init__(self, slides, cli_dict, slide_patient, slide_feats_dict, transform=None,
                 pre_transform=None):
        super(CRCyjDataset, self).__init__(transform, pre_transform)

        self.slides = slides
        self.cli_dict = cli_dict
        self.slide_patient = slide_patient
        self.slide_feats_dict = slide_feats_dict

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.slides)

    def get(self, idx):
        id = self.slides[idx]
        surv_time = self.cli_dict[self.slide_patient[id][0]][1]
        censorship = self.cli_dict[self.slide_patient[id][0]][0]
        slide_data = self.slide_feats_dict[id]

        return id, slide_data, surv_time, censorship 
    

class CRCyjDataset2(Dataset):
    def __init__(self, slides, cli_dict, slide_patient, slide_feats_dict, thumb_fea_root, transform=None,
                 pre_transform=None):
        super(CRCyjDataset2, self).__init__(transform, pre_transform)

        self.slides = slides
        self.cli_dict = cli_dict
        self.slide_patient = slide_patient
        self.slide_feats_dict = slide_feats_dict
        self.thumb_fea_root = thumb_fea_root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.slides)

    def get(self, idx):
        id = self.slides[idx]
        surv_time = self.cli_dict[self.slide_patient[id][0]][1]
        censorship = self.cli_dict[self.slide_patient[id][0]][0]
        slide_data = self.slide_feats_dict[id]
        thumb = torch.load(os.path.join(self.thumb_fea_root,f'{id.replace(".svs","")}.pt'),map_location='cpu').detach()

        return id, slide_data, surv_time, censorship, thumb