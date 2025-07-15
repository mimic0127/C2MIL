import os, sys
from os.path import join
import h5py
import math
from math import floor
import pdb
from time import time
from tqdm import tqdm

### Numerical Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import percentileofscore

### Graph Network Packages
import nmslib
import networkx as nx

### PyTorch / PyG
import torch 
from torch_geometric.utils import remove_self_loops, to_undirected
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import convert
import glob

class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

def pt2graph(wsi_h5, radius=9):
    from torch_geometric.data import Data as geomData
    from itertools import chain
    coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    if coords.shape[0] != features.shape[0]:
        print(coords)
        return
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]
    
    model = Hnsw(space='l2')
    model.fit(coords)
    radius = min(radius,num_patches)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),dtype=int)
    edge_spatial = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)
    
    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(features[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),dtype=int)
    edge_latent = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)

    G = geomData(x = torch.Tensor(features),
                 edge_index = edge_spatial,
                 edge_latent = edge_latent,
                 centroid = torch.Tensor(coords))
    print(len(G.x))
    return G


def get_edge_candidates(num_nodes, edge_index):
    """
    获取候选边集，排除已知边。
    
    参数:
    - num_nodes: 节点总数
    - edge_index: 已知的边索引，形状为 [2, num_edges]
    
    返回:
    - edge_candidate: 候选边集，排除已知边后的结果，形状为 [2, num_edge_candidates]
    """
    row = torch.arange(num_nodes).repeat(num_nodes)
    col = torch.arange(num_nodes).view(-1, 1).repeat(1, num_nodes).view(-1)
    
    all_edges = torch.stack([row, col], dim=0)
    
    all_edges, _ = remove_self_loops(all_edges)
    
    all_edges = to_undirected(all_edges)

    edge_set = set(map(tuple, edge_index.t().tolist()))
    
    candidate_edges = []
    for i in range(all_edges.size(1)):
        edge_tuple = tuple(all_edges[:, i].tolist())
        if edge_tuple not in edge_set:
            candidate_edges.append(edge_tuple)

    edge_candidate = torch.tensor(candidate_edges).t()
    print(edge_index.shape, edge_candidate.shape)
    
    return edge_candidate

ts = []
import time 
import numpy as np 
def createDir_h5toPyG(h5_path, save_path):
    pbar = tqdm(os.listdir(h5_path)[:50])
    # pbar = ['8_81132023-08-07_17_15_58','8_71142023-08-07_17_14_17','240401_133','240401_131','240401_132','240401_134','240401_135','8_41172023-08-07_17_09_41','8_51162023-08-07_17_11_27','8_91122023-08-07_17_17_51']
    for h5_fname in pbar:
        try:
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            # s = time.time()
            G = pt2graph(wsi_h5)
            # e = time.time()
            # print(e-s,)
            # ts.append((e-s))
            # G.edge_candidate = get_edge_candidates(len(G.x), G.edge_index)
            # G.num_edge_candidate = torch.Tensor([len(G.edge_candidate)])
            torch.save(G, os.path.join(save_path, h5_fname[:-3]+'.pt'))
            wsi_h5.close()
        except OSError:
            # pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')
    print(np.mean(ts))
import sys 
h5_path = './data/Feature/BLCA_UNI/h5_files'
save_path = h5_path.replace('h5_files','graph_files')
os.makedirs(save_path, exist_ok=True)
createDir_h5toPyG(h5_path, save_path)
