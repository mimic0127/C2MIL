import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, softmax, to_dense_adj, dense_to_sparse
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import (
    DeepGCNLayer, SGConv, GCNConv, SAGEConv, GATConv, TransformerConv,
    GINConv, TAGConv, GENConv, SAGPooling, GraphConv, TopKPooling,
    global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap, 
    GlobalAttention
)

from torch_scatter import scatter_add
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torch.nn import Sequential as Seq, Linear, LayerNorm, ReLU
from VGAE import DeepVGAE_mse, DeepVGAE
from Graphtransformer import DeepGraphTransformer_mse
from bias_removal import BiasRemoval

def bernoulli_sample_with_ste(logits):
    probs = torch.sigmoid(logits)
    samples = torch.bernoulli(probs)
    samples = samples.detach() + probs - probs.detach()
    return samples

def get_subgraph_data(edge_index_sub, data_x):
    nodes = torch.unique(edge_index_sub)
    x_sub = data_x[nodes]
    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(nodes)}
    edge_index_sub_renumbered = torch.tensor(
        [[node_map[old_idx.item()] for old_idx in edge_index_sub[0]],
         [node_map[old_idx.item()] for old_idx in edge_index_sub[1]]],
        dtype=torch.long
    ).to(data_x.device)
    
    return x_sub, edge_index_sub_renumbered, nodes

class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x
    
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1, edge_dim=None):
        super(GraphTransformer, self).__init__()
        self.edge_dim = edge_dim
        self.conv = TransformerConv(in_channels, out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.edge_gvae = DeepVGAE(out_channels,out_channels//2,1)
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None and self.edge_dim is not None:
            edge_attr = torch.ones((edge_index.size(1), self.edge_dim), device=x.device)
        
        x, attention_weights = self.conv(x, edge_index, return_attention_weights=True, edge_attr=edge_attr)
        return x, attention_weights

    def get_node_and_edge_scores(self, x, edge_index, edge_attr=None):
        node_embeddings, attention_weights = self.forward(x, edge_index, edge_attr)
        node_scores = torch.norm(node_embeddings, dim=1)
        edge_scores = attention_weights[1].mean(dim=1)
        
        return node_scores, edge_scores, node_embeddings

    def filter_edges(self, edge_index, node_mask):
        """Filter out edges where both nodes are not in the subgraph."""
        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        return edge_index[:, mask]
    
    def select_high_score_subgraph(self, x, edge_index, node_threshold, edge_threshold, edge_attr=None):
        node_scores, edge_scores, node_embeddings = self.get_node_and_edge_scores(x, edge_index, edge_attr)
        high_score_node_indices = torch.nonzero(node_scores > node_threshold).squeeze()
        low_score_node_indices = torch.nonzero(node_scores <= node_threshold).squeeze()
        
        high_score_node_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        high_score_node_mask[high_score_node_indices] = True
        low_score_node_mask = ~high_score_node_mask

        edge_mask = edge_scores > edge_threshold
        low_score_edge_mask = edge_scores <= edge_threshold
        
        sub_edge_index = self.filter_edges(edge_index[:, edge_mask], high_score_node_mask)
        sub_x = x[high_score_node_indices] if high_score_node_indices.numel() > 0 else x.new_zeros((0, x.size(1)))

        remaining_edge_index = self.filter_edges(edge_index[:, low_score_edge_mask], low_score_node_mask)
        remaining_x = x[low_score_node_indices] if low_score_node_indices.numel() > 0 else x.new_zeros((0, x.size(1)))
        
        total_nodes = x.size(0)
        high_score_node_ratio = len(high_score_node_indices) / total_nodes if total_nodes > 0 else 0.0
        
        return sub_x, sub_edge_index, None, high_score_node_indices, remaining_x, remaining_edge_index, None, high_score_node_ratio


class PatchGCN_Surv_causal(torch.nn.Module):
    def __init__(self, num_layers=4, edge_agg='spatial', multires=False, resample=0,
        fusion=None, num_features=1024, hidden_dim=256, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=1):
        super(PatchGCN_Surv_causal, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample
        self.node_threshold=10
        self.edge_threshold=0.15
        # self.edge_gvae_mse = DeepVGAE_mse(hidden_dim,hidden_dim,1)
        self.edge_gvae_mse = DeepGraphTransformer_mse(hidden_dim,hidden_dim,1)
        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU()])

        self.classifier = torch.nn.Linear(hidden_dim*4, n_classes)    

        self.Subgraph = GraphTransformer(in_channels=hidden_dim,out_channels=hidden_dim)

    def VGAE_subgraph(self, x_old_new, edge_index):
        # batch = x.new_zeros((x.shape[0])).long()
        # edge_adj = to_dense_adj(edge_index,batch)
        if self.training:
            z_prob, z_old, z_new = self.edge_gvae_mse(x_old_new, edge_index)
            node_preds = bernoulli_sample_with_ste(z_prob)
            # node_preds = torch.sigmoid(z_prob)
        else:
            z_prob, z_old, z_new = self.edge_gvae_mse(x_old_new, edge_index)
            # node_preds = (torch.sigmoid(z_prob)>0.5).long()
            node_preds = torch.sigmoid(z_prob)
        print(node_preds)
        with torch.no_grad():
            causal_mask = node_preds.bool()
            remain_mask = ~causal_mask
            causal_set = torch.where(causal_mask)[0].long()
            remain_set = torch.where(remain_mask)[0].long()
        # causal_x = x[causal_mask]
        # remain_x = x[remain_mask]
        causal_x = x_old_new[x_old_new.shape[0] // 2:] * node_preds.unsqueeze(1)
        remain_x = x_old_new[x_old_new.shape[0] // 2:] * (1-node_preds.unsqueeze(1))
        causal_edge_index, causal_edge_attr = subgraph(causal_set,edge_index,relabel_nodes=False)
        remain_edge_index, remain_edge_attr = subgraph(remain_set,edge_index,relabel_nodes=False)
        return causal_x, causal_edge_index, causal_edge_attr, causal_set, remain_x, remain_edge_index, remain_edge_attr, torch.sum(node_preds)/len(x_old_new[x_old_new.shape[0] // 2:]), z_old, z_new

    def GCN_forward(self, x_, edge_index, edge_attr, batch):
        B = batch.max()+1
        # print(B)
        # batch: [0,0,0..,1,1,1,..,2,2,2]
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
        
        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path) # N * 1, N * D
        A_path = torch.transpose(A_path, 1, 0) # d * N 
        A_path = softmax(A_path,index=batch,dim=-1)
        h_list = []
        for bi in range(B):
            idx_i = torch.where(batch==bi)[0]
            h_p_i = torch.mm(A_path[:,idx_i],h_path[idx_i])
            h_list.append(h_p_i)
        h_p = torch.stack(h_list,dim=0)
        h = self.path_rho(h_p.squeeze(1)) # B * D
    
        return h

    def forward(self, fea_old, x, edge_index, edge_attr, **kwargs):

        # 原图输出      
        if self.edge_agg == 'spatial':
            edge_index = edge_index
        elif self.edge_agg == 'latent':
            edge_index = edge_index

        # batch = data.batch
        edge_attr = None

        x_old_new = torch.cat([fea_old,x])
        
        x0_old_new = self.fc(x_old_new)
        half = x0_old_new.shape[0]//2
        x_ = x0_old_new[half:] 
        causal_x, causal_edge_index, causal_edge_attr, high_score_node_indices, remaining_x, remaining_edge_index, remaining_edge_attr, high_score_node_ratio, z_old, z_new = self.VGAE_subgraph(x0_old_new, edge_index)

        x_all = torch.cat([x_,causal_x, remaining_x])
        batch_all = torch.cat([x_.new_ones(len(x_)).long()-1, causal_x.new_ones(len(causal_x)).long(), remaining_x.new_ones(len(remaining_x)).long()+1])
        causal_edge_index_new = causal_edge_index + len(x_)
        remaining_edge_index_new = remaining_edge_index + len(x_)+len(causal_x)
        edge_index_all = torch.cat([edge_index, causal_edge_index_new, remaining_edge_index_new],dim=1)
        edge_attr_all = None
        h_all = self.GCN_forward(x_all,edge_index_all,edge_attr_all,batch_all) # B*hidden_dim
        print("h_all",h_all.shape)
        
        # h = self.GCN_forward(x_, edge_index, edge_attr)
        # logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        logits_all = self.classifier(h_all) # (N1+N2+N3) * num_cls

        hazards = torch.sigmoid(logits_all[0]).unsqueeze(dim=0)
        S = torch.cumprod(1 - hazards, dim=1)

        hazards_c = torch.sigmoid(logits_all[1]).unsqueeze(dim=0)

        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
#       
        
        print('causal_x:',causal_x.size())
        print('causual_edge_index:',causal_edge_index.size())
        print('remaining_x:',remaining_x.size())
        print('remaining_edge_index:',remaining_edge_index.size())
        # causal部分

        h = h_all[0]
        h_causal = h_all[1]
        h_remaining = h_all[2]
        return hazards,h,hazards_c,h_causal,h_remaining,high_score_node_ratio, z_old, z_new

class GraphCausal2_Surv(nn.Module):
    def __init__(self, thumb_all, patch_all, K = 5, num_layers=4, edge_agg='spatial', multires=False, resample=0,
        fusion=None, num_features=1024, hidden_dim=256, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=1):
        super(GraphCausal2_Surv, self).__init__()

        self.biasremoval = BiasRemoval(thumb_all = thumb_all, patch_all=patch_all, K = K)
        self.graphcausal = PatchGCN_Surv_causal(num_layers=num_layers, edge_agg=edge_agg, multires=multires, resample=resample,
        fusion=fusion, num_features=num_features, hidden_dim=hidden_dim, linear_dim=linear_dim, use_edges=use_edges, pool=pool, dropout=dropout, n_classes=n_classes)
    
    def forward(self, data,thumb):
        # print(f"thumb shape: {thumb if thumb is not None else None}")
        # print(f"data.x shape: {data.x if data.x is not None else None}")
        Fea_new = self.biasremoval(thumb, data.x) 
        Fea_old = data.x.clone()
        # print("Fea_new: ", Fea_new)
        hazards, h, hazards_c, h_causal, h_remaining, causal_ratio, z_old, z_new = self.graphcausal(Fea_old, Fea_new, data.edge_index,data.edge_attr)    

        return hazards, h, hazards_c, h_causal, h_remaining, causal_ratio, z_old, z_new
