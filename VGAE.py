import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops


# class DeepVNAE(nn.Module):

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(enc_in_channels,
                                                          enc_hidden_channels,
                                                          enc_out_channels))

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        # print(z)

        # z,_ = self.encoder(x,edge_index)
        return z


# old和new一起过
class GCNEncoder_oldnew(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder_oldnew, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE_mse(VGAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels):
        super(DeepVGAE_mse, self).__init__(encoder=GCNEncoder_oldnew(enc_in_channels,
                                                                     enc_hidden_channels,
                                                                     enc_out_channels))

    def forward(self, x, edge_index):
        # 生成两个图的边索引并合并
        # num_nodes = edge_index.max().item() + 1
        num_nodes = len(x)//2
        edge_index1_offset = edge_index + num_nodes
        combined_edge_index = torch.cat([edge_index, edge_index1_offset], dim=1)

        # 编码得到 z
        z = self.encode(x, combined_edge_index)
        half_index = z.shape[0] // 2
        z_old = z[:half_index]
        z_new = z[half_index:]
        # 计算 MSE 损失
        # mse_loss = F.mse_loss(z_old, z_new)
        return z[half_index:].flatten(), z_old, z_new