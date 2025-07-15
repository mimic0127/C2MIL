import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_transformer_pytorch_v import GraphTransformer

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, depth=1):
        super(GraphTransformerEncoder, self).__init__()
        self.transformer = GraphTransformer(
            dim=in_channels,
            depth=depth,
            heads=num_heads,
            dim_head=hidden_channels // num_heads,
            edge_dim=2*in_channels,  # Ensure edge_dim matches attention dimensions
            with_feedforwards=True,
            gated_residual=True
        )
        self.proj_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index):
        x = x.unsqueeze(0)  # (1, N, in_channels)
        edge_attr = edge_attr.unsqueeze(0)
        # print(x.shape,edge_attr.shape)
        x, _ = self.transformer(x, edges=edge_attr,edge_index=edge_index)  # Pass transformed edge_attr
        x = self.proj_out(x)
        return x.squeeze(0)  

class DeepGraphTransformer_mse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, depth=1):
        super(DeepGraphTransformer_mse, self).__init__()
        self.encoder = GraphTransformerEncoder(
            in_channels, hidden_channels, out_channels, num_heads, depth
        )

    def forward(self, x, edge_index):
        num_nodes = len(x) // 2
        N,D = x.shape
        edge_index1_offset = edge_index + num_nodes
        combined_edge_index = torch.cat([edge_index, edge_index1_offset], dim=1)

        # node_features_u = x[edge_index[0]]
        # node_features_v = x[edge_index[1]]

        src_features = x[edge_index[0]]  # 起点特征 (num_edges x dim)
        dst_features = x[edge_index[1]]  # 终点特征 (num_edges x dim)

        # 拼接边的特征
        edge_features = torch.cat([src_features, dst_features], dim=-1)  # (num_edges x 2*dim)

        # 构建完整的 N x N x (2*dim) 特征矩阵
        row_indices, col_indices = edge_index  # 获取起点和终点索引
        # edge_features_full = torch.zeros(N, N, 2 * D).to(x.device)  # 初始化为零
        # edge_features_full[row_indices, col_indices] = edge_features  # 使用张量索引填充

        edge_attr = torch.cat([src_features, dst_features], dim=1)  # (E, 2 * in_channels)

        # combined_edge_attr = torch.cat([edge_attr, edge_attr.clone()], dim=0)  # (2E, 2 * in_channels)

        z = self.encoder(x, edge_attr, edge_index)
        half_index = z.shape[0] // 2
        z_old, z_new = z[:half_index], z[half_index:]

        return z_new.view(-1), z_old, z_new 