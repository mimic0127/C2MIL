import os
import torch
# from conch_lora import conch_img_lora
import torch.nn as nn
from softKmeans import SoftCluster, SoftCluster_adaptiveK
import torch.nn.functional as F
import time


class BiasRemoval(nn.Module):
    def __init__(self, thumb_all, patch_all, K=3):
        super(BiasRemoval, self).__init__()
        self.thumb_encoder = MLPImageEncoder(input_dim=thumb_all.shape[1])
        self.thumb_all = thumb_all
        self.patch_all = patch_all
        self.K = K
        # self.cluster = SoftCluster(num_clusters=self.K)
        self.cluster = SoftCluster_adaptiveK()
        self.bias = None
    def forward(self, thumb, Fea):
        if self.training:
            # 训练阶段，重新计算聚类中心
            # print(f"thumb_all shape: {self.thumb_all.shape}")
            thumb_fea_all = self.thumb_encoder(self.thumb_all)
            # print(f"thumb_fea_all shape: {thumb_fea_all}")
            # if thumb_fea_all.shape[0] == 0:
            #     raise ValueError("thumb_fea_all is empty, check input data!")
            s = time.time()
            self.cluster_centers, responsibilities = self.cluster(thumb_fea_all)
            e = time.time()
            print('time:',e-s)


            print("cluster_center:",self.cluster_centers)
            # 计算 T 矩阵
            # 固定K
            # T = torch.zeros((self.K + 1, Fea.size(-1)), device=Fea.device)
            # 自适应K
            self.K = self.cluster.best_k
            T = torch.zeros((self.K + 1, Fea.size(-1)), device=Fea.device)

            T[0] = self.patch_all.mean(dim=(0, 1))
            for k in range(self.K):
                weights = responsibilities[:, k].unsqueeze(-1)
                patch_fea_weighted_sum = (self.patch_all.mean(dim=1, keepdim=True) * weights.unsqueeze(2)).sum(dim=0)
                T[k + 1] = patch_fea_weighted_sum / torch.sum(weights)

            self.biases = T[1:] - T[0]  # (Tk - T0) for k in 1 to K
        else:
            # 验证/测试阶段，使用训练保存的聚类中心
            assert self.cluster.saved_cluster_centers is not None, "Cluster centers must be precomputed during training."
            self.cluster_centers = self.cluster.saved_cluster_centers
            print('cluster_num:',self.cluster_centers.shape)
            assert hasattr(self, "biases"), "Biases must be precomputed during training."

        thumb_fea = self.thumb_encoder(thumb)  # 当前批次的缩略图特征 # 1 x 1024
        cluster_pred = self.cluster.predict_cluster(thumb_fea, self.cluster_centers) # 1 x 5
        # cluster_pred = self.cluster.predict_cluster(thumb_fea) 
        Fea_new = Fea - torch.matmul(cluster_pred, self.biases)  # 去除偏差
        # if self.training:
        return Fea_new
        # else:
        #     return thumb_fea, cluster_pred

class MLPImageEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024):
        super(MLPImageEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Initialize weights with normal distribution
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc2(x)
        return x
    
# Example usage
def main():
    thumb_all = torch.randn(100, 1024)  
    patch_all = torch.randn(100, 500, 128)  
    K = 5

    model = BiasRemoval(thumb_all, patch_all,K)

    thumb = torch.randn(10, 3, 224, 224)  
    Fea = torch.randn(10, 500, 128)  

    Fea_new = model(thumb, Fea)

    print("Updated features:", Fea_new)

if __name__ == "__main__":
    main()