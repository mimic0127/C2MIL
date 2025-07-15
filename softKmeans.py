import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def plot_k_means(x, r, k):

    random_colors = np.random.random((k, 3))
    colors = r.dot(random_colors)
    print(r[:20])
    plt.scatter(x[:,0], x[:,1], c=colors)
    plt.show()


def initialize_centers(x, num_k):
    N, D = x.shape
    centers = np.zeros((num_k, D))
    used_idx = []
    for k in range(num_k):
        idx = np.random.choice(N)
        while idx in used_idx:
            idx = np.random.choice(N)
        used_idx.append(idx)
        centers[k] = x[idx]
    return centers

def update_centers(x, r, K):
    N, D = x.shape
    centers = np.zeros((K, D))
    for k in range(K):
        centers[k] = r[:, k].dot(x) / r[:, k].sum()
    return centers

def square_dist(a, b):
    return (a - b) ** 2

def cost_func(x, r, centers, K):
    
    cost = 0
    for k in range(K):
        norm = np.linalg.norm(x - centers[k], 2)
        cost += (norm * np.expand_dims(r[:, k], axis=1) ).sum()
    return cost


def cluster_responsibilities(centers, x, beta):
    N, _ = x.shape
    K, D = centers.shape
    R = np.zeros((N, K))

    for n in range(N):        
        R[n] = np.exp(-beta * np.linalg.norm(centers - x[n], 2, axis=1)) 
    R /= R.sum(axis=1, keepdims=True)

    return R

def soft_k_means(x, K, max_iters=100, beta=1.):
    centers = initialize_centers(x, K)
    prev_cost = 0
    for _ in range(max_iters):
        r = cluster_responsibilities(centers, x, beta)
        centers = update_centers(x, r, K)
        cost = cost_func(x, r, centers, K)
        if np.abs(cost - prev_cost) < 1e-5:
            break
        prev_cost = cost
        
    # plot_k_means(x, r, K)
    return 


def generate_samples(std=1, dim=2, dist=4):
    mu0 = np.array([0,0])
    mu1 = np.array([dist, dist])
    mu2 = np.array([0, dist])
    # num samps per class
    Nc = 300
    x0 = np.random.randn(Nc, dim) * std + mu0
    x1 = np.random.randn(Nc, dim) * std + mu1
    x2 = np.random.randn(Nc, dim) * std + mu2
    x = np.concatenate((x0, x1, x2), axis=0)
    return x
    

def main():
    x = generate_samples()
    soft_k_means(x, K=3)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftCluster(nn.Module):
    def __init__(self, num_clusters=5, beta=1.0, seed=988):
        super(SoftCluster, self).__init__()
        self.num_clusters = num_clusters
        self.beta = beta
        self.seed = seed
        self.generator = None
        self.saved_cluster_centers = None  # 保存训练阶段的聚类中心
        self._set_seed()
        # self.cluster_centers = nn.Parameter(torch.randn(num_clusters, input_dim))
        # nn.init.xavier_uniform_(self.cluster_centers)  # 使用 xavier_uniform 初始化权重
    def forward(self, x, max_iters=400, tol=1e-5):

        centers = self.kmeans_plusplus_init(x, self.num_clusters)
        # centers = self.cluster_centers  # 可学习参数
        prev_centers = torch.zeros_like(centers)
        
        for _ in range(max_iters):
            r = self.cluster_responsibilities(centers, x)
            centers = self.update_centers(x, r)
            
            # 检查收敛
            center_shift = torch.norm(centers - prev_centers)
            if center_shift < tol:
                break
            prev_centers = centers.clone()

        self.saved_cluster_centers = centers  # 保存训练阶段的聚类中心

        return centers, r

    def kmeans_plusplus_init(self, x, num_clusters):
        """初始化聚类中心"""
        N, D = x.shape
        centers = torch.empty((num_clusters, D), device=x.device)
        centers[0] = x[torch.randint(0, N, (1,))]  # 随机选择第一个中心点
        
        for i in range(1, num_clusters):
            dists = torch.cdist(x, centers[:i], p=2).min(dim=1)[0]
            prob = dists / dists.sum()
            new_center_idx = torch.multinomial(prob, 1)
            centers[i] = x[new_center_idx]

        return centers


    def cluster_responsibilities(self, centers, x):
        """计算软责任值"""
        dists = torch.cdist(x, centers, p=2)  # (N, K) 距离矩阵
        logits = -self.beta * dists  # 负距离作为 logits
        r = F.softmax(logits, dim=1)  # 计算责任值
        return r

    def update_centers(self, x, r):
        """更新聚类中心"""
        centers = torch.matmul(r.T, x) / r.sum(dim=0, keepdim=True).T
        return centers

    def predict_cluster(self, thumb_fea, cluster_centers):
        """预测聚类分布"""
        distances = torch.cdist(thumb_fea, cluster_centers)
        cluster_pred = F.softmax(-self.beta * distances, dim=1)  # (batch, K)
        return cluster_pred

    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


# 自适应聚类
class SoftCluster_adaptiveK(nn.Module):
    def __init__(self, max_clusters=6, beta=1.0, tau=1.0, seed=988):
        """
        - max_clusters: 最大可能的簇数
        - beta: 控制聚类分配的平滑度
        - tau: Gumbel-Softmax 的温度参数，控制分配的随机性
        """
        super(SoftCluster_adaptiveK, self).__init__()
        self.max_clusters = max_clusters
        self.beta = beta
        self.tau = tau
        self.seed = seed
        self.saved_cluster_centers = None  # 保存训练阶段的聚类中心
        self.cluster_logits = nn.Parameter(torch.randn(max_clusters))  # 可学习的聚类参数
        self._set_seed()

    def forward(self, x, max_iters=400, tol=1e-5):
        """
        训练过程中 k 变成可学习的。
        """
        # 计算 soft cluster weights
        soft_k_weights = F.gumbel_softmax(self.cluster_logits, tau=self.tau, hard=False)
        effective_k = (soft_k_weights > 0.1).sum().item()  # 统计有效的 k
        
        centers = self.kmeans_plusplus_init(x, effective_k)
        prev_centers = torch.zeros_like(centers)

        for _ in range(max_iters):
            r = self.cluster_responsibilities(centers, x, soft_k_weights)
            centers = self.update_centers(x, r)

            # 检查收敛
            center_shift = torch.norm(centers - prev_centers)
            if center_shift < tol:
                break
            prev_centers = centers.clone()

        self.saved_cluster_centers = centers
        self.best_k = effective_k
        return centers, r  # 返回当前可学习的 k

    def kmeans_plusplus_init(self, x, num_clusters):
        """K-Means++ 初始化聚类中心"""
        N, D = x.shape
        centers = torch.empty((num_clusters, D), device=x.device)
        centers[0] = x[torch.randint(0, N, (1,))]  # 随机选择第一个中心点
        
        for i in range(1, num_clusters):
            dists = torch.cdist(x, centers[:i], p=2).min(dim=1)[0]
            prob = dists / dists.sum()
            new_center_idx = torch.multinomial(prob, 1)
            centers[i] = x[new_center_idx]

        return centers

    def cluster_responsibilities(self, centers, x, soft_k_weights):
        """计算软责任值，并且通过 soft_k_weights 使 k 变成可学习的"""
        dists = torch.cdist(x, centers, p=2)  # (N, K) 距离矩阵
        logits = -self.beta * dists  # 负距离作为 logits
        r = F.softmax(logits, dim=1)  # 计算责任值
        
        # 通过 soft_k_weights 重新加权，使得 k 可学习
        r = r * soft_k_weights[:r.shape[1]]
        r = r / r.sum(dim=1, keepdim=True)  # 归一化
        return r

    def update_centers(self, x, r):
        """更新聚类中心"""
        centers = torch.matmul(r.T, x) / r.sum(dim=0, keepdim=True).T
        return centers

    def predict_cluster(self, thumb_fea, cluster_centers):
        """预测聚类分布"""
        distances = torch.cdist(thumb_fea, cluster_centers)
        cluster_pred = F.softmax(-self.beta * distances, dim=1)  # (batch, K)
        return cluster_pred

    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

if __name__ == "__main__":
    main()