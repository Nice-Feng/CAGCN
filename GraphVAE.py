import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch_geometric.nn.glob import global_mean_pool
from sklearn.cluster import KMeans


eps = 1e-15
MAX_LOGSTD = 10
class GraphEncoder(torch.nn.Module):
 
    def __init__(self,in_channels, out_channels, device):
        super(GraphEncoder, self).__init__()

        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv_mu= GCNConv(2*out_channels,out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        self.device = device

    def encode(self, x, edge_index):
        x = x.to(torch.float32)
        x = self.conv1(x, edge_index)
        return x

    def getnew_adj(self,z):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)

    def forward(self, data,device):
        x, edge_indextop = data.x.to(self.device), data.edge_index.to(self.device)
        x = F.normalize(x, dim=1)
        ztop = self.encode(x,edge_indextop)


        aindex = self.getnew_adj(ztop)
        aedge = aindex[edge_indextop[0], edge_indextop[1]].to(device)
        #aedge = torch.sigmoid(aedge)
        #aedge = aedge.double()
        #grap_new = global_mean_pool(ztop, data.batch)
        x, edge_indexlast = data.x.to(self.device), data.lastedg_index.to(self.device)
        zlast = self.encode(x, edge_indexlast)
        return ztop, zlast, aedge




def V2_loss(x, y, z, device):
    s_x = calculate_sigma(x) ** 2
    s_y = calculate_sigma(y) ** 2
    s_z = calculate_sigma(z) ** 2
    Hyz = joint_entropy(y, z, s_y, s_z, device)
    Hxz = joint_entropy(x, z, s_x, s_z, device)
    Hz = reyi_entropy(z, sigma=s_z, device=device)
    Hxyz = joint_entropy3(x, y, z, s_x, s_y, s_z, device)
    CI = Hyz + Hxz - Hz - Hxyz
    return CI



def V1_loss(ztop,zlast,device):
    ##  mi(x,y)=H(x)-H(x|,y)\
    V1 = calculate_MI(ztop, zlast,device)
    return V1

def calculate_MI(x, y, device):
    s_x = calculate_sigma(x)
    s_y = calculate_sigma(y)
    Hx = reyi_entropy(x, s_x ** 2, device)
    Hy = reyi_entropy(y, s_y ** 2, device)
    Hxy = joint_entropy(x, y, s_x ** 2, s_y ** 2, device)
    Ixy = Hx + Hy - Hxy
    return Ixy

def joint_entropy3(x,y,z,s_x,s_y,s_z,device):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    z = calculate_gram_mat(z,s_z)
    k = torch.mul(x,y)
    k = torch.mul(k,z)
    k = k/(torch.trace(k)+eps)
    k = k.cpu()
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eigv = eigv.to(device)
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy


def reyi_entropy(x, sigma, device):
    alpha = 1.01# lunwen delta

    k = calculate_gram_mat(x,sigma)
    k = k/(torch.trace(k)+eps)
    k=k.cpu()
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eigv=eigv.to(device)
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy




def joint_entropy(x,y,s_x,s_y,device):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/(torch.trace(k)+eps)
    k = k.cpu()
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eigv = eigv.to(device)
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_sigma(Z_numpy):

    if Z_numpy.dim()==1:
        Z_numpy = Z_numpy.unsqueeze(1)
    Z_numpy = Z_numpy.cpu().detach().numpy()
    #print(Z_numpy.shape)
    k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
    sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
    if sigma < 0.1:
        sigma = 0.1
    return sigma

def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)


def pairwise_distances(x):
    #x should be two dimensional
    if x.dim()==1:
        x = x.unsqueeze(1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def kmeans_neigbour(feats):
    _N = feats.size(0)
    np_feats = feats.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=20, random_state=0).fit(np_feats)  # Kmeans聚类;
    centers = kmeans.cluster_centers_  # 聚类中心
    dis = euclidean_distances(np_feats, centers)  # 每个距离聚类中心的聚类 (2708, 400)
    _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)  # 距离最近的k个(1个)
    cluster_center_dict = cluster_center_dict.numpy()
    point_labels = kmeans.labels_  # 每个节点属于的类别
    point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]  # 每个类别中包含的节点;
    idx = torch.LongTensor([[sample_ids_v2(point_in_which_cluster[cluster_center_dict[point][i]], self.kc)  # 簇中最近的k个节点
                             for i in range(self.n_center)] for point in range(_N)])  # 遍历每个节点; 遍历k个kmeans超边;
    self.kmeans = idx  # 每个节点对应到最近簇的节点
    idx = idx[ids]  # 训练节点对应的簇邻居节点
    N = idx.size(0)
    d = feats.size(1)
    cluster_feats = feats[idx.view(-1)].view(N, self.n_center, self.kc, d)  # 簇邻居节点的特征; [140, 1, 64, 256]
