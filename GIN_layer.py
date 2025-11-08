import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


# GIN
class GIN_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(GIN_layer, self).__init__()
        self.latent_dim = [hidden_dim]

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, self.latent_dim[0], bias=False),
            nn.BatchNorm1d(self.latent_dim[0]),
            nn.ReLU(),
            nn.Linear(self.latent_dim[0], self.latent_dim[0], bias=False),
            nn.BatchNorm1d(self.latent_dim[0])),
            train_eps=True))
        self.device = device
        self.num_gnn_layers = 1
        self.emb_normlize = False
        self.gnn_non_linear = nn.ReLU()

    def forward(self, x, edge_index, batch, edge_weight=None):

        x, edge_index = x.to(self.device), edge_index.to(self.device)
        x = x.to(torch.float32)
        for i in range(self.num_gnn_layers):
            if edge_weight == None:
                x = self.gnn_layers[i](x, edge_index)
            else:
                x = self.gnn_layers[i](x, edge_index, edge_weight)
            if self.emb_normlize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)
        return x
