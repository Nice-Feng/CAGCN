from torch_geometric.nn import MessagePassing
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
import torch
from GraphVAE import GraphEncoder, V1_loss, V2_loss
import torch.nn as nn


criterion = nn.CrossEntropyLoss()
def clear_masks(model):
    """ clear the edge weights to None """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None

def set_masks(model, edgemask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edgemask

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

def ginconv_cal(subtop, sublast,batchdata,aedge,gin_layer,class_model, device):

    edge_index, batch, x = batchdata.edge_index, batchdata.batch,batchdata.x
    clear_masks(gin_layer)
    set_masks(gin_layer, aedge)
    x2 = gin_layer(x.to(device), edge_index.to(device), batch.to(device))
    y_out, node_emd,= class_model(x2, batch)
    labels = torch.LongTensor(batchdata.y).to(device)
    ###### graph_top vs graph_last
    readout_layers = get_readout_layers("mean")
    pooled = []
    for readout in readout_layers:
        pooled.append(readout(subtop.to(device), batch.to(device)))
    graph_top = torch.cat(pooled, dim=-1)
    pooled = []
    for readout in readout_layers:
        pooled.append(readout(sublast.to(device), batch.to(device)))
    graph_last = torch.cat(pooled, dim=-1)
    V1 = V1_loss(subtop, sublast, device)
    V2 = V2_loss(graph_top, labels.float(), graph_last, device)
    V3 = criterion(y_out, labels)
    r = 0.001
    ce = 0.001
    #loss = V3 + r * V1 - ce * V2
    loss =  V3
    return loss,y_out,labels,node_emd




