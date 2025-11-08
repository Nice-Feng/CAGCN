import numpy as np
import torch


def edge_twosampels(edge):
    ##
    edge_top = np.zeros_like(edge)
    edge_last = np.zeros_like(edge)
    for e in range(0, edge.shape[3]):
        for l in range(0, edge.shape[0]):
            tmp_edge = edge[l, :, :, e]
            np.fill_diagonal(tmp_edge, 0)
            ###########################################
            flattened_matrix = tmp_edge.flatten()
            # 获取前450个最大的值的索引
            top_450_indices = np.argsort(flattened_matrix)[-100:]
            # 将这些索引转换回二维索引（行, 列）
            top_450_coords = np.unravel_index(top_450_indices, tmp_edge.shape)
            # 创建一个新的矩阵，初始化为 0
            top_new_matrix = np.zeros_like(tmp_edge)
            top_new_matrix[top_450_coords] = 1
            ###################################################
            # 获取后450个最大的值的索引
            tmp_edge = edge[l, :, :, e]
            np.fill_diagonal(tmp_edge, 10)
            flattened_matrix = tmp_edge.flatten()
            last_450_indices = np.argsort(flattened_matrix)[1:100]
            # 将这些索引转换回二维索引（行, 列）

            last_450_coords = np.unravel_index(last_450_indices, tmp_edge.shape)
            # 创建一个新的矩阵，初始化为 0
            last_new_matrix = np.zeros_like(tmp_edge)
            # 将前300个最大值的位置置为 1
            last_new_matrix[last_450_coords] = 1

            edge_top[l, :, :, e] = top_new_matrix
            edge_last[l, :, :, e] = last_new_matrix

    return edge_top, edge_last


