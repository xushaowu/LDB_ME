import copy
import random
import numpy as np
import torch
from torch_geometric.utils import degree, to_undirected, to_dense_adj, dense_to_sparse
from torch.utils.data import WeightedRandomSampler

from pGRACE.utils import compute_pr, eigenvector_centrality

from pdb import set_trace as st

def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[drop_mask] = 0.

    return x


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    # numpy.where(cond, value): elements who unsatisfied 'cond' in this array will be changed to 'value'
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    #st()
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c # @ means matrix multiplication
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    #st()
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    # sel_mask = (1-torch.bernoulli(1. - edge_weights)).to(torch.bool)

    return edge_index[:, sel_mask]


def drop_edge_weighted_ldp(edge_index, edge_weights, p: float, threshold: float = 1.):

    edge_num   = edge_weights.shape[0]
    center_ind = random.randint(0, edge_num - 1)

    dis_values = torch.abs(edge_weights - edge_weights[center_ind])
    dis_values = (dis_values.max() - dis_values) / (dis_values.max() - dis_values.mean())

    dis_values = dis_values / dis_values.mean() * p
    dis_values = dis_values.where(dis_values < threshold, torch.ones_like(dis_values) * threshold)
    
    sel_mask = torch.bernoulli(1. - dis_values).to(torch.bool)

    return edge_index[:, sel_mask]


def drop_edge_weighted_me(edge_index, edge_weights, p: float, threshold: float = 1.):
    
    def get_save_list(start_ind):
        sel_ind = sorted_dis_ind[start_ind::2]
        sel_val = edge_weights[sel_ind]
                
        sel_val = sel_val / sel_val.mean() * (p[start_ind] * 2)
        sel_val = sel_val.where(sel_val < threshold, torch.ones_like(sel_val) * threshold)
        
        sel_mask = torch.bernoulli(1. - sel_val).to(torch.bool)

        return torch.cat([sel_ind[sel_mask], sorted_dis_ind[(1-start_ind)::2]])
    
    _, sorted_dis_ind = edge_weights.sort()

    sel_list_1 = get_save_list(0)

    sel_list_2 = get_save_list(1)
    
    return edge_index[:, sel_list_1], edge_index[:, sel_list_2]


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)


    #pv_row = pv[edge_index[0]].to(torch.float32)
    s_row = pv[edge_index[0]].to(torch.float32)
    #pv_col = pv[edge_index[1]].to(torch.float32)
    s_col = pv[edge_index[1]].to(torch.float32)
    
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    #st()
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())


def drop_node_weighted_ldp(feature, edge_index, node_weights, p: float, threshold: float = 1.):

    def drop_zero_by_soft(probs):
        zero_ind = probs == 0
        zero_num = torch.sum(zero_ind)

        soft_value = probs[-(zero_num + 1)] / (zero_num + 1)

        probs[-(zero_num + 1):] = soft_value

        return probs


    def delete_row_col(input_matrix, drop_list, only_row=False):

        remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]

        out = input_matrix[remain_list, :]

        return out if only_row else out[:, remain_list]


    def get_list(start_ind):
        sel_ind = sorted_dis_ind[start_ind::2].cpu()
        sel_val = dis_values[sel_ind]
        
        prob     = torch.sigmoid(sel_val)        
        sel_mask = torch.bernoulli(prob).to(torch.bool)

        return sel_ind[sel_mask]
        '''
        if torch.sum(sel_mask) < drop_num[start_ind]:
            return sel_ind[sel_mask]
        else:
            return sel_ind[sel_mask][:drop_num[start_ind]]
        
        '''

    
    input_adj = to_dense_adj(edge_index)[0]
    input_fea = feature

    second_ajd = copy.deepcopy(input_adj)
    second_fea = copy.deepcopy(input_fea)

    node_num = input_fea.shape[0]
    drop_num = [int(node_num * p[0]), int(node_num * p[1])]

    center_ind = random.randint(0, node_num - 1)
    dis_values = torch.abs(node_weights - node_weights[center_ind])

    _, sorted_dis_ind = torch.abs(dis_values).sort() # for locate
    
    drop_node_list_1st = get_list(0)

    aug_input_fea_1st = delete_row_col(input_fea, drop_node_list_1st, only_row=True)
    aug_input_adj_1st = delete_row_col(input_adj, drop_node_list_1st)
    
    drop_node_list_2nd = get_list(1)

    aug_input_fea_2nd = delete_row_col(second_fea, drop_node_list_2nd, only_row=True)
    aug_input_adj_2nd = delete_row_col(second_ajd, drop_node_list_2nd)
    #st()
    return [aug_input_fea_1st, aug_input_fea_2nd], \
           [dense_to_sparse(aug_input_adj_1st)[0], dense_to_sparse(aug_input_adj_2nd)[0]]


def degree_drop_node_weights(edge_index):

    return degree(to_undirected(edge_index)[1])


def pr_drop_node_weights(edge_index, aggr: str = 'sink', k: int = 10):

    return compute_pr(edge_index, k=k)


def evc_drop_node_weights(data):

    return eigenvector_centrality(data)
