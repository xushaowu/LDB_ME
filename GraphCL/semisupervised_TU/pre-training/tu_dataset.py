from torch_geometric.datasets import TUDataset
import torch
from itertools import repeat, product
import numpy as np
import numpy.matlib
import random
from scipy.spatial.distance import pdist
from pdb import set_trace as st

class TUDatasetExt(TUDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """
    
    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/' \
          'graphkerneldatasets'
    
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 processed_filename='data.pt',
                 aug="none", aug_ratio=None, npower=1):
        self.processed_filename = processed_filename
        self.aug = "none"
        self.aug_ratio = None
        self.npower = npower
        self.is_2th_drop = False
        self.center_loc = random.randint(0, 100) / 100
        super(TUDatasetExt, self).__init__(root, name, transform, pre_transform,
                                           pre_filter, use_node_attr)

    @property
    def processed_file_names(self):
        return self.processed_filename

    def get(self, idx):
        st()
        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        
        if self.aug == 'dropN':
            data = drop_nodes(data, self.aug_ratio)
        elif self.aug == 'dropNME':
            data = drop_nodes_me(data, self.aug_ratio, self.is_2th_drop)
        elif self.aug == 'dropNLDP':
            data = drop_nodes_ldp(data, self.aug_ratio, self.npower)
        elif self.aug == 'dropNMELDP':
            data = drop_nodes_meldp(data, 
                                    self.aug_ratio, 
                                    self.is_2th_drop, 
                                    self.npower, 
                                    self.center_loc, 
                                    self.aug)
        elif self.aug == 'wdropN':
            data = weighted_drop_nodes(data, self.aug_ratio, self.npower)
        elif self.aug == 'permE':
            data = permute_edges(data, self.aug_ratio)
        elif self.aug == 'subgraph':
            data = subgraph(data, self.aug_ratio)
        elif self.aug == 'maskN':
            data = mask_nodes(data, self.aug_ratio)
        elif self.aug == 'none':
            data = data
        elif self.aug == 'random4':
            ri = np.random.randint(4)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            elif ri == 2:
                data = permute_edges(data, self.aug_ratio)
            elif ri == 3:
                data = mask_nodes(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False

        elif self.aug == 'random3':
            ri = np.random.randint(3)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            elif ri == 2:
                data = permute_edges(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False


        elif self.aug == 'random2':
            ri = np.random.randint(2)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False

        else:
            #st()
            print('augmentation error')
            assert False

        # print(data)
        # print(self.aug)
        # assert False

        return data


def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:        
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


def drop_nodes_me(data, aug_ratio, is_2th_drop):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)
    #st()
    if is_2th_drop:
        #st()
        idx_list = np.array(range(0, node_num, 2))
        idx_comp = np.array(range(1, node_num, 2))
    else:
        idx_list = np.array(range(1, node_num, 2))
        idx_comp = np.array(range(0, node_num, 2))

    idx_perm = np.random.permutation(idx_list)

    idx_drop = idx_perm[:drop_num]
    
    idx_nondrop = np.concatenate((idx_perm[drop_num:], idx_comp), axis=0)
    
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


def drop_nodes_ldp(data, aug_ratio, npower):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()    
    drop_num    = int(node_num  * aug_ratio)

    adj = np.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    deg = adj.sum(axis=1)
    deg[deg==0] = 0.1
    deg = deg ** (npower)

    c_index = random.randint(0, node_num - 1)
    weight  = np.abs(deg - deg[c_index])    
    weight  = (weight.max() - weight) / (weight.max() - weight.min())
    
    #idx_drop = np.random.choice(node_num, drop_num, replace=False, p=weight / weight.sum())
    #idx_nondrop = np.array([n for n in range(node_num) if not n in idx_drop])
    
    try:
        idx_drop = np.random.choice(node_num, drop_num, replace=False, p=weight / weight.sum())
        idx_nondrop = np.array([n for n in range(node_num) if not n in idx_drop])
    except:
        #print("warning: choice failed.")
        idx_sorted_w = np.argsort(weight)
        idx_nondrop  = idx_sorted_w[drop_num:]


    # idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    ###
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


def drop_nodes_meldp(data, aug_ratio, is_2th_drop, npower, center_loc):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()    
    drop_num    = int(node_num  * aug_ratio)

    adj = np.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    deg = adj.sum(axis=1)
    deg[deg==0] = 0.1
    deg = deg ** (npower)

    if is_2th_drop:
        idx_list = np.array(range(0, node_num, 2))
        idx_comp = np.array(range(1, node_num, 2))
    else:
        idx_list = np.array(range(1, node_num, 2))
        idx_comp = np.array(range(0, node_num, 2))

    center = int(np.floor((node_num - 1) * center_loc))
    weight = np.abs(deg - deg[center])

    idx_sorted_w = np.argsort(weight)
    idx_selected = idx_sorted_w[idx_list]
    sorted_w     = weight[idx_selected]
    sorted_w     = (sorted_w.max() - sorted_w) / (sorted_w.max() - sorted_w.min())

    try:
        idx_drop = np.random.choice(idx_selected.shape[0], drop_num, replace=False, p=sorted_w / sorted_w.sum())
        idx_drop = idx_selected[idx_drop]
    except:
        idx_drop = idx_selected[:drop_num]
    
    idx_nondrop = np.array([n for n in idx_selected if not n in idx_drop])
    idx_nondrop = np.concatenate((idx_nondrop, idx_sorted_w[idx_comp]), axis=0)
    
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    ###
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
        
    return data


def weighted_drop_nodes(data, aug_ratio, npower):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    adj = np.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    deg = adj.sum(axis=1)
    deg[deg==0] = 0.1
    # print(deg)
    # deg = deg ** (-1)
    deg = deg ** (npower)
    # print(deg)
    # print(deg / deg.sum())
    # assert False

    idx_drop = np.random.choice(node_num, drop_num, replace=False, p=deg / deg.sum())

    # idx_perm = np.random.permutation(node_num)
    # idx_drop = idx_perm[:drop_num]
    # idx_nondrop = idx_perm[drop_num:]

    idx_nondrop = np.array([n for n in range(node_num) if not n in idx_drop])

    # idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    ###
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    ###
    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))

    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add

    edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)

    return data

def subgraph(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    data.x = data.x[idx_nondrop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[list(range(node_num)), list(range(node_num))] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    data.edge_index = edge_index

    return data

def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)

    return data

def distance(seq1, seq2, type='dtw'):

    if type == 'dtw':
        #st()

        def d_value(a, b):
            return max(a, b) / min(a, b) - 1

        M, N = np.shape(seq1)[1], np.shape(seq2)[1]

        cost = np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d_value(seq1[:,0], seq2[:,0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d_value(seq1[:,i], seq2[:,0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d_value(seq1[:,0], seq2[:,j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(1, N):
                choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d_value(seq1[:,i], seq2[:,j])

        # Return DTW distance given window 
        return cost[-1, -1]

    else:
        if type == 'manhattan' or type == 'mah':
            type = 'cityblock'

        if type == 'che':
            type = 'chebyshev'
            
        if type == 'eu':
            type = 'euclidean'

        if type == 'seu':
            type = 'seuclidean'

        return pdist(np.vstack([seq1, seq2]), type)

def getDistances(dgs, center, type='dtw'):
    node_num  = dgs.shape[0]
    distances = np.zeros([1, node_num])

    if type == 'mahalanobis':
        #st()
        dgsT = dgs.T
        SI   = np.linalg.inv(np.cov(dgs)) #the inverse of the covariance matrix

        for i in range(node_num):
            delta = dgsT[center] - dgsT[i]
            distances[0, i] = np.sqrt(np.dot(np.dot(delta, SI), delta.T)) if i != center else 0
    else:
        for i in range(node_num):
            distances[0, i] = distance(dgs[center, :], dgs[i, :], type=type) if i != center else 0

    return distances
