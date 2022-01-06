import os
import torch
import numpy as np
from partition_utils import get_partition_list
from functools import partial
from torch.utils.data import DataLoader
class ClusterIdIter(object):
    def __init__(self, dn, g, psize, batch_size, seed_nid):
        self.psize = psize
        self.batch_size = batch_size
        if dn:
            fn = os.path.join('./partitions/', dn + '/psize_{}.npy'.format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs('./partitions/', exist_ok=True)
                self.par_li = get_partition_list(g, psize)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(g, psize)
        par_list = []
        for p in self.par_li:
            par = torch.Tensor(p)
            par_list.append(par)
        self.par_list = par_list


    def __len__(self):
        return self.psize

    def __getitem__(self, idx):
        return self.par_li[idx]

def subgraph_collate_fn(g, batch):
    nids = np.concatenate(batch).reshape(-1).astype(np.int64)
    g1 = g.subgraph(nids)
    return g1

def ClusterIter(dn, g, psize, batch_size, seed_nid=None):
    cluster_iter_data = ClusterIdIter(
                        dn, g, psize, batch_size,None)
    cluster_iterator = DataLoader(cluster_iter_data, batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=partial(subgraph_collate_fn, g))
    return cluster_iterator
