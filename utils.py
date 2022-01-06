#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:17:08 2021

@author: gj
"""
import dgl
from dgl.data import  load_data
import torch
from ogb.nodeproppred import DglNodePropPredDataset

#%%

class Data_info():
    in_feats = None
    n_classes = None
def load_graph_data(args):
    if args.dataset=='reddit':
        data = load_data(args)
        g = data[0]
        g = g.remove_self_loop()
        g = g.add_self_loop()
    else:
        if args.dataset=='mag':
            data = DglNodePropPredDataset(name='ogbn-mag')
            splitted_idx = data.get_idx_split()
            train_idx, val_idx, test_idx = splitted_idx['train']["paper"], splitted_idx['valid']["paper"], splitted_idx['test']["paper"]
            g, labels = data[0]
            g = g.node_type_subgraph(["paper"])
            g = dgl.to_bidirected(g,copy_ndata=True)
            labels = labels['paper']
        elif args.dataset=='arxiv':
            data = DglNodePropPredDataset(name='ogbn-arxiv')
            splitted_idx = data.get_idx_split()
            train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
            g, labels = data[0]
            g = dgl.to_bidirected(g,copy_ndata=True)
        elif args.dataset=='products':
            data = DglNodePropPredDataset(name='ogbn-products')
            splitted_idx = data.get_idx_split()
            train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
            g, labels = data[0]
        g = g.remove_self_loop()
        g = g.add_self_loop()
        num_nodes = g.number_of_nodes()
        g.ndata['label'] = labels[:, 0]
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[train_idx] = True
        g.ndata['train_mask'] = mask
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[val_idx] = True
        g.ndata['val_mask'] = mask
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[test_idx] = True
        g.ndata['test_mask'] = mask
    
    data_info = Data_info()
    data_info.in_feats = g.ndata['feat'].shape[1]
    data_info.n_classes = data.num_classes
    return g, data_info

def get_random_train_mask(n, n_train, val_mask, test_mask):
    index = torch.arange(n)
    pool = index[~val_mask & ~test_mask]
    chosen = pool[torch.randperm(pool.shape[0])[:n_train]]
    train_mask = torch.zeros(val_mask.shape,dtype=bool).cuda()
    train_mask[chosen] = True
    return train_mask

def get_degree_train_mask(g, need):
    d = g.in_degrees()
    can_choose_mask = ~g.ndata['val_mask'] & ~g.ndata['test_mask']
    pool_ids = torch.arange(g.ndata['test_mask'].shape[0])[can_choose_mask]
    d_left = d[can_choose_mask]
    chosen = d_left.topk(need)[1]
    chosen_id = pool_ids[chosen]
    train_mask = torch.zeros(g.ndata['test_mask'].shape, dtype=bool)
    train_mask[chosen_id] = True
    return train_mask

















