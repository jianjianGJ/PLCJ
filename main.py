#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:36:27 2021

@author: gj
"""
import os
import argparse
import torch
from models import DGIclassifier,get_PLCJ_train_mask
from utils import load_graph_data,get_random_train_mask,get_degree_train_mask
#%%
#python main.py --dataset=arxiv --need=500 --n-run=1 --mode=PLCJ --gpu=0
argparser = argparse.ArgumentParser("Test",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--dataset", type=str, default='arxiv')
argparser.add_argument("--need", type=int, default=500)
argparser.add_argument("--n-run", type=int, help="running times", default=1)
argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
argparser.add_argument("--mode", type=str, default='PLCJ')
args = argparser.parse_args()
# class Args:
#     dropout = 0.
#     gpu = -1
#     dgi_lr = 1e-3
#     classifier_lr = 1e-2
#     n_dgi_epochs = 300
#     n_classifier_epochs = 300
#     n_hidden = 512
#     n_layers = 1
#     weight_decay = 0.
#     patience = 20
#     self_loop = True
# args = Args()
args.dropout = 0.
args.dgi_lr = 1e-3
args.classifier_lr = 1e-2
args.cluster_lr = 0.1
args.n_hidden = 128
args.n_layers = 1
args.weight_decay = 0.0
args.patience = 20
args.self_loop = True
args.report = True
args.gpu = 0
args.n_dgi_epochs = 1000
args.n_cluster_epochs = 300
args.n_classifier_epochs = 300
batch_size_dict = {'arxiv':40, 'reddit':20, 'mag':40, 'products':40}
psize_dict = {'arxiv':300, 'reddit':1500, 'mag':800, 'products':800}
args.batch_size  = batch_size_dict[args.dataset]
args.psize = psize_dict[args.dataset]
torch.cuda.set_device(args.gpu)

g, graph_info = load_graph_data(args)
args.n_classes = graph_info.n_classes

if not os.path.exists(f'./partitions/{args.dataset}/'):
    os.makedirs(f'./partitions/{args.dataset}/')
if not os.path.exists(f'./model_save/{args.dataset}/'):
    os.makedirs(f'./model_save/{args.dataset}/')
#%%

for i in range(args.n_run):
    model = DGIclassifier(graph_info, args)
    if args.mode=='rand': 
        g.ndata['train_mask'] = get_random_train_mask(g.num_nodes(), 
                                                      args.need, g.ndata['val_mask'], g.ndata['test_mask']).cpu()
        print('Done! Get '+str(g.ndata['train_mask'].sum().item())+' training nodes.')
        args.batch_size  = batch_size_dict[args.dataset]
        args.psize = psize_dict[args.dataset]
        model.train_ori_classifier_batch(g, args)
    elif args.mode=='degree':
        g.ndata['train_mask'] = get_degree_train_mask(g, args.need).cpu()
        print('Done! Get '+str(g.ndata['train_mask'].sum().item())+' training nodes.')
        args.batch_size  = batch_size_dict[args.dataset]
        args.psize = psize_dict[args.dataset]
        model.train_ori_classifier_batch(g, args)
    elif args.mode=='PLCJ':
        if not os.path.exists('./model_save/'+args.dataset+'/best_dgi.pkl'):
            print('Training DGI...')
            model.train_dgi_batch(g, args)
        else:
            model.load_state_dict(torch.load('./model_save/'+args.dataset+'/best_dgi.pkl'))
        g.ndata['dgi_feat'] = model.get_embedding_batch(g, args).cpu()
        args.batch_size  = 4
        args.psize = int(args.need)
        args.n_clusters = 4
        print('Getting training nodes...')
        g.ndata['train_mask']  = get_PLCJ_train_mask(g, args)
        print('Done! Get '+str(g.ndata['train_mask'].sum().item())+' training nodes.')
        args.batch_size  = batch_size_dict[args.dataset]
        args.psize = psize_dict[args.dataset]
        print('Evaluating..')
        model.train_classifier_batch(g, args)
    acc,mif1 = model.evaluate_batch(g, 'test_mask',args)
    print(args.dataset, ' args.need:',args.need,'run id:',i,'acc:',acc)



