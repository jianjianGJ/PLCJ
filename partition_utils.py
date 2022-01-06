# from time import time

import numpy as np

# from utils import arg_list

from dgl.transform import metis_partition
from dgl import backend as F
import dgl

def get_partition_list(g, psize):
    g = g.cpu()
    p_gs = metis_partition(g, psize)#partition dict
    graphs = []
    for k, val in p_gs.items():
        nids = val.ndata[dgl.NID]
        nids = F.asnumpy(nids)
        graphs.append(nids)
    return graphs#[npy(1,2,3),npy(5,9,0),...]

